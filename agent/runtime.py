from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import openai
from dotenv import dotenv_values

from agent.schema import coerce_structured_output, describe_output_schema, parse_output_schema

TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_CALL_LINE_RE = re.compile(r"^\s*TOOL_CALL:\s*(.+?)\s*$", re.MULTILINE)
FINAL_XML_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    command: str


@dataclass(frozen=True)
class ToolResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class PlannerDecision:
    executor_name: str
    reason: str


@dataclass(frozen=True)
class RunResult:
    executor_name: str
    planner_reason: str
    output: Any
    output_schema: str
    raw_response: str
    tool_results: tuple[ToolResult, ...] = ()


@dataclass
class ExecutorContext:
    user_input: str
    executor: dict[str, Any]
    tool_results: list[ToolResult] = field(default_factory=list)

    @property
    def task(self) -> dict[str, Any]:
        return self.executor["task"]


class ModelClient(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any],
    ) -> str: ...


class ShellToolExecutor:
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def execute(self, command: str) -> ToolResult:
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        return ToolResult(
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )


class HeuristicPlanner:
    def choose_executor(self, user_input: str, executors: list[dict[str, Any]]) -> PlannerDecision:
        best_executor = None
        best_score = -1

        user_tokens = _tokenize(user_input)
        for executor in executors:
            task = executor["task"]
            score = 0
            score += _overlap_score(user_tokens, _tokenize(task["name"])) * 5
            score += _overlap_score(user_tokens, _tokenize(task["input_description"])) * 4
            score += _overlap_score(user_tokens, _tokenize(task["behavior"])) * 3
            for example in task["examples"]:
                score += _overlap_score(user_tokens, _tokenize(example["input"]))
            if score > best_score:
                best_score = score
                best_executor = executor

        chosen = best_executor or executors[0]
        return PlannerDecision(
            executor_name=chosen["name"],
            reason=f"Selected executor '{chosen['name']}' using heuristic task matching.",
        )


class ExampleDrivenModel:
    """
    Deterministic fallback used when no external model client is wired in.

    It picks the nearest example, emits its commands as tool calls one-by-one,
    then synthesizes a final answer from the gathered tool output.
    """

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}

    def generate(self, *, system_prompt: str, user_prompt: str, context: dict[str, Any]) -> str:
        del system_prompt, user_prompt
        task = context["executor"]["task"]
        user_input = context["user_input"]
        key = context["executor"]["name"]

        if key not in self._sessions:
            example = _select_best_example(user_input, task["examples"])
            self._sessions[key] = {
                "example": example,
                "pending_commands": list(example["commands"]) if example else [],
            }

        session = self._sessions[key]
        pending = session["pending_commands"]
        if pending:
            command = pending.pop(0)
            return f"<tool_call>{command}</tool_call>"

        example = session["example"]
        if example is not None:
            return f"<final>{example['output']}</final>"

        return _build_fallback_final_response(context)


class OpenAIChatModel:
    def __init__(self, *, api_key: str, base_url: str | None = None, timeout_seconds: int = 60):
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def generate(self, *, system_prompt: str, user_prompt: str, context: dict[str, Any]) -> str:
        model_name = context["executor"]["llm"]
        completion = self._client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content


class DSPyExampleModel:
    def __init__(self, *, api_key: str, base_url: str | None = None, timeout_seconds: int = 60):
        import dspy

        self._dspy = dspy
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._predictors: dict[str, Any] = {}
        self._lms: dict[str, Any] = {}

    def generate(self, *, system_prompt: str, user_prompt: str, context: dict[str, Any]) -> str:
        executor = context["executor"]
        executor_name = executor["name"]
        predictor = self._predictors.get(executor_name)
        if predictor is None:
            predictor = self._dspy.Predict("system_prompt, user_prompt -> response")
            predictor.demos = _build_dspy_demos(self._dspy, executor)
            self._predictors[executor_name] = predictor

        lm = self._lms.get(executor["llm"])
        if lm is None:
            lm_kwargs: dict[str, Any] = {
                "model": f"openai/{executor['llm']}",
                "api_key": self.api_key,
                "cache": False,
                "num_retries": 2,
                "timeout": self.timeout_seconds,
            }
            if self.base_url:
                lm_kwargs["api_base"] = self.base_url
                lm_kwargs["base_url"] = self.base_url
            lm = self._dspy.LM(**lm_kwargs)
            self._lms[executor["llm"]] = lm

        prediction = predictor(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            lm=lm,
        )
        return prediction.response


class AgentSystemRuntime:
    def __init__(
        self,
        system_spec: dict[str, Any],
        *,
        model_client: ModelClient | None = None,
        tool_executor: ShellToolExecutor | None = None,
        planner: HeuristicPlanner | None = None,
        max_tool_round_trips: int = 6,
        source_model_path: str | None = None,
        require_provider: bool = False,
        use_dspy: bool = False,
    ):
        self.system_spec = system_spec
        self.source_model_path = source_model_path
        self.model_client = model_client or create_model_client(
            source_model_path=source_model_path,
            require_provider=require_provider,
            use_dspy=use_dspy,
        )
        self.tool_executor = tool_executor or ShellToolExecutor()
        self.planner = planner or HeuristicPlanner()
        self.max_tool_round_trips = max_tool_round_trips

    def run(self, user_input: str) -> RunResult:
        decision = self.planner.choose_executor(user_input, self.system_spec["executors"])
        executor = self._find_executor(decision.executor_name)
        context = ExecutorContext(user_input=user_input, executor=executor)

        raw_response = ""
        for _ in range(self.max_tool_round_trips + 1):
            response = self.model_client.generate(
                system_prompt=build_executor_system_prompt(executor),
                user_prompt=build_executor_user_prompt(context),
                context=_context_dict(context),
            )
            raw_response = response
            tool_call = extract_tool_call(response)
            if tool_call is None:
                return self._finalize_response(decision, executor, response, context.tool_results)

            context.tool_results.append(self.tool_executor.execute(tool_call.command))

        raise RuntimeError("Agent exceeded the maximum number of tool round trips.")

    def _finalize_response(
        self,
        decision: PlannerDecision,
        executor: dict[str, Any],
        response: str,
        tool_results: list[ToolResult],
    ) -> RunResult:
        final_text = extract_final_output(response)
        schema = parse_output_schema(executor["task"]["output_schema"])
        output: Any = final_text

        if schema.is_structured:
            parsed_json = json.loads(final_text)
            output = coerce_structured_output(parsed_json, schema)

        return RunResult(
            executor_name=decision.executor_name,
            planner_reason=decision.reason,
            output=output,
            output_schema=executor["task"]["output_schema"],
            raw_response=response,
            tool_results=tuple(tool_results),
        )

    def _find_executor(self, executor_name: str) -> dict[str, Any]:
        for executor in self.system_spec["executors"]:
            if executor["name"] == executor_name:
                return executor
        raise KeyError(f"Unknown executor name: {executor_name}")


def build_executor_system_prompt(executor: dict[str, Any]) -> str:
    task = executor["task"]
    rules = "\n".join(_format_rule(rule) for rule in executor["rules"]) or "None"
    skills = "\n".join(_format_skill(skill) for skill in task["skills"]) or "None"
    examples = "\n\n".join(_format_example(example) for example in task["examples"]) or "None"
    schema = parse_output_schema(task["output_schema"])
    schema_instruction = describe_output_schema(schema)

    return (
        f"You are {executor['persona']}.\n"
        f"Task: {task['name']}\n"
        f"Behavior: {task['behavior']}\n"
        f"Rules:\n{rules}\n\n"
        f"Available shell skills:\n{skills}\n\n"
        f"Output contract: {schema_instruction}\n\n"
        "When you need a terminal command, respond with exactly one tool call "
        "using either <tool_call>command</tool_call> or TOOL_CALL: command.\n"
        "When you are done, return the final answer directly or wrap it in "
        "<final>...</final>.\n\n"
        f"Examples:\n{examples}"
    )


def build_executor_user_prompt(context: ExecutorContext) -> str:
    tool_history = "\n\n".join(_format_tool_result(result) for result in context.tool_results)
    if not tool_history:
        tool_history = "No tool calls yet."
    return f"User request:\n{context.user_input}\n\n" f"Tool history:\n{tool_history}"


def extract_tool_call(response_text: str) -> ToolCall | None:
    xml_match = TOOL_CALL_XML_RE.search(response_text)
    if xml_match:
        command = xml_match.group(1).strip()
        if command:
            return ToolCall(command=command)

    line_match = TOOL_CALL_LINE_RE.search(response_text)
    if line_match:
        command = line_match.group(1).strip()
        if command:
            return ToolCall(command=command)

    return None


def extract_final_output(response_text: str) -> str:
    final_match = FINAL_XML_RE.search(response_text)
    if final_match:
        return final_match.group(1).strip()
    return response_text.strip()


def load_system_spec(spec_json: str) -> dict[str, Any]:
    return json.loads(spec_json)


def create_model_client(
    *,
    source_model_path: str | None = None,
    require_provider: bool = False,
    use_dspy: bool = False,
) -> ModelClient:
    config = load_openai_config(source_model_path=source_model_path)
    if config is not None:
        model_cls = DSPyExampleModel if use_dspy else OpenAIChatModel
        return model_cls(
            api_key=config["OPENAI_API_KEY"],
            base_url=config.get("OPENAI_BASE_URL"),
        )

    if require_provider:
        raise RuntimeError(missing_provider_message(source_model_path))

    return ExampleDrivenModel()


def has_openai_provider_config(source_model_path: str | None = None) -> bool:
    return load_openai_config(source_model_path=source_model_path) is not None


def missing_provider_message(source_model_path: str | None = None) -> str:
    locations = ", ".join(str(path) for path in candidate_dotenv_paths(source_model_path))
    return (
        "No OpenAI provider configuration found. Set OPENAI_API_KEY "
        f"in the environment or in a .env file. Looked for .env in: {locations}"
    )


def load_openai_config(source_model_path: str | None = None) -> dict[str, str] | None:
    file_values: dict[str, str] = {}
    for path in candidate_dotenv_paths(source_model_path):
        if path.exists():
            file_values.update(dotenv_values(path))

    api_key = os.environ.get("OPENAI_API_KEY") or file_values.get("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.environ.get("OPENAI_BASE_URL") or file_values.get("OPENAI_BASE_URL")
    config: dict[str, str] = {"OPENAI_API_KEY": api_key}
    if base_url:
        config["OPENAI_BASE_URL"] = base_url
    return config


def candidate_dotenv_paths(source_model_path: str | None = None) -> list[Path]:
    candidates = [Path.cwd() / ".env"]
    if source_model_path:
        model_dir = Path(source_model_path).resolve().parent
        candidates.append(model_dir / ".env")
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def _context_dict(context: ExecutorContext) -> dict[str, Any]:
    return {
        "user_input": context.user_input,
        "executor": context.executor,
        "tool_results": [
            {
                "command": result.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }
            for result in context.tool_results
        ],
    }


def _build_fallback_final_response(context: dict[str, Any]) -> str:
    task = context["executor"]["task"]
    tool_results = context["tool_results"]
    schema = parse_output_schema(task["output_schema"])

    if schema.is_structured:
        payload = {}
        for field in schema.fields:
            payload[field.name] = _default_structured_value(field.type_name, tool_results)
        return json.dumps(payload, indent=2)

    if not tool_results:
        return "<final>No commands were required for this task.</final>"

    sections = []
    for result in tool_results:
        stdout = result["stdout"].strip()
        stderr = result["stderr"].strip()
        body = stdout or stderr or f"Command exited with code {result['exit_code']}"
        sections.append(f"$ {result['command']}\n{body}")

    if task["output_schema"].lower() == "markdown":
        content = "\n\n".join(f"```text\n{section}\n```" for section in sections)
        return f"<final>{content}</final>"

    return "<final>" + "\n\n".join(sections) + "</final>"


def _default_structured_value(type_name: str, tool_results: list[dict[str, Any]]) -> Any:
    if type_name in {"str", "string"}:
        if not tool_results:
            return ""
        return tool_results[-1]["stdout"].strip() or tool_results[-1]["stderr"].strip()
    if type_name == "int":
        return len(tool_results)
    if type_name == "float":
        return float(len(tool_results))
    if type_name == "bool":
        return any(result["exit_code"] == 0 for result in tool_results)
    if type_name.startswith("list["):
        return [result["command"] for result in tool_results]
    raise ValueError(f"Unsupported structured output type: {type_name}")


def _format_rule(rule: dict[str, Any]) -> str:
    prefix = "DON'T" if rule["negative"] else "DO"
    return f"- {prefix}: {rule['description']}"


def _format_skill(skill: dict[str, Any]) -> str:
    args = ", ".join(arg["name"] for arg in skill["arguments"]) or "no arguments"
    return f"- {skill['name']}: {skill['description']} (command: {skill['command']}; args: {args})"


def _format_example(example: dict[str, Any]) -> str:
    commands = ", ".join(example["commands"]) or "no commands"
    return f"Input: {example['input']}\n" f"Commands: {commands}\n" f"Output: {example['output']}"


def _build_dspy_demos(dspy_module, executor: dict[str, Any]) -> list[Any]:
    examples = []
    for example in executor["task"]["examples"]:
        tool_history = "No tool calls yet."
        user_prompt = f"User request:\n{example['input']}\n\nTool history:\n{tool_history}"
        if example["commands"]:
            response = f"<tool_call>{example['commands'][0]}</tool_call>"
        else:
            response = f"<final>{example['output']}</final>"
        examples.append(
            dspy_module.Example(
                system_prompt=build_executor_system_prompt(executor),
                user_prompt=user_prompt,
                response=response,
            ).with_inputs("system_prompt", "user_prompt")
        )
    return examples


def _format_tool_result(result: ToolResult) -> str:
    chunks = [f"$ {result.command}", f"exit_code: {result.exit_code}"]
    if result.stdout.strip():
        chunks.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        chunks.append(f"stderr:\n{result.stderr.strip()}")
    return "\n".join(chunks)


def _select_best_example(user_input: str, examples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not examples:
        return None
    user_tokens = _tokenize(user_input)
    return max(examples, key=lambda example: _overlap_score(user_tokens, _tokenize(example["input"])))


def _overlap_score(left: set[str], right: set[str]) -> int:
    return len(left & right)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 1}


def command_to_argv(command: str) -> list[str]:
    return shlex.split(command)
