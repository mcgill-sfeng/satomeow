from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, RunConfig, Runner, handoff
from agents.items import ToolCallOutputItem
from agents.lifecycle import RunHooksBase
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run_context import RunContextWrapper
from dotenv import dotenv_values
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model

from agent.schema import coerce_structured_output, describe_output_schema, parse_output_schema

_PARAM_PATTERN = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True)
class RunResult:
    executor_name: str
    planner_reason: str
    output: Any
    output_format: str
    raw_response: str
    system_prompt: str
    user_input: str
    raw_responses: tuple[str, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


# ---------------------------------------------------------------------------
# Shell tool executor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Handoff-based planner internals
# ---------------------------------------------------------------------------


class _HandoffCtx(BaseModel):
    """Structured payload the LLM passes when invoking a transfer function."""

    reason: str


class _RoutingHooks(RunHooksBase):
    """Captures which executor was selected and why during a multi-agent run."""

    def __init__(self) -> None:
        self.executor_name: str | None = None
        self.planner_reason: str = "Direct execution (single executor)."

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        self.executor_name = to_agent.name


def build_planner_prompt(executors: list[dict[str, Any]]) -> str:
    lines = [
        "You are a task router for a multi-agent system.",
        "Your only job is to immediately transfer the user's request to the most appropriate agent.",
        "Do not answer the user directly — always call a transfer function.",
        "",
        "Agents:",
    ]
    for executor in executors:
        task = executor.get("task") or {}
        lines.append(
            f"- {executor['name']}: {executor.get('persona', '')}. "
            f"Handles: {task.get('input_description', 'general tasks')}. "
            f"Produces: {_format_output_spec(task)}."
        )
    lines += [
        "",
        "When transferring, include a concise reason explaining why this agent is the best fit.",
    ]
    return "\n".join(lines)


def build_planner_agent(
    executors: list[dict[str, Any]],
    executor_agents: dict[str, Agent[Any]],
    hooks: _RoutingHooks,
    planner_llm: str,
) -> Agent[Any]:
    """Build a static routing Agent whose only role is to hand off to the right executor."""

    def _make_on_handoff(name: str):
        async def _cb(_ctx: RunContextWrapper[Any], args: _HandoffCtx) -> None:
            hooks.planner_reason = args.reason

        return _cb

    handoffs = [
        handoff(
            executor_agents[executor["name"]],
            on_handoff=_make_on_handoff(executor["name"]),
            input_type=_HandoffCtx,
        )
        for executor in executors
    ]

    return Agent(
        name="Planner",
        instructions=build_planner_prompt(executors),
        model=planner_llm,
        handoffs=handoffs,
    )


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class AgentSystemRuntime:
    def __init__(
        self,
        system_spec: dict[str, Any],
        *,
        tool_executor: ShellToolExecutor | None = None,
        source_model_path: str | None = None,
        require_provider: bool = False,
        use_dspy: bool = False,
    ):
        self.system_spec = system_spec
        self.tool_executor = tool_executor or ShellToolExecutor()
        self.source_model_path = source_model_path
        self.require_provider = require_provider
        self.use_dspy = use_dspy

    def run(self, user_input: str) -> RunResult:
        executors = self.system_spec["executors"]
        run_config = self._build_run_config()
        hooks = _RoutingHooks()

        if len(executors) == 1:
            executor = executors[0]
            hooks.executor_name = executor["name"]
            agent = build_openai_agent(executor, tool_executor=self.tool_executor, use_dspy=self.use_dspy)
            sdk_result = Runner.run_sync(agent, user_input, run_config=run_config, hooks=hooks)
        else:
            executor_agents = {
                e["name"]: build_openai_agent(e, tool_executor=self.tool_executor, use_dspy=self.use_dspy)
                for e in executors
            }
            planner_llm = self.system_spec.get("planner", {}).get("llm") or executors[0]["llm"]
            planner = build_planner_agent(executors, executor_agents, hooks, planner_llm)
            sdk_result = Runner.run_sync(planner, user_input, run_config=run_config, hooks=hooks)

            if hooks.executor_name is None:
                # Planner did not hand off — fall back to first executor gracefully.
                hooks.executor_name = executors[0]["name"]
                hooks.planner_reason = "Planner did not hand off; defaulted to first executor."

        executor = self._find_executor(hooks.executor_name)
        return self._finalize_result(hooks, executor, sdk_result, user_input)

    def _build_run_config(self) -> RunConfig:
        config = load_openai_config(source_model_path=self.source_model_path)
        if config is None:
            raise RuntimeError(missing_provider_message(self.source_model_path))

        client = AsyncOpenAI(
            api_key=config["OPENAI_API_KEY"],
            base_url=config.get("OPENAI_BASE_URL"),
        )
        planner_llm = self.system_spec.get("planner", {}).get("llm") or "gpt-4o-mini"
        return RunConfig(
            model=OpenAIChatCompletionsModel(planner_llm, openai_client=client),
            tracing_disabled=True,
            workflow_name=f"{self.system_spec.get('planner', {}).get('persona', 'agent')} runtime",
        )

    def _finalize_result(
        self,
        hooks: _RoutingHooks,
        executor: dict[str, Any],
        sdk_result: Any,
        user_input: str,
    ) -> RunResult:
        system_prompt = build_executor_system_prompt(executor, use_dspy=self.use_dspy)
        task = executor["task"]
        schema = parse_output_schema(
            output_format=task["output_format"],
            output_fields=task["output_fields"],
        )
        raw_responses = tuple(_dump_raw_response(response) for response in sdk_result.raw_responses)
        output = sdk_result.final_output
        if hasattr(output, "model_dump"):
            output = output.model_dump()
        if schema.is_structured:
            output = coerce_structured_output(output, schema)
        tool_results = tuple(_extract_tool_results(sdk_result.new_items))
        raw_response = raw_responses[-1] if raw_responses else _stringify_output(output)
        return RunResult(
            executor_name=hooks.executor_name,
            planner_reason=hooks.planner_reason,
            output=output,
            output_format=task["output_format"],
            raw_response=raw_response,
            system_prompt=system_prompt,
            user_input=user_input,
            raw_responses=raw_responses,
            tool_results=tool_results,
        )

    def _find_executor(self, executor_name: str) -> dict[str, Any]:
        for executor in self.system_spec["executors"]:
            if executor["name"] == executor_name:
                return executor
        raise KeyError(f"Unknown executor name: {executor_name}")


# ---------------------------------------------------------------------------
# Agent / prompt builders
# ---------------------------------------------------------------------------


def build_openai_agent(
    executor: dict[str, Any],
    *,
    tool_executor: ShellToolExecutor,
    use_dspy: bool,
) -> Agent[Any]:
    return Agent(
        name=executor["name"],
        instructions=build_executor_system_prompt(executor, use_dspy=use_dspy),
        model=executor["llm"],
        tools=[build_function_tool(skill, tool_executor=tool_executor) for skill in executor["task"]["skills"]],
        output_type=build_output_type(executor["task"]["output_format"], executor["task"]["output_fields"], executor["name"]),
    )


def build_executor_system_prompt(executor: dict[str, Any], *, use_dspy: bool = False) -> str:
    task = executor["task"]
    rules = "\n".join(_format_rule(rule) for rule in executor["rules"]) or "None"
    skills = "\n".join(_format_skill(skill) for skill in task["skills"]) or "None"
    schema = parse_output_schema(output_format=task["output_format"], output_fields=task["output_fields"])
    schema_instruction = describe_output_schema(schema)
    examples = build_examples_prompt(executor, use_dspy=use_dspy)

    return (
        f"You are {executor['persona']}.\n"
        f"Task: {task['name']}\n"
        f"Behavior: {task['behavior']}\n"
        f"Rules:\n{rules}\n\n"
        f"Available tools:\n{skills}\n\n"
        f"Output contract: {schema_instruction}\n\n"
        "Execution guidance:\n"
        "1. Use the provided tools instead of inventing shell transcripts.\n"
        "2. When state is uncertain, inspect first. Check what exists and what is needed before "
        "running mutating or expensive commands.\n"
        "3. Use tool outputs to decide the next step.\n"
        "4. Only return success after the required artifacts are actually created.\n"
        "5. If work cannot proceed safely, return the structured failure explanation instead of inventing outputs.\n\n"
        f"{examples}"
    )


def build_examples_prompt(executor: dict[str, Any], *, use_dspy: bool) -> str:
    examples = executor["task"]["examples"]
    if not examples:
        return "Examples:\nNone"
    heading = "Examples:\n"
    if use_dspy:
        heading = (
            "Examples:\n"
            "Treat these examples as high-signal task demonstrations. Reuse their patterns when they fit the "
            "observed state, but still adapt to actual tool results.\n"
        )
    return heading + "\n\n".join(_format_example(example) for example in examples)


def build_function_tool(skill: dict[str, Any], *, tool_executor: ShellToolExecutor) -> FunctionTool:
    properties = {
        argument["name"]: {
            "type": "string",
            "description": argument["description"],
        }
        for argument in skill["arguments"]
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }

    async def on_invoke_tool(_tool_context: Any, arguments_json: str) -> dict[str, Any]:
        arguments = json.loads(arguments_json) if arguments_json else {}
        command = render_skill_command(skill["command"], arguments)
        result = tool_executor.execute(command)
        return {
            "command": result.command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    return FunctionTool(
        name=skill["name"],
        description=skill["description"],
        params_json_schema=schema,
        on_invoke_tool=on_invoke_tool,
        strict_json_schema=True,
    )


def render_skill_command(template: str, arguments: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in arguments:
            raise KeyError(f"Missing required skill argument: {name}")
        return shlex.quote(str(arguments[name]))

    return _PARAM_PATTERN.sub(replace, template)


def build_output_type(output_format: str, output_fields: list[dict], executor_name: str) -> type[Any] | None:
    """Build a Pydantic model for json-format outputs; return None for text formats."""
    if output_format != "json":
        return None

    schema = parse_output_schema(output_format=output_format, output_fields=output_fields)
    fields = {}
    for field_spec in schema.fields:
        fields[field_spec.name] = (
            _python_type_for_schema(field_spec.type_name),
            Field(description=field_spec.name.replace("_", " ")),
        )

    model_name = f"{executor_name}Output"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


def load_system_spec(spec_json: str) -> dict[str, Any]:
    return json.loads(spec_json)


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
            file_values.update({key: value for key, value in dotenv_values(path).items() if value is not None})

    api_key = file_values.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = file_values.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_tool_results(items: list[Any]) -> list[ToolResult]:
    results = []
    for item in items:
        if not isinstance(item, ToolCallOutputItem):
            continue
        output = item.output
        if not isinstance(output, dict):
            continue
        required = {"command", "stdout", "stderr", "exit_code"}
        if not required.issubset(output):
            continue
        results.append(
            ToolResult(
                command=str(output["command"]),
                stdout=str(output["stdout"]),
                stderr=str(output["stderr"]),
                exit_code=int(output["exit_code"]),
            )
        )
    return results


def _dump_raw_response(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    try:
        return json.dumps(response, indent=2)
    except TypeError:
        return str(response)


def _stringify_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, indent=2)
    except TypeError:
        return str(output)


def _python_type_for_schema(type_name: str) -> Any:
    normalized = type_name.lower()
    if normalized in {"str", "string"}:
        return str
    if normalized == "int":
        return int
    if normalized == "float":
        return float
    if normalized == "bool":
        return bool
    list_match = re.fullmatch(r"list\[(str|string|int|float|bool)]", normalized)
    if list_match:
        inner = _python_type_for_schema(list_match.group(1))
        return list[inner]
    raise ValueError(f"Unsupported output schema type: {type_name}")


def _format_rule(rule: dict[str, Any]) -> str:
    prefix = "DON'T" if rule["negative"] else "DO"
    return f"- {prefix}: {rule['description']}"


def _format_skill(skill: dict[str, Any]) -> str:
    args = ", ".join(argument["name"] for argument in skill["arguments"]) or "no arguments"
    return f"- {skill['name']}: {skill['description']} (args: {args})"


def _format_output_spec(task: dict[str, Any]) -> str:
    fmt = task.get("output_format", "string")
    fields = task.get("output_fields", [])
    if not fields:
        return fmt
    parts = ", ".join(f"{f['name']}: {f['type']}" for f in fields)
    return f"{fmt} {{ {parts} }}"


def _format_example(example: dict[str, Any]) -> str:
    commands = ", ".join(example["commands"]) or "no commands"
    return f"Input: {example['input']}\n" f"Command trajectory: {commands}\n" f"Final output: {example['output']}"
