import asyncio
import json
from types import SimpleNamespace

import pytest
from agents import Agent
from agents.items import ToolCallOutputItem

from agent.ir import build_prompt_ir
from agent.parser import parse_model
from agent.runtime import (
    AgentSystemRuntime,
    ShellToolExecutor,
    _HandoffCtx,
    _RoutingHooks,
    build_examples_prompt,
    build_executor_system_prompt,
    build_function_tool,
    build_openai_agent,
    build_output_type,
    build_planner_agent,
    build_planner_prompt,
    load_openai_config,
    render_skill_command,
)
from agent.schema import coerce_structured_output, parse_output_schema


# ---------------------------------------------------------------------------
# Shell / tool primitives
# ---------------------------------------------------------------------------


def test_shell_tool_executor_runs_command():
    result = ShellToolExecutor().execute("printf 'hello'")
    assert result.exit_code == 0
    assert result.stdout == "hello"


def test_render_skill_command_quotes_arguments():
    command = render_skill_command(
        "<python_bin> <script_path> --input <input_path> --output <output_path>",
        {
            "python_bin": "python3",
            "script_path": "demo script.py",
            "input_path": "input file.json",
            "output_path": "out.svg",
        },
    )
    assert command == "python3 'demo script.py' --input 'input file.json' --output out.svg"


def test_build_function_tool_executes_shell_skill():
    skill = {
        "name": "echoTool",
        "description": "Echo a value",
        "command": "printf <value>",
        "arguments": [{"name": "value", "description": "Value to print"}],
    }
    tool = build_function_tool(skill, tool_executor=ShellToolExecutor())
    payload = asyncio.run(tool.on_invoke_tool(None, json.dumps({"value": "hello"})))
    assert payload == {
        "command": "printf hello",
        "stdout": "hello",
        "stderr": "",
        "exit_code": 0,
    }


def test_build_output_type_for_structured_schema():
    fields = [
        {"name": "answer", "type": "str"},
        {"name": "count", "type": "int"},
        {"name": "ok", "type": "bool"},
    ]
    output_type = build_output_type("json", fields, "Summarizer")
    assert output_type is not None
    instance = output_type(answer="done", count=2, ok=True)
    assert instance.model_dump() == {"answer": "done", "count": 2, "ok": True}


def test_build_output_type_returns_none_for_non_json():
    assert build_output_type("string", [], "Summarizer") is None
    assert build_output_type("markdown", [], "Summarizer") is None
    assert build_output_type("toml", [{"name": "x", "type": "str"}], "Summarizer") is None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_build_executor_system_prompt_mentions_inspect_first():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    prompt = build_executor_system_prompt(build_prompt_ir(system)["executors"][0], use_dspy=False)
    assert "Use the provided tools instead of inventing shell transcripts." in prompt
    assert "When state is uncertain, inspect first." in prompt


def test_build_examples_prompt_can_enable_dspy_style_guidance():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    executor = build_prompt_ir(system)["executors"][0]
    normal = build_examples_prompt(executor, use_dspy=False)
    enriched = build_examples_prompt(executor, use_dspy=True)
    assert "Treat these examples as high-signal task demonstrations." not in normal
    assert "Treat these examples as high-signal task demonstrations." in enriched


def test_build_openai_agent_compiles_tools_and_output_type():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    executor = build_prompt_ir(system)["executors"][0]
    agent = build_openai_agent(executor, tool_executor=ShellToolExecutor(), use_dspy=False)
    assert agent.name == "DataVisualizer"
    assert [tool.name for tool in agent.tools] == [
        "preparePythonEnv",
        "runPythonScript",
        "writePreprocessor",
    ]
    assert agent.output_type is not None


# ---------------------------------------------------------------------------
# Handoff-based planner
# ---------------------------------------------------------------------------


def _make_executors_ir():
    """Two-executor IR fixture for planner tests."""
    return [
        {
            "name": "WebResearch",
            "llm": "gpt-5.4-nano",
            "persona": "research agent",
            "rules": [],
            "task": {
                "name": "WebResearch",
                "input_description": "A user question requiring web research",
                "behavior": "Search, extract, and summarize relevant information",
                "output_format": "string",
                "output_fields": [],
                "examples": [],
                "skills": [],
            },
        },
        {
            "name": "TextEditor",
            "llm": "gpt-5.4-nano",
            "persona": "editor agent",
            "rules": [],
            "task": {
                "name": "TextEditor",
                "input_description": "Text to edit with instructions",
                "behavior": "Apply edits based on user instructions",
                "output_format": "string",
                "output_fields": [],
                "examples": [],
                "skills": [],
            },
        },
    ]


def test_planner_prompt_lists_all_executors():
    executors = _make_executors_ir()
    prompt = build_planner_prompt(executors)
    assert "WebResearch" in prompt
    assert "TextEditor" in prompt
    assert "transfer" in prompt.lower()


def test_build_planner_agent_has_handoffs_for_each_executor():
    executors = _make_executors_ir()
    hooks = _RoutingHooks()
    executor_agents = {
        e["name"]: build_openai_agent(e, tool_executor=ShellToolExecutor(), use_dspy=False)
        for e in executors
    }
    planner = build_planner_agent(executors, executor_agents, hooks, planner_llm="gpt-5.4-nano")
    assert planner.name == "Planner"
    handoff_names = {h.tool_name for h in planner.handoffs}
    assert "transfer_to_webresearch" in handoff_names
    assert "transfer_to_texteditor" in handoff_names


def test_routing_hooks_captures_executor_name():
    hooks = _RoutingHooks()
    assert hooks.executor_name is None
    assert "single executor" in hooks.planner_reason.lower()

    # Simulate what the SDK calls on handoff
    import asyncio

    async def _sim():
        from unittest.mock import MagicMock
        from_agent = MagicMock()
        from_agent.name = "Planner"
        to_agent = MagicMock()
        to_agent.name = "WebResearch"
        await hooks.on_handoff(None, from_agent, to_agent)

    asyncio.run(_sim())
    assert hooks.executor_name == "WebResearch"


def test_handoff_ctx_schema():
    ctx = _HandoffCtx(reason="User asked for research.")
    assert ctx.reason == "User asked for research."
    schema = _HandoffCtx.model_json_schema()
    assert "reason" in schema["properties"]


# ---------------------------------------------------------------------------
# AgentSystemRuntime — single executor
# ---------------------------------------------------------------------------


def test_agent_runtime_single_executor_runs_directly(monkeypatch):
    """Single-executor: Runner.run_sync is called with the executor agent (no planner)."""
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    prompt_ir = build_prompt_ir(system)
    captured = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        captured["agent_name"] = agent.name
        captured["agent_model"] = agent.model
        captured["run_config"] = run_config
        return SimpleNamespace(
            final_output={"status": "success", "message": "ok", "artifact_path": "out.svg",
                          "preprocessed_data_path": "", "preprocessor_script_path": ""},
            raw_responses=[],
            new_items=[],
        )

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(prompt_ir)
    result = runtime.run("Visualize aligned_sales.json")

    assert captured["agent_name"] == "DataVisualizer"
    assert captured["agent_model"] == "gpt-5.4-nano"
    assert captured["run_config"].model is None
    assert result.executor_name == "DataVisualizer"
    assert result.planner_reason == "Direct execution (single executor)."


def test_agent_runtime_coerces_structured_output(monkeypatch):
    system_spec = {
        "planner": {"reasoning_strategy": "react", "llm": "gpt-5.4-nano", "persona": "planner", "rules": []},
        "executors": [
            {
                "name": "Summarizer",
                "reasoning_strategy": "react",
                "llm": "gpt-5.4-nano",
                "persona": "summarizer",
                "rules": [],
                "task": {
                    "name": "Summarizer",
                    "input_description": "summarize documents",
                    "behavior": "summarize",
                    "output_format": "json",
                    "output_fields": [
                        {"name": "answer", "type": "str"},
                        {"name": "commands_run", "type": "int"},
                        {"name": "success", "type": "bool"},
                        {"name": "commands", "type": "list[str]"},
                    ],
                    "examples": [],
                    "skills": [],
                },
            }
        ],
        "rules": [],
        "skills": [],
    }

    monkeypatch.setattr(
        "agent.runtime.Runner.run_sync",
        lambda *args, **kwargs: SimpleNamespace(
            final_output={"answer": "done", "commands_run": 1, "success": True, "commands": ["printf 'hello'"]},
            raw_responses=[],
            new_items=[],
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system_spec)
    result = runtime.run("summarize this")
    assert result.output == {
        "answer": "done",
        "commands_run": 1,
        "success": True,
        "commands": ["printf 'hello'"],
    }


# ---------------------------------------------------------------------------
# AgentSystemRuntime — multi executor (handoff routing)
# ---------------------------------------------------------------------------


def test_agent_runtime_multi_executor_starts_with_planner(monkeypatch):
    """Multi-executor: Runner.run_sync is called with the Planner agent."""
    system = parse_model("models/example_full.agent")
    prompt_ir = build_prompt_ir(system)
    captured = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        captured["agent_name"] = agent.name
        captured["agent_model"] = agent.model
        captured["run_config"] = run_config
        # Simulate SDK handoff: set hooks as if WebResearch was chosen
        if hooks is not None:
            hooks.executor_name = "WebResearch"
            hooks.planner_reason = "User is asking for web research."
        return SimpleNamespace(
            final_output="## Summary\n\nsource line",
            raw_responses=[{"id": "resp_1"}],
            new_items=[
                ToolCallOutputItem(
                    agent=Agent(name="fake"),
                    raw_item={"type": "function_call_output"},
                    output={
                        "command": "printf 'source line'",
                        "stdout": "source line",
                        "stderr": "",
                        "exit_code": 0,
                    },
                )
            ],
        )

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(prompt_ir)
    result = runtime.run("Find a source")

    assert captured["agent_name"] == "Planner"
    assert captured["agent_model"] == "gpt-5.4-nano"
    assert captured["run_config"].model is None
    assert result.executor_name == "WebResearch"
    assert result.planner_reason == "User is asking for web research."
    assert result.output == "## Summary\n\nsource line"
    assert result.tool_results[0].stdout == "source line"
    assert result.user_input == "Find a source"
    assert "You are research agent." in result.system_prompt


def test_agent_runtime_multi_executor_falls_back_when_no_handoff(monkeypatch):
    """If SDK never triggers a handoff, runtime defaults to first executor."""
    system = parse_model("models/example_full.agent")
    prompt_ir = build_prompt_ir(system)

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        # Do NOT set hooks.executor_name — simulates planner returning direct text
        return SimpleNamespace(final_output="direct answer", raw_responses=[], new_items=[])

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(prompt_ir)
    result = runtime.run("something")
    assert result.executor_name == prompt_ir["executors"][0]["name"]
    assert "defaulted" in result.planner_reason.lower()


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


def test_parse_output_schema_for_structured_types():
    schema = parse_output_schema("answer: str, score: float, sources: list[str]")
    assert schema.is_structured
    assert [field.name for field in schema.fields] == ["answer", "score", "sources"]


def test_coerce_structured_output_validates_shape():
    schema = parse_output_schema("answer: str, count: int")
    payload = coerce_structured_output({"answer": "ok", "count": "2"}, schema)
    assert payload == {"answer": "ok", "count": 2}


def test_load_openai_config_reads_model_local_dotenv(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://example.test/v1\nOPENAI_API_KEY=secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_openai_config(str(model_path))
    assert config == {
        "OPENAI_BASE_URL": "https://example.test/v1",
        "OPENAI_API_KEY": "secret",
    }


def test_load_openai_config_api_key_only(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    config = load_openai_config(str(model_path))
    assert config == {"OPENAI_API_KEY": "secret"}


def test_load_openai_config_prefers_dotenv_over_environment(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://example.test/v1\nOPENAI_API_KEY=from_dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "from_env")
    monkeypatch.chdir(tmp_path)

    config = load_openai_config(str(model_path))
    assert config == {
        "OPENAI_BASE_URL": "https://example.test/v1",
        "OPENAI_API_KEY": "from_dotenv",
    }


def test_runtime_raises_without_provider(tmp_path, monkeypatch):
    system = parse_model("models/example_full.agent")
    prompt_ir = build_prompt_ir(system)
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    runtime = AgentSystemRuntime(prompt_ir, source_model_path=str(model_path))
    with pytest.raises(RuntimeError, match="No OpenAI provider configuration found"):
        runtime.run("Find a source")
