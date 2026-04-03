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
    build_examples_prompt,
    build_executor_system_prompt,
    build_function_tool,
    build_openai_agent,
    build_output_type,
    load_openai_config,
    render_skill_command,
)
from agent.schema import coerce_structured_output, parse_output_schema


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
    output_type = build_output_type("answer: str, count: int, ok: bool", "Summarizer")
    assert output_type is not None
    instance = output_type(answer="done", count=2, ok=True)
    assert instance.model_dump() == {"answer": "done", "count": 2, "ok": True}


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


def test_agent_runtime_uses_sdk_runner(monkeypatch):
    system = parse_model("models/example_full.agent")
    prompt_ir = build_prompt_ir(system)
    captured = {}

    def fake_run_sync(agent, user_input, run_config):
        captured["agent"] = agent
        captured["user_input"] = user_input
        captured["run_config"] = run_config
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

    runtime = AgentSystemRuntime(prompt_ir, require_provider=True)
    result = runtime.run("Find a source")

    assert result.executor_name == "WebResearch"
    assert result.output == "## Summary\n\nsource line"
    assert result.tool_results[0].stdout == "source line"
    assert result.user_input == "Find a source"
    assert "You are research agent." in result.system_prompt
    assert captured["user_input"] == "Find a source"
    assert captured["agent"].name == "WebResearch"


def test_agent_runtime_coerces_structured_output(monkeypatch):
    system_spec = {
        "planner": {
            "reasoning_strategy": "react",
            "llm": "test",
            "persona": "planner",
            "rules": [],
        },
        "executors": [
            {
                "name": "Summarizer",
                "reasoning_strategy": "react",
                "llm": "test",
                "persona": "summarizer",
                "rules": [],
                "task": {
                    "name": "Summarizer",
                    "input_description": "summarize documents",
                    "behavior": "summarize",
                    "output_schema": "answer: str, commands_run: int, success: bool, commands: list[str]",
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
            final_output={
                "answer": "done",
                "commands_run": 1,
                "success": True,
                "commands": ["printf 'hello'"],
            },
            raw_responses=[],
            new_items=[],
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system_spec, require_provider=True)
    result = runtime.run("summarize this")
    assert result.output == {
        "answer": "done",
        "commands_run": 1,
        "success": True,
        "commands": ["printf 'hello'"],
    }


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


def test_runtime_requires_provider(tmp_path, monkeypatch):
    system = parse_model("models/example_full.agent")
    prompt_ir = build_prompt_ir(system)
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    runtime = AgentSystemRuntime(prompt_ir, source_model_path=str(model_path), require_provider=True)
    with pytest.raises(RuntimeError, match="No OpenAI provider configuration found"):
        runtime.run("Find a source")
