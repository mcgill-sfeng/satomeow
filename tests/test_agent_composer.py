"""Tests for the AgentComposer multi-executor demo.

Covers parsing, IR structure, tool execution, and planner routing.
Does NOT make real API calls — e2e tests for that are in test_e2e_*.py.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from agent.ir import serialize_system_to_dict
from agent.parser import parse_model
from agent.runtime import (
    ShellToolExecutor,
    _RoutingHooks,
    build_openai_agent,
    build_planner_agent,
    build_planner_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "agent_composer" / "agent_composer.agent"


# ---------------------------------------------------------------------------
# Parsing and IR
# ---------------------------------------------------------------------------


def test_agent_composer_parses_without_error():
    system = parse_model(MODEL_PATH)
    assert len(system.executors) == 2
    names = [e.task.name for e in system.executors]
    assert "Composer" in names
    assert "Validator" in names


def test_agent_composer_ir_has_correct_executors():
    ir = serialize_system_to_dict(parse_model(MODEL_PATH))
    executors = ir["executors"]
    assert len(executors) == 2
    names = {e["name"] for e in executors}
    assert names == {"Composer", "Validator"}


def test_composer_ir_has_string_output():
    ir = serialize_system_to_dict(parse_model(MODEL_PATH))
    composer = next(e for e in ir["executors"] if e["name"] == "Composer")
    assert composer["task"]["output_format"] == "string"
    assert composer["task"]["output_fields"] == []


def test_validator_ir_has_json_output():
    ir = serialize_system_to_dict(parse_model(MODEL_PATH))
    validator = next(e for e in ir["executors"] if e["name"] == "Validator")
    assert validator["task"]["output_format"] == "json"
    field_names = {f["name"] for f in validator["task"]["output_fields"]}
    assert field_names == {"valid", "errors", "suggestions"}


def test_skills_are_defined_globally():
    ir = serialize_system_to_dict(parse_model(MODEL_PATH))
    skill_names = {s["name"] for s in ir["skills"]}
    assert {"write_file", "read_file", "run_inspect"}.issubset(skill_names)


def test_composer_has_required_skills():
    ir = serialize_system_to_dict(parse_model(MODEL_PATH))
    composer = next(e for e in ir["executors"] if e["name"] == "Composer")
    skill_names = {s["name"] for s in composer["task"]["skills"]}
    assert skill_names == {"write_file", "read_file", "run_inspect"}


def test_validator_has_required_skills():
    ir = serialize_system_to_dict(parse_model(MODEL_PATH))
    validator = next(e for e in ir["executors"] if e["name"] == "Validator")
    skill_names = {s["name"] for s in validator["task"]["skills"]}
    assert skill_names == {"read_file", "run_inspect"}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def test_write_file_skill_creates_file(tmp_path):
    system = parse_model(MODEL_PATH)
    composer = next(e for e in system.executors if e.task.name == "Composer")
    agent = build_openai_agent(composer, tool_executor=ShellToolExecutor(), use_dspy=False)
    tools = {t.name: t for t in agent.tools}

    out_path = PROJECT_ROOT / "models" / "agent_composer" / "generated" / "test_output.agent"
    out_path.unlink(missing_ok=True)
    payload = json.dumps({"path": str(out_path).replace("\\", "/"), "content": "hello_world"})
    result = asyncio.run(tools["write_file"].on_invoke_tool(None, payload))

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == "hello_world"
    assert result["exit_code"] == 0


@pytest.mark.skipif(os.name == "nt", reason="AgentComposer read_file skill uses POSIX 'cat'.")
def test_read_file_skill_returns_content(tmp_path):
    system = parse_model(MODEL_PATH)
    composer = next(e for e in system.executors if e.task.name == "Composer")
    agent = build_openai_agent(composer, tool_executor=ShellToolExecutor(), use_dspy=False)
    tools = {t.name: t for t in agent.tools}

    src = PROJECT_ROOT / "models" / "agent_composer" / "generated" / "hello.txt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("hello from read_file", encoding="utf-8")
    result = asyncio.run(
        tools["read_file"].on_invoke_tool(None, json.dumps({"path": str(src).replace("\\", "/")}))
    )

    assert "hello from read_file" in result["stdout"]
    assert result["exit_code"] == 0


def test_run_inspect_skill_validates_real_agent_file():
    system = parse_model(MODEL_PATH)
    validator = next(e for e in system.executors if e.task.name == "Validator")
    agent = build_openai_agent(validator, tool_executor=ShellToolExecutor(), use_dspy=False)
    tools = {t.name: t for t in agent.tools}

    result = asyncio.run(
        tools["run_inspect"].on_invoke_tool(
            None,
            json.dumps({"path": "models/data_visualizer/data_visualizer.agent"}),
        )
    )
    assert result["exit_code"] == 0
    parsed = json.loads(result["stdout"])
    assert "executors" in parsed


def test_run_inspect_skill_reports_invalid_file(tmp_path):
    system = parse_model(MODEL_PATH)
    validator = next(e for e in system.executors if e.task.name == "Validator")
    agent = build_openai_agent(validator, tool_executor=ShellToolExecutor(), use_dspy=False)
    tools = {t.name: t for t in agent.tools}

    bad = tmp_path / "bad.agent"
    bad.write_text("this is not valid DSL", encoding="utf-8")
    result = asyncio.run(tools["run_inspect"].on_invoke_tool(None, json.dumps({"path": str(bad)})))
    assert result["exit_code"] != 0


# ---------------------------------------------------------------------------
# Planner routing
# ---------------------------------------------------------------------------


def test_planner_has_handoffs_for_both_executors():
    system = parse_model(MODEL_PATH)
    executors = system.executors
    hooks = _RoutingHooks()
    executor_agents = {e.task.name: build_openai_agent(e, tool_executor=ShellToolExecutor(), use_dspy=False) for e in executors}
    planner = build_planner_agent(executors, executor_agents, hooks, planner_llm="gpt-5.4-nano")
    handoff_names = {h.tool_name for h in planner.handoffs}
    assert "transfer_to_composer" in handoff_names
    assert "transfer_to_validator" in handoff_names


def test_planner_prompt_mentions_both_executors():
    system = parse_model(MODEL_PATH)
    prompt = build_planner_prompt(system.executors)
    assert "Composer" in prompt
    assert "Validator" in prompt
