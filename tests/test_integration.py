from agent.parser import parse_model
from agent.ir import build_prompt_ir

def test_cross_references_are_resolved():
    system = parse_model("models/example_full.agent")

    assert system.rules[0].description == "Do not hallucinate facts"
    assert system.skills[0].command == "search_web"

def test_task_skills_are_resolved():
    system = parse_model("models/example_full.agent")

    task = system.executors[0].task
    assert len(task.skills) == 2
    assert task.skills[0].command == "search_web"
    assert task.skills[1].command == "read_docs"

def test_prompt_ir_structure():
    system = parse_model("models/example_full.agent")
    ir = build_prompt_ir(system)

    assert "planner" in ir
    assert "executors" in ir
    assert isinstance(ir["executors"], list)

    assert ir["planner"]["persona"] == "senior planning agent"
    assert ir["executors"][0]["task"]["name"] == "WebResearch"