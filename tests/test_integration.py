from agent.ir import serialize_system_to_dict
from agent.parser import parse_model


def test_cross_references_are_resolved():
    system = parse_model("models/example_full.agent")
    rule_names = {r.name for r in system.rules}
    assert "verify_sources" in rule_names
    assert "no_hallucination" in rule_names
    skill_commands = {s.command for s in system.skills}
    assert "ddgr <query>" in skill_commands
    assert "curl -s <url>" in skill_commands


def test_task_skills_are_resolved():
    system = parse_model("models/example_full.agent")
    task = system.executors[0].task
    assert len(task.skills) == 2
    skill_commands = {s.command for s in task.skills}
    assert "ddgr <query>" in skill_commands
    assert "curl -s <url>" in skill_commands


def test_task_rules_are_resolved():
    system = parse_model("models/example_full.agent")
    executor = system.executors[0]
    rule_names = {r.name for r in executor.rules}
    assert "verify_sources" in rule_names
    assert "no_hallucination" in rule_names


def test_rule_negative_do():
    system = parse_model("models/example_full.agent")
    verify = next(r for r in system.rules if r.name == "verify_sources")
    assert verify.negative is False


def test_rule_negative_dont():
    system = parse_model("models/example_full.agent")
    no_hall = next(r for r in system.rules if r.name == "no_hallucination")
    assert no_hall.negative is True


def test_skill_arguments_converted():
    system = parse_model("models/example_full.agent")
    web_search = next(s for s in system.skills if s.name == "webSearch")
    assert len(web_search.skillArguments) == 1
    assert web_search.skillArguments[0].name == "query"


def test_multiple_examples_parsed():
    system = parse_model("models/example_full.agent")
    examples = system.executors[0].task.examples
    assert len(examples) == 2


def test_prompt_ir_structure():
    system = parse_model("models/example_full.agent")
    ir = serialize_system_to_dict(system)

    assert "planner" in ir
    assert "executors" in ir
    assert isinstance(ir["executors"], list)

    assert ir["planner"]["llm"] == "gpt-5.4-nano"
    assert ir["planner"]["persona"] == "planner"
    assert ir["executors"][0]["task"]["name"] == "WebResearch"
    assert ir["executors"][0]["name"] == "WebResearch"


def test_prompt_ir_output_schema():
    system = parse_model("models/example_full.agent")
    ir = serialize_system_to_dict(system)
    assert ir["executors"][0]["task"]["output_format"] == "markdown"
    assert ir["executors"][0]["task"]["output_fields"] == []


def test_prompt_ir_skills():
    system = parse_model("models/example_full.agent")
    ir = serialize_system_to_dict(system)
    task_skills = ir["executors"][0]["task"]["skills"]
    assert len(task_skills) == 2
    skill_names = {s["name"] for s in task_skills}
    assert "webSearch" in skill_names
    assert "docReader" in skill_names


def test_prompt_ir_rules_include_metadata():
    system = parse_model("models/example_full.agent")
    ir = serialize_system_to_dict(system)
    rule = next(rule for rule in ir["rules"] if rule["name"] == "no_hallucination")
    assert rule["negative"] is True
    assert rule["description"] == "do not make up facts"
