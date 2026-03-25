from agent.parser import load_metamodel, parse_model

def test_load_metamodel():
    meta_model = load_metamodel()
    assert meta_model is not None


def test_parse_minimal_model():
    system = parse_model("models/example_minimal.agent")
    assert system is not None
    assert system.planner is not None
    assert len(system.executors) >= 1


def test_parse_full_model():
    system = parse_model("models/example_full.agent")
    assert system.planner.persona == "senior planning agent"
    assert len(system.executors) == 1
    assert system.executors[0].task.name == "WebResearch"