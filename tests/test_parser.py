from agent.parser import load_metamodel, parse_model

def test_load_metamodel():
    meta_model = load_metamodel()
    assert meta_model is not None


def test_parse_minimal_model():
    model = parse_model("models/example_minimal.agent")
    assert model is not None
    assert model.system is not None
    assert model.system.planner is not None
    assert len(model.system.executors) >= 1


def test_parse_full_model():
    model = parse_model("models/example_full.agent")
    assert model.system.planner.persona == "senior planning agent"
    assert len(model.system.executors) == 1
    assert model.system.executors[0].task.name == "WebResearch"