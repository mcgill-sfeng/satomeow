import pytest

from textx import TextXSemanticError
from agent.parser import parse_model


def test_invalid_duplicate_rule():
    with pytest.raises(TextXSemanticError):
        parse_model("models/invalid_example_duplicate_rule.agent")


def test_invalid_duplicate_skill():
    with pytest.raises(TextXSemanticError):
        parse_model("models/invalid_example_duplicate_skill.agent")


def test_invalid_duplicate_skill_argument():
    with pytest.raises(TextXSemanticError):
        parse_model("models/invalid_example_duplicate_skill_argument.agent")



def test_invalid_duplicate_executor_names():
    with pytest.raises(TextXSemanticError, match="Duplicate"):
        parse_model("models/invalid_example_duplicate_executors.agent")


def test_valid_minimal_does_not_raise():
    system = parse_model("models/example_minimal.agent")
    assert system is not None


def test_valid_full_does_not_raise():
    system = parse_model("models/example_full.agent")
    assert system is not None
