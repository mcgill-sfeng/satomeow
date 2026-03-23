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

def test_invalid_empty_task_example_commands():
    with pytest.raises(TextXSemanticError):
        parse_model("models/invalid_example_empty_commands.agent")