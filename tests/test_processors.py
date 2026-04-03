"""Unit tests for agent/processors.py."""

from agent.metamodel import SkillArgument
from agent.processors import process_rule, process_skill

# ---------------------------------------------------------------------------
# process_rule
# ---------------------------------------------------------------------------


class _MockRule:
    def __init__(self, rule_type):
        self.ruleType = rule_type
        self.name = "test_rule"
        self.description = "a test rule"


def test_process_rule_do_sets_negative_false():
    rule = process_rule(_MockRule("do"))
    assert rule.negative is False


def test_process_rule_dont_sets_negative_true():
    rule = process_rule(_MockRule("dont"))
    assert rule.negative is True


def test_process_rule_preserves_name_and_description():
    rule = process_rule(_MockRule("do"))
    assert rule.name == "test_rule"
    assert rule.description == "a test rule"


def test_process_rule_returns_same_object():
    mock = _MockRule("dont")
    result = process_rule(mock)
    assert result is mock


# ---------------------------------------------------------------------------
# process_skill
# ---------------------------------------------------------------------------


class _MockParam:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class _MockSkill:
    def __init__(self, params):
        self.name = "my_skill"
        self.command = "run <x>"
        self.description = "does things"
        self.params = params


def test_process_skill_creates_skill_arguments():
    skill = _MockSkill([_MockParam("x", "first param"), _MockParam("y", "second param")])
    process_skill(skill)
    assert hasattr(skill, "skillArguments")
    assert len(skill.skillArguments) == 2


def test_process_skill_arguments_are_skill_argument_instances():
    skill = _MockSkill([_MockParam("x", "desc")])
    process_skill(skill)
    assert isinstance(skill.skillArguments[0], SkillArgument)


def test_process_skill_copies_name_and_description():
    skill = _MockSkill([_MockParam("query", "the query")])
    process_skill(skill)
    arg = skill.skillArguments[0]
    assert arg.name == "query"
    assert arg.description == "the query"


def test_process_skill_empty_params_produces_empty_list():
    skill = _MockSkill([])
    process_skill(skill)
    assert skill.skillArguments == []


def test_process_skill_returns_same_object():
    skill = _MockSkill([])
    result = process_skill(skill)
    assert result is skill
