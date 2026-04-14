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


def test_example_command_rejects_unknown_tool(tmp_path):
    model_path = tmp_path / "bad_example_tool.agent"
    model_path.write_text(
        """llm: "gpt-5.4-nano"

Summarize : "text summarizer" {
    input: "A long document"
    behavior: "Summarize the document"
    skills: [read_file]

    example {
        input: "doc text"
        commands: [missing_tool(path: "doc.txt")]
        output: "summary"
    }
}

read_file {
    command: "cat <path>"
    description: "Read a file"
    param path: "File path"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(TextXSemanticError, match="Unknown example command tool"):
        parse_model(model_path)


def test_example_command_rejects_missing_argument(tmp_path):
    model_path = tmp_path / "bad_example_args.agent"
    model_path.write_text(
        """llm: "gpt-5.4-nano"

Summarize : "text summarizer" {
    input: "A long document"
    behavior: "Summarize the document"
    skills: [read_file]

    example {
        input: "doc text"
        commands: [read_file()]
        output: "summary"
    }
}

read_file {
    command: "cat <path>"
    description: "Read a file"
    param path: "File path"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(TextXSemanticError, match="Missing example command arguments"):
        parse_model(model_path)
