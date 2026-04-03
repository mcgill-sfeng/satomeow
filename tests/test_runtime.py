import json

from agent.ir import build_prompt_ir
from agent.parser import parse_model
from agent.runtime import (
    AgentSystemRuntime,
    DSPyExampleModel,
    ExampleDrivenModel,
    OpenAIChatModel,
    ShellToolExecutor,
    create_model_client,
    load_openai_config,
    extract_final_output,
    extract_tool_call,
)
from agent.schema import coerce_structured_output, parse_output_schema


class ScriptedModelClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, *, system_prompt, user_prompt, context):
        del system_prompt, user_prompt, context
        return self._responses.pop(0)


def test_extract_tool_call_from_xml():
    tool_call = extract_tool_call("<tool_call>printf 'hello'</tool_call>")
    assert tool_call is not None
    assert tool_call.command == "printf 'hello'"


def test_extract_tool_call_from_line():
    tool_call = extract_tool_call("TOOL_CALL: printf 'hello'")
    assert tool_call is not None
    assert tool_call.command == "printf 'hello'"


def test_extract_final_output_from_wrapper():
    assert extract_final_output("<final>done</final>") == "done"


def test_shell_tool_executor_runs_command():
    result = ShellToolExecutor().execute("printf 'hello'")
    assert result.exit_code == 0
    assert result.stdout == "hello"


def test_agent_runtime_executes_tool_calls_and_returns_markdown():
    system = parse_model("models/example_full.agent")
    runtime = AgentSystemRuntime(
        build_prompt_ir(system),
        model_client=ScriptedModelClient(
            [
                "<tool_call>printf 'source line'</tool_call>",
                "<final>## Summary\n\nsource line</final>",
            ]
        ),
    )

    result = runtime.run("Find a source")

    assert result.executor_name == "WebResearch"
    assert result.output == "## Summary\n\nsource line"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].stdout == "source line"


def test_agent_runtime_coerces_structured_output():
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

    runtime = AgentSystemRuntime(
        system_spec,
        model_client=ScriptedModelClient(
            [
                json.dumps(
                    {
                        "answer": "done",
                        "commands_run": 1,
                        "success": True,
                        "commands": ["printf 'hello'"],
                    }
                ),
            ]
        ),
    )

    result = runtime.run("summarize this")
    assert result.output["answer"] == "done"
    assert result.output["commands_run"] == 1
    assert result.output["success"] is True


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

    config = load_openai_config(str(model_path))
    assert config == {"OPENAI_API_KEY": "secret"}


def test_openai_chat_model_reads_chat_completion(monkeypatch):
    class _FakeMessage:
        content = "model output"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeChat:
        class completions:
            @staticmethod
            def create(*, model, messages, **kwargs):
                assert model == "gpt-4.1-mini"
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                return _FakeCompletion()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr("agent.runtime.openai.OpenAI", lambda **kwargs: _FakeClient())
    model = OpenAIChatModel(
        base_url="https://example.test/v1",
        api_key="secret",
    )
    output = model.generate(
        system_prompt="system",
        user_prompt="user",
        context={"executor": {"llm": "gpt-4.1-mini"}},
    )
    assert output == "model output"


def test_example_driven_model_executes_commands_before_final():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    runtime = AgentSystemRuntime(
        build_prompt_ir(system),
        model_client=ExampleDrivenModel(),
    )
    result = runtime.run("Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json")
    assert len(result.tool_results) == 2
    assert result.output["status"] == "success"
    assert result.output["artifact_path"].endswith("aligned_chart.svg")


def test_create_model_client_returns_dspy_variant(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://example.test/v1\nOPENAI_API_KEY=secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_model_client(source_model_path=str(model_path), use_dspy=True)
    assert isinstance(client, DSPyExampleModel)


def test_dspy_example_model_uses_examples(monkeypatch):
    captured = {}

    class FakeLM:
        def __init__(self, **kwargs):
            captured["lm_kwargs"] = kwargs

    class FakePredict:
        def __init__(self, signature):
            captured["signature"] = signature
            self.demos = []

        def __call__(self, *, system_prompt, user_prompt, lm):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["lm"] = lm

            class Result:
                response = "<tool_call>printf 'from dspy'</tool_call>"

            return Result()

    class FakeDSPy:
        LM = FakeLM
        Predict = FakePredict

        class Example:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def with_inputs(self, *fields):
                self.input_fields = fields
                return self

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dspy":
            return FakeDSPy
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = DSPyExampleModel(api_key="secret", base_url="https://example.test/v1")
    executor = {
        "name": "Demo",
        "llm": "gpt-4.1-mini",
        "task": {
            "name": "Demo",
            "behavior": "demo",
            "input_description": "demo input",
            "output_schema": "string",
            "examples": [
                {
                    "input": "Say hello",
                    "commands": ["printf 'hello'"],
                    "output": "hello",
                }
            ],
            "skills": [],
        },
        "persona": "demo agent",
        "rules": [],
    }
    response = model.generate(
        system_prompt="sys",
        user_prompt="user",
        context={"executor": executor},
    )

    assert response == "<tool_call>printf 'from dspy'</tool_call>"
    assert len(model._predictors["Demo"].demos) == 1
    assert captured["lm_kwargs"]["model"] == "openai/gpt-4.1-mini"
