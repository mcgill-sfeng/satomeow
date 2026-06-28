import asyncio
import json
from types import SimpleNamespace

import pytest
from agents import Agent
from agents.items import HandoffOutputItem, ToolCallItem, ToolCallOutputItem

from agent.metamodel import Executor, OutputField, OutputSpec, Planner, System, Task
from agent.parser import parse_model
from agent.runtime import (
    AgentSystemRuntime,
    CallEdge,
    CallGraph,
    ShellToolExecutor,
    _HandoffCtx,
    _RoutingHooks,
    build_call_graph,
    build_examples_prompt,
    build_executor_system_prompt,
    build_function_tool,
    build_model_settings,
    build_openai_agent,
    build_output_type,
    build_planner_agent,
    build_planner_prompt,
    load_openai_config,
    render_call_graph_dot,
    render_call_graph_text,
    render_skill_command,
)
from agent.schema import coerce_structured_output, parse_output_schema

# ---------------------------------------------------------------------------
# Shell / tool primitives
# ---------------------------------------------------------------------------


def test_shell_tool_executor_runs_command():
    result = ShellToolExecutor().execute("python -c \"import sys; sys.stdout.write('hello')\"")
    assert result.exit_code == 0
    assert result.stdout == "hello"


def test_render_skill_command_quotes_arguments():
    command = render_skill_command(
        "<python_bin> <script_path> --input <input_path> --output <output_path>",
        {
            "python_bin": "python3",
            "script_path": "demo script.py",
            "input_path": "input file.json",
            "output_path": "out.svg",
        },
    )
    assert command == "python3 'demo script.py' --input 'input file.json' --output out.svg"


def test_build_function_tool_executes_shell_skill():
    skill = SimpleNamespace(
        name="echoTool",
        description="Echo a value",
        command="echo <value>",
        skillArguments=[SimpleNamespace(name="value", description="Value to print")],
    )
    tool = build_function_tool(skill, tool_executor=ShellToolExecutor())
    payload = asyncio.run(tool.on_invoke_tool(None, json.dumps({"value": "hello"})))
    assert payload["command"] == "echo hello"
    assert payload["exit_code"] == 0
    assert "hello" in payload["stdout"].lower()


def test_build_output_type_for_structured_schema():
    output_spec = OutputSpec(
        format="json",
        fields=[
            OutputField(name="answer", type="str"),
            OutputField(name="count", type="int"),
            OutputField(name="ok", type="bool"),
        ],
    )
    output_type = build_output_type(output_spec, "Summarizer")
    assert output_type is not None
    instance = output_type(answer="done", count=2, ok=True)
    assert instance.model_dump() == {"answer": "done", "count": 2, "ok": True}


def test_build_output_type_returns_none_for_non_json():
    assert build_output_type(OutputSpec(format="string", fields=[]), "Summarizer") is None
    assert build_output_type(OutputSpec(format="markdown", fields=[]), "Summarizer") is None
    assert build_output_type(OutputSpec(format="toml", fields=[OutputField(name="x", type="str")]), "Summarizer") is None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_build_executor_system_prompt_mentions_inspect_first():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    prompt = build_executor_system_prompt(system.executors[0], use_dspy=False)
    assert "Use the provided tools instead of inventing shell transcripts." in prompt
    assert "When state is uncertain, inspect first." in prompt


def test_build_examples_prompt_can_enable_dspy_style_guidance():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    executor = system.executors[0]
    normal = build_examples_prompt(executor, use_dspy=False)
    enriched = build_examples_prompt(executor, use_dspy=True)
    assert "Treat these examples as high-signal task demonstrations." not in normal
    assert "Treat these examples as high-signal task demonstrations." in enriched


def test_build_model_settings_maps_reasoning_effort():
    settings = build_model_settings("medium")
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "medium"


def test_build_model_settings_leaves_reasoning_unset_by_default():
    settings = build_model_settings(None)
    assert settings.reasoning is None


def test_build_openai_agent_compiles_tools_and_output_type():
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    executor = system.executors[0]
    agent = build_openai_agent(executor, tool_executor=ShellToolExecutor(), use_dspy=False)
    assert agent.name == "DataVisualizer"
    assert agent.model_settings.reasoning is None
    assert [tool.name for tool in agent.tools] == [
        "preparePythonEnv",
        "runPythonScript",
        "writePreprocessor",
    ]
    assert agent.output_type is not None


def test_build_openai_agent_applies_explicit_reasoning_effort():
    system = parse_model("models/example_full.agent")
    executor = system.executors[1]
    agent = build_openai_agent(executor, tool_executor=ShellToolExecutor(), use_dspy=False)
    assert agent.model_settings.reasoning is not None
    assert agent.model_settings.reasoning.effort == "medium"


# ---------------------------------------------------------------------------
# Handoff-based planner
# ---------------------------------------------------------------------------


def _make_executor(name: str, persona: str, input_description: str, behavior: str) -> Executor:
    executor = Executor()
    executor.llm = "gpt-5.4-nano"
    executor.reasoningStrategy = "medium"
    executor.persona = persona
    executor.rules = []

    task = Task()
    task.name = name
    task.inputDescription = input_description
    task.behavior = behavior
    task.outputSpec = OutputSpec(format="string", fields=[])
    task.examples = []
    task.skills = []

    executor.task = task
    return executor


def _make_executors():
    """Two-executor metamodel fixture for planner tests."""
    return [
        _make_executor(
            "WebResearch",
            "research agent",
            "A user question requiring web research",
            "Search, extract, and summarize relevant information",
        ),
        _make_executor(
            "TextEditor",
            "editor agent",
            "Text to edit with instructions",
            "Apply edits based on user instructions",
        ),
    ]


def test_planner_prompt_lists_all_executors():
    executors = _make_executors()
    prompt = build_planner_prompt(executors)
    assert "WebResearch" in prompt
    assert "TextEditor" in prompt
    assert "transfer" in prompt.lower()


def test_build_planner_agent_has_handoffs_for_each_executor():
    executors = _make_executors()
    hooks = _RoutingHooks()
    executor_agents = {e.task.name: build_openai_agent(e, tool_executor=ShellToolExecutor(), use_dspy=False) for e in executors}
    planner = build_planner_agent(
        executors,
        executor_agents,
        hooks,
        planner_llm="gpt-5.4-nano",
        planner_reasoning_effort="medium",
    )
    assert planner.name == "Planner"
    assert planner.model_settings.reasoning is not None
    assert planner.model_settings.reasoning.effort == "medium"
    handoff_names = {h.tool_name for h in planner.handoffs}
    assert "transfer_to_webresearch" in handoff_names
    assert "transfer_to_texteditor" in handoff_names


def test_routing_hooks_captures_executor_name():
    hooks = _RoutingHooks()
    assert hooks.executor_name is None
    assert "single executor" in hooks.planner_reason.lower()

    # Simulate what the SDK calls on handoff
    import asyncio

    async def _sim():
        from unittest.mock import MagicMock

        from_agent = MagicMock()
        from_agent.name = "Planner"
        to_agent = MagicMock()
        to_agent.name = "WebResearch"
        await hooks.on_handoff(None, from_agent, to_agent)

    asyncio.run(_sim())
    assert hooks.executor_name == "WebResearch"


def test_handoff_ctx_schema():
    ctx = _HandoffCtx(reason="User asked for research.")
    assert ctx.reason == "User asked for research."
    schema = _HandoffCtx.model_json_schema()
    assert "reason" in schema["properties"]


# ---------------------------------------------------------------------------
# AgentSystemRuntime — single executor
# ---------------------------------------------------------------------------


def test_agent_runtime_single_executor_runs_directly(monkeypatch):
    """Single-executor: Runner.run_sync is called with the executor agent (no planner)."""
    system = parse_model("models/data_visualizer/data_visualizer.agent")
    captured = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        captured["agent_name"] = agent.name
        captured["agent_model"] = agent.model
        captured["run_config"] = run_config
        return SimpleNamespace(
            final_output={
                "status": "success",
                "message": "ok",
                "artifact_path": "out.svg",
                "preprocessed_data_path": "",
                "preprocessor_script_path": "",
            },
            raw_responses=[],
            new_items=[],
        )

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system)
    result = runtime.run("Visualize aligned_sales.json")

    assert captured["agent_name"] == "DataVisualizer"
    assert captured["agent_model"] == "gpt-5.4-nano"
    assert captured["run_config"].model is None
    assert result.executor_name == "DataVisualizer"
    assert result.planner_reason == "Direct execution (single executor)."


def test_agent_runtime_coerces_structured_output(monkeypatch):
    system_spec = System()
    planner = Planner()
    planner.llm = "gpt-5.4-nano"
    planner.reasoningStrategy = "medium"
    planner.persona = "planner"
    planner.rules = []
    system_spec.planner = planner

    executor = Executor()
    executor.llm = "gpt-5.4-nano"
    executor.reasoningStrategy = "medium"
    executor.persona = "summarizer"
    executor.rules = []

    task = Task()
    task.name = "Summarizer"
    task.inputDescription = "summarize documents"
    task.behavior = "summarize"
    task.outputSpec = OutputSpec(
        format="json",
        fields=[
            OutputField(name="answer", type="str"),
            OutputField(name="commands_run", type="int"),
            OutputField(name="success", type="bool"),
            OutputField(name="commands", type="list[str]"),
        ],
    )
    task.examples = []
    task.skills = []
    executor.task = task

    system_spec.executors = [executor]
    system_spec.rules = []
    system_spec.skills = []

    monkeypatch.setattr(
        "agent.runtime.Runner.run_sync",
        lambda *args, **kwargs: SimpleNamespace(
            final_output={"answer": "done", "commands_run": 1, "success": True, "commands": ["printf 'hello'"]},
            raw_responses=[],
            new_items=[],
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system_spec)
    result = runtime.run("summarize this")
    assert result.output == {
        "answer": "done",
        "commands_run": 1,
        "success": True,
        "commands": ["printf 'hello'"],
    }


# ---------------------------------------------------------------------------
# AgentSystemRuntime — multi executor (handoff routing)
# ---------------------------------------------------------------------------


def test_agent_runtime_multi_executor_starts_with_planner(monkeypatch):
    """Multi-executor: Runner.run_sync is called with the Planner agent."""
    system = parse_model("models/example_full.agent")
    captured = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        captured["agent_name"] = agent.name
        captured["agent_model"] = agent.model
        captured["run_config"] = run_config
        # Simulate SDK handoff: set hooks as if WebResearch was chosen
        if hooks is not None:
            hooks.executor_name = "WebResearch"
            hooks.planner_reason = "User is asking for web research."
        return SimpleNamespace(
            final_output="## Summary\n\nsource line",
            raw_responses=[{"id": "resp_1"}],
            new_items=[
                ToolCallOutputItem(
                    agent=Agent(name="fake"),
                    raw_item={"type": "function_call_output"},
                    output={
                        "command": "printf 'source line'",
                        "stdout": "source line",
                        "stderr": "",
                        "exit_code": 0,
                    },
                )
            ],
        )

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system)
    result = runtime.run("Find a source")

    assert captured["agent_name"] == "Planner"
    assert captured["agent_model"] == "gpt-5.4-nano"
    assert captured["run_config"].model is None
    assert result.executor_name == "WebResearch"
    assert result.planner_reason == "User is asking for web research."
    assert result.output == "## Summary\n\nsource line"
    assert result.tool_results[0].stdout == "source line"
    assert result.user_input == "Find a source"
    assert "You are research agent." in result.system_prompt


def test_agent_runtime_multi_executor_falls_back_when_no_handoff(monkeypatch):
    """If SDK never triggers a handoff, runtime defaults to first executor."""
    system = parse_model("models/example_full.agent")

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        # Do NOT set hooks.executor_name — simulates planner returning direct text
        return SimpleNamespace(final_output="direct answer", raw_responses=[], new_items=[])

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system)
    result = runtime.run("something")
    assert result.executor_name == system.executors[0].task.name
    assert "defaulted" in result.planner_reason.lower()


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


def test_parse_output_schema_for_structured_types():
    schema = parse_output_schema(
        OutputSpec(
            format="json",
            fields=[
                OutputField(name="answer", type="str"),
                OutputField(name="score", type="float"),
                OutputField(name="sources", type="list[str]"),
            ],
        )
    )
    assert schema.is_structured
    assert [field.name for field in schema.fields] == ["answer", "score", "sources"]


def test_coerce_structured_output_validates_shape():
    schema = parse_output_schema(
        OutputSpec(
            format="json",
            fields=[OutputField(name="answer", type="str"), OutputField(name="count", type="int")],
        )
    )
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
    monkeypatch.chdir(tmp_path)

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
    monkeypatch.chdir(tmp_path)

    config = load_openai_config(str(model_path))
    assert config == {"OPENAI_API_KEY": "secret"}


def test_load_openai_config_prefers_dotenv_over_environment(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://example.test/v1\nOPENAI_API_KEY=from_dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "from_env")
    monkeypatch.chdir(tmp_path)

    config = load_openai_config(str(model_path))
    assert config == {
        "OPENAI_BASE_URL": "https://example.test/v1",
        "OPENAI_API_KEY": "from_dotenv",
    }


def test_runtime_raises_without_provider(tmp_path, monkeypatch):
    system = parse_model("models/example_full.agent")
    model_path = tmp_path / "demo.agent"
    model_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    runtime = AgentSystemRuntime(system, source_model_path=str(model_path))
    with pytest.raises(RuntimeError, match="No OpenAI provider configuration found"):
        runtime.run("Find a source")


# ---------------------------------------------------------------------------
# Call graph
# ---------------------------------------------------------------------------


def _make_tool_call_item(agent, call_id, tool_name, arguments_json):
    """Build a ToolCallItem with a SimpleNamespace raw_item."""
    raw = SimpleNamespace(name=tool_name, call_id=call_id, arguments=arguments_json)
    return ToolCallItem(agent=agent, raw_item=raw)


def _make_tool_output_item(agent, call_id, exit_code):
    """Build a ToolCallOutputItem that looks like a shell tool result."""
    return ToolCallOutputItem(
        agent=agent,
        raw_item={"type": "function_call_output", "call_id": call_id},
        output={"command": "echo hi", "stdout": "hi", "stderr": "", "exit_code": exit_code},
    )


def _make_handoff_item(planner_agent, executor_agent):
    """Build a HandoffOutputItem from planner to executor."""
    return HandoffOutputItem(
        agent=planner_agent,
        raw_item={"type": "function_call_output"},
        source_agent=planner_agent,
        target_agent=executor_agent,
    )


def test_build_call_graph_empty_items():
    graph = build_call_graph([], executor_name="Executor")
    assert graph.edges == (CallEdge(from_node="Executor", to_node="[output]"),)


def test_build_call_graph_single_tool_call():
    agent = Agent(name="Executor")
    items = [
        _make_tool_call_item(agent, "cid_1", "run_shell", '{"cmd": "ls"}'),
        _make_tool_output_item(agent, "cid_1", 0),
    ]
    graph = build_call_graph(items, executor_name="Executor")

    edge_pairs = [(e.from_node, e.to_node) for e in graph.edges]
    assert ("User", "Executor") in edge_pairs
    assert ("Executor", "run_shell") in edge_pairs
    assert ("run_shell", "Executor") in edge_pairs
    assert ("Executor", "[output]") in edge_pairs


def test_build_call_graph_tool_call_exit_code_label():
    agent = Agent(name="Executor")
    items = [
        _make_tool_call_item(agent, "cid_1", "run_shell", '{"cmd": "ls"}'),
        _make_tool_output_item(agent, "cid_1", 1),
    ]
    graph = build_call_graph(items, executor_name="Executor")
    return_edge = next(e for e in graph.edges if e.from_node == "run_shell")
    assert return_edge.label == "exit 1"


def test_build_call_graph_repeated_tool_call_numbered():
    agent = Agent(name="Executor")
    items = [
        _make_tool_call_item(agent, "cid_1", "run_shell", '{"cmd": "ls"}'),
        _make_tool_output_item(agent, "cid_1", 0),
        _make_tool_call_item(agent, "cid_2", "run_shell", '{"cmd": "pwd"}'),
        _make_tool_output_item(agent, "cid_2", 0),
    ]
    graph = build_call_graph(items, executor_name="Executor")
    to_nodes = [e.to_node for e in graph.edges]
    assert "run_shell" in to_nodes
    assert "run_shell#2" in to_nodes


def test_build_call_graph_handoff():
    planner = Agent(name="Planner")
    executor = Agent(name="Summarizer")
    items = [_make_handoff_item(planner, executor)]
    graph = build_call_graph(items, executor_name="Summarizer")

    edge_pairs = [(e.from_node, e.to_node) for e in graph.edges]
    assert ("User", "Planner") in edge_pairs
    assert ("Planner", "Summarizer") in edge_pairs
    assert ("Summarizer", "[output]") in edge_pairs


def test_render_call_graph_text_format():
    graph = CallGraph(
        edges=(
            CallEdge("User", "Executor"),
            CallEdge("Executor", "run_shell", label='cmd="ls"'),
            CallEdge("run_shell", "Executor", label="exit 0"),
            CallEdge("Executor", "[output]"),
        )
    )
    text = render_call_graph_text(graph)
    assert "[call graph]" in text
    assert "User ──► Executor" in text
    assert 'cmd="ls"' in text
    assert "exit 0" in text
    assert "Executor ──► [output]" in text


def test_render_call_graph_dot_format():
    graph = CallGraph(
        edges=(
            CallEdge("User", "Executor"),
            CallEdge("Executor", "run_shell", label='cmd="ls"'),
            CallEdge("Executor", "[output]"),
        )
    )
    dot = render_call_graph_dot(graph)
    assert "digraph agent_run" in dot
    assert '"User" -> "Executor"' in dot
    assert '"Executor" -> "[output]"' in dot
    assert "label=" in dot


def test_run_result_has_call_graph(monkeypatch):
    """RunResult.call_graph is populated after a successful run."""
    system = parse_model("models/example_minimal.agent")
    executor_agent = Agent(name=system.executors[0].task.name)

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        if hooks is not None:
            hooks.executor_name = executor_agent.name
        return SimpleNamespace(
            final_output="ok",
            raw_responses=[],
            new_items=[
                _make_tool_call_item(executor_agent, "cid_1", "echo_tool", '{"value": "hi"}'),
                _make_tool_output_item(executor_agent, "cid_1", 0),
            ],
        )

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(system)
    result = runtime.run("hello")

    assert isinstance(result.call_graph, CallGraph)
    to_nodes = {e.to_node for e in result.call_graph.edges}
    assert "[output]" in to_nodes
    assert "echo_tool" in to_nodes
