# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_runtime.py

# Run a single test by name
pytest tests/test_integration.py::test_cross_references

# Parse and inspect a .agent file
python -m agent.cli inspect models/example_full.agent --print-ir

# Generate a runnable Python module
python -m agent.cli generate models/example_full.agent --output generated_agent.py

# Run an agent directly (requires OPENAI_API_KEY in env or .env)
python -m agent.cli run models/example_full.agent "Compare the REST and GraphQL APIs of GitHub"

# Print a call graph of the run (text or dot/Graphviz)
python -m agent.cli run models/example_full.agent --call-graph text "Compare the REST and GraphQL APIs of GitHub"
python -m agent.cli run models/example_full.agent --call-graph dot "Compare the REST and GraphQL APIs of GitHub" | dot -Tpng -o graph.png

# Print the prompt payload that would be sent for a given input
python -m agent.cli prompt models/example_full.agent "Compare REST and GraphQL"

# Run interactive chat intake mode (requires a 'chat' block in the model)
python -m agent.cli chat models/example_chat.agent

# Compile DSPy-optimised examples into a sidecar file
python -m agent.cli compile models/example_full.agent
```

## Architecture

This is a **DSL compiler and runtime** for specifying AI agent systems via `.agent` files. The pipeline is:

```
.agent file → Grammar (textX) → Object Processors → Metamodel → Validation → IR → Code Generation → Runtime
```

### Compiler Pipeline

1. **Grammar** (`grammar/agent.tx`) — textX grammar defining the `.agent` syntax. Core constructs: `Model`,
   `ExecutorSyntax`, `ChatAgent`, `Rule`, `Skill`, `Example`, `OutputSpec`.

2. **Parsing** (`agent/parser.py`) — Entry point for loading the grammar and parsing `.agent` files into textX model
   objects.

3. **Object Processors** (`agent/processors.py`) — Post-parse transformations: converts `ruleType` strings (`do`/`dont`)
   to a boolean `negative` attribute, renames `params` → `skillArguments`, builds `OutputSpec`/`ExampleCommand`
   metamodel objects, and assembles the `System` from the raw textX model.

4. **Metamodel** (`agent/metamodel.py`) — Stable Python classes (`System`, `Planner`, `Executor`, `Task`, `OutputSpec`,
   `OutputField`, `ChatModeAgent`, `ExampleCommand`, `ExampleCommandArgument`, `SkillArgument`) that decouple the rest
   of the compiler from textX internals.

5. **Validation** (`agent/validation.py`) — Semantic checks: required fields, duplicate names (rules/skills/executors),
   cross-reference existence, valid reasoning effort values, and chat agent executor references.

6. **IR** (`agent/ir.py`) — Converts `System` into a Jinja2-friendly flat dict with snake_case keys and resolved
   cross-references. Executors are keyed by task name. Output schema is split into `output_format` + `output_fields`.

7. **Code Generation** (`agent/codegen.py` + `agent/templates/generated_agent.py.j2`) — Renders a self-contained Python
   module from the IR. Generated modules embed the system spec as JSON and import `AgentSystemRuntime` from
   `agent.runtime`. An experimental portable bundle (with vendored dependencies) is also available via
   `agent/templates/portable_agent.py.j2`.

8. **Runtime** (`agent/runtime.py`) — Orchestrates planner/executor interactions via the OpenAI Agents SDK. Key
   classes and functions:
    - `AgentSystemRuntime` — main orchestration engine; handles single-executor, multi-executor (planner handoff), and
      chat modes
    - `build_planner_agent` — builds a routing Agent whose only job is to hand off to the right executor
    - `build_openai_agent` — builds an executor Agent with tools and structured output type
    - `ShellToolExecutor` — runs shell commands rendered from skill templates
    - `render_skill_command` — interpolates `<param>` placeholders in skill command strings using `shlex.quote`
    - `CallGraph` / `CallEdge` — directed call graph captured from a run; render via `.render_text()` or `.render_dot()`
    - `build_call_graph` — walks `sdk_result.new_items` to produce `User → Planner → Executor → tools → [output]` edges;
      repeated tool calls are numbered (`run_shell#2`); stored on `RunResult.call_graph`

9. **Schema** (`agent/schema.py`) — Parses and coerces structured output schemas (`output_format` + `output_fields`).
   Supports `json`, `toml`, `yaml`, `markdown`, and `string`. Optionally bridges to DSPy signatures if `dspy` is
   installed.

10. **DSPy Compile** (`agent/dspy_compile.py`) — Runs `BootstrapFewShot` on executors that declare examples and writes
    a sidecar `<model>.agent.compiled.json`. The runtime loads this automatically when `use_dspy=True`.

### Key Design Decisions

- The **Planner** is implicit — auto-created from global `llm`/`reasoning` defaults. `reasoning` is optional.
- **Executor + Task** are merged into a single DSL declaration (`TaskName : "persona" { ... }`).
- Per-executor `llm`/`reasoning` override globals when specified.
- Generated agents are self-contained and runnable without the compiler (they only need `agent.runtime`).
- The runtime requires an OpenAI API key — there is no offline fallback.
- Output schema uses a typed block DSL: `output: json { field: type, ... }` or `output: markdown`.
- Example commands are structured: `tool_name(arg: "value")` instead of opaque strings.

### Provider Configuration

Set in shell environment or `.env` at repo root (or next to the `.agent` file):

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # optional, defaults to OpenAI
```

### Example Models

- `models/example_minimal.agent` — minimal valid file
- `models/example_full.agent` — full-featured (two executors, rules, skills, examples)
- `models/example_chat.agent` — demonstrates the `chat` block for interactive intake
- `models/agent_composer/` — multi-executor demo with planner routing
- `models/data_visualizer/` — end-to-end runnable demo with file I/O skills
- `models/invalid_example_*.agent` — invalid files used in validation tests

### VSCode Extension

`vscode-extension/` contains a language extension with LSP-based validation. `agent/lsp_validate.py` is the LSP
backend — it validates `.agent` files and outputs JSON diagnostics to stdout.