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

# Run an agent directly (requires OPENAI_BASE_URL and OPENAI_API_KEY in env or .env)
python -m agent.cli run models/example_full.agent "Compare the REST and GraphQL APIs of GitHub"
```

## Architecture

This is a **DSL compiler and runtime** for specifying AI agent systems via `.agent` files. The pipeline is:

```
.agent file → Grammar (textX) → Object Processors → Metamodel → Validation → IR → Code Generation → Runtime
```

### Compiler Pipeline

1. **Grammar** (`grammar/agent.tx`) — textX grammar defining the `.agent` syntax. Core constructs: `Model`,
   `ExecutorSyntax`, `Rule`, `Skill`, `Example`.

2. **Parsing** (`agent/parser.py`) — Entry point for loading the grammar and parsing `.agent` files into textX model
   objects.

3. **Object Processors** (`agent/processors.py`) — Post-parse transformations: converts `ruleType` strings (`do`/`dont`)
   to a boolean `negative` attribute, renames `params` → `skillArguments`, and builds the `System` metamodel from the
   raw textX model.

4. **Metamodel** (`agent/metamodel.py`) — Stable Python dataclasses (`System`, `Planner`, `Executor`, `Task`,
   `SkillArgument`) that decouple the rest of the compiler from textX internals.

5. **Validation** (`agent/validation.py`) — Semantic checks: required fields, duplicate names (rules/skills/executors),
   and cross-reference existence.

6. **IR** (`agent/ir.py`) — Converts `System` into a Jinja2-friendly flat dict with snake_case keys and resolved
   cross-references.

7. **Code Generation** (`agent/codegen.py` + `agent/templates/generated_agent.py.j2`) — Renders a self-contained Python
   module from the IR. Generated modules embed the system spec as JSON and import `AgentSystemRuntime` from
   `agent.runtime`.

8. **Runtime** (`agent/runtime.py`) — Orchestrates planner/executor interactions. Key classes:
    - `AgentSystemRuntime` — main orchestration engine
    - `HeuristicPlanner` — selects executor by token overlap
    - `OpenAICompatibleModel` — LLM client for OpenAI-compatible APIs
    - `ExampleDrivenModel` — deterministic fallback using few-shot examples (no API needed)
    - `ShellToolExecutor` — runs shell commands extracted from `<tool_call>...</tool_call>` or `TOOL_CALL: ...` LLM
      responses

9. **Schema** (`agent/schema.py`) — Parses and coerces structured output schemas (`answer: str, confidence: float`).
   Optionally bridges to DSPy signatures if `dspy` is installed.

### Key Design Decisions

- The **Planner** is implicit — auto-created from global `llm`/`reasoning` defaults.
- **Executor + Task** are merged into a single DSL declaration (`TaskName : "persona" { ... }`).
- Per-executor `llm`/`reasoning` override globals when specified.
- Generated agents are self-contained and runnable without the compiler (they only need `agent.runtime`).
- Without a provider configured, the runtime falls back to `ExampleDrivenModel` (deterministic, example-driven).

### Provider Configuration

Set in shell environment or `.env` at repo root (or next to the `.agent` file):

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
```

### Example Models

- `models/example_minimal.agent` — minimal valid file
- `models/example_full.agent` — full-featured (two executors, rules, skills, examples)
- `models/data_visualizer/` — end-to-end runnable demo
- `models/invalid_example_*.agent` — invalid files used in validation tests

### VSCode Extension

`vscode-extension/` contains a language extension with LSP-based validation. `agent/lsp_validate.py` is the LSP
backend — it validates `.agent` files and outputs JSON diagnostics to stdout.