# Agent DSL

## Overview

This project implements a Domain-Specific Language (DSL) for specifying agent-based systems, generating runnable Python
agents, and executing them from the command line.

The DSL provides a structured and systematic way to define:

- Agent roles (Planner and Executors)
- Tasks and their behavior
- Operational rules and constraints
- Few-shot examples
- External tools (skills)

The goal is to reduce the complexity of prompt engineering and enable users to describe agent behavior at a higher level
of abstraction.

## Quick Start

After cloning the repository:

1. Install the project dependencies:

```bash
pip install -r requirements.txt
```

2. Configure an OpenAI-compatible provider in either `./.env` or `models/data_visualizer/.env`:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
```

3. Run the data visualizer demo:

```bash
python -m agent.cli run models/data_visualizer/data_visualizer.agent --verbose \
  "Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json"
```

Expected result:

- Verbose output shows the selected executor, tool calls, and raw model responses
- The final JSON contains `status: "success"`
- The chart artifact is created at `models/data_visualizer/output/aligned_sales_chart.svg`

4. Run the multi-executor `agent_composer` demo in one-shot mode:

```bash
python -m agent.cli run models/agent_composer/agent_composer.agent --json \
  "Validate models/data_visualizer/data_visualizer.agent"
```

Expected result:

- `executor_name` is `Validator`
- `output.valid` is `true`
- `output.errors` is an empty string

You can also generate a new `.agent` file in one shot:

```bash
python -m agent.cli run models/agent_composer/agent_composer.agent \
  "Create an agent that drafts short polite email replies. Input: an incoming email plus a desired tone. Output: string. No shell tools needed. Save it to models/agent_composer/generated/email_reply_drafter.agent"
```

Expected result:

- The request is routed to `Composer`
- The tool calls include `cat DSL_REFERENCE.md`, a file write, and `python -m agent.cli inspect ...`
- A valid file is created at `models/agent_composer/generated/email_reply_drafter.agent`

5. Run the same `agent_composer` demo in interactive chat mode:

```bash
python -m agent.cli chat models/agent_composer/agent_composer.agent --verbose
```

Then answer the prompts like this:

```text
[1/5] What should the agent do? Describe the task in one or two sentences.
> Draft short polite email replies.

[2/5] What input does the agent accept? (e.g. a file path, a code snippet, a search query)
> An incoming email plus a desired tone.

[3/5] What output format do you want? (string, markdown, or json with named fields)
> string

[4/5] Does the agent need to run shell commands or scripts to complete its task? If so, briefly describe what tools it needs.
> No shell tools are needed.

[5/5] Any rules or constraints the agent must follow? (optional — press Enter to skip)
> Keep replies under 120 words.
```

The runtime will print a confirmation message. Confirm with:

```text
Proceed? [Y/n] y
```

Expected result:

- Verbose output shows the bundled chat input, then `Composer` is executed directly
- The tool calls include `cat DSL_REFERENCE.md` and `python -m agent.cli inspect ...`
- A new validated file is written at `models/agent_composer/generated/email_reply_drafter.agent`
- The final line reports `Agent file created and validated successfully`

6. DSPy prompt compilation is a separate compile-time step. First compile the examples into a sidecar:

```bash
python -m agent.cli compile models/data_visualizer/data_visualizer.agent --model openai/gpt-5.4-nano
```

Then run with `--dspy`:

```bash
python -m agent.cli run models/data_visualizer/data_visualizer.agent --dspy --verbose \
  "Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json"
```

Important note:

- `--dspy` only changes the runtime prompt when `models/data_visualizer/data_visualizer.agent.compiled.json` exists
- Without a compiled sidecar, `--dspy` only adds lightweight guidance text around the original examples
- For tool-heavy tasks like `data_visualizer`, the current DSPy path is experimental and should be validated by inspecting the composed prompt

## DSL Syntax

For the full language reference, see **[DSL_REFERENCE.md](DSL_REFERENCE.md)**.

A `.agent` file begins with a required global `llm` default plus an optional global `reasoning` default, followed by any number of executor declarations, chat declarations,
rules, and skills in any order.

```text
llm: "<model-id>"
reasoning: "<effort>"          // optional; one of: none | minimal | low | medium | high | xhigh

TaskName : "persona" {
    llm: "<override>"           // optional — inherits global if omitted
    reasoning: "<override>"     // optional — same enum, inherits global if omitted
    input: "<description>"
    behavior: "<description>"
    output: string              // optional — string | markdown | json { ... } | toml { ... } | yaml { ... }
    skills: [skill1, skill2]    // optional
    rules: [rule1, rule2]       // optional

    example {
        input: "..."
        commands: ["cmd1", "cmd2"]
        output: "..."
    }
}

chat AgentName : "persona" {
    llm: "<override>"           // optional — inherits global if omitted
    reasoning: "<override>"     // optional — same enum, inherits global if omitted
    goal: "<one-sentence goal>"
    questions: ["q1", "q2"]     // at least one question
    executor: SomeExecutor      // optional
}

do rule_name: "positive constraint description"
dont rule_name: "negative constraint description"

skill_name {
    command: "<shell command template with <param> placeholders>"
    description: "<description>"
    param param_name: "<description>"
}
```

**Key design decisions:**

- The **Planner** is implicit — it is auto-created from the global `llm` default and optional `reasoning` default.
- **Executor + Task** are merged into a single declaration using `TaskName : "persona" { ... }`.
- **Rules** are prefixed with `do` (positive) or `dont` (negative).
- **Skills** are bare identifier blocks — no keyword prefix required.
- Per-executor `llm` and `reasoning` override the global defaults when specified.
- `reasoning` maps to the OpenAI Agents SDK reasoning effort setting and is limited to `none | minimal | low | medium | high | xhigh`.
- If `reasoning` is omitted, the runtime does not send an explicit reasoning effort to the SDK.
- **Skills are optional** — pure LLM tasks (text conversion, Q&A, writing) need no `skills` block at all.

## Output Formats

The `output:` field supports five formats:

| Format | Example | Description |
|---|---|---|
| `string` | `output: string` | Plain text (default) |
| `markdown` | `output: markdown` | Markdown-formatted text |
| `json` | `output: json { status: str, count: int }` | JSON object, schema-validated, Pydantic-enforced |
| `toml` | `output: toml { name: str, deps: list[str] }` | TOML, field-validated |
| `yaml` | `output: yaml { key: str, value: float }` | YAML, field-validated |

Allowed field types: `str`, `int`, `float`, `bool`, `list[str]`, `list[int]`, `list[float]`, `list[bool]`.

See [DSL_REFERENCE.md](DSL_REFERENCE.md) for full details, constraints, and common error causes.

## Provider Configuration

`python -m agent.cli run ...` currently supports OpenAI-compatible APIs only.

Set these in your shell or in a `.env` file at the repository root or next to the `.agent` file:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
```

If no provider is configured, `agent.cli run` fails fast with a clear message instead of silently falling back.

## Implementation

This repository includes the frontend compiler stages plus a code-generated runtime:

### 1. Grammar Definition (textX)

- Formal grammar defined in `grammar/agent.tx` using textX
- Core constructs: `Model`, `ExecutorSyntax`, `Rule`, `Skill`, `Example`, `Param`

### 2. Metamodel

- Plain Python classes defined in `agent/metamodel.py`
- Provides a stable, fixed object interface (`System`, `Planner`, `Executor`, `Task`, `SkillArgument`) shared by
  parsing, validation, and IR generation

### 3. Object Processors

- Implemented in `agent/processors.py`
- `process_rule` — converts `ruleType` string (`'do'`/`'dont'`) to a boolean `negative` attribute
- `process_skill` — renames `params` to `skillArguments`
- `build_system_from_model` — constructs the `System` metamodel object from the parsed `Model`

### 4. Parsing

- Entry point: `agent/parser.py`
- `load_metamodel()` — loads `grammar/agent.tx` and registers object processors
- `parse_model(path)` — parses and validates a `.agent` file, returning a `System`

### 5. Semantic Validation

- Implemented in `agent/validation.py`
- Required field checks
- Duplicate detection: rule names, skill names, skill argument names, executor/task names
- Default value handling (`output` defaults to `"string"` when omitted)

### 6. Prompt IR (Intermediate Representation)

- Implemented in `agent/ir.py`
- Converts the `System` object into a Jinja2-friendly dictionary
- Flattened structure, consistent snake_case naming, cross-references resolved

### 7. Runtime and Code Generation

- `agent/codegen.py` renders runnable Python modules from the IR using Jinja2
- `agent/runtime.py` maps executors to `openai-agents-python` agents, compiles DSL skills into SDK function tools, and
  compiles structured output schemas into Pydantic output models
- The OpenAI Agents SDK owns the model/tool loop at runtime
- `python -m agent.cli compile ...` writes a `*.compiled.json` sidecar with DSPy-compiled demonstrations
- `--dspy` loads that sidecar when present and swaps the executor prompt examples to the compiled demonstrations
- Without a sidecar, `--dspy` is only a light prompt-phrasing change and should not be treated as a proven quality gain

### 8. CLI Interface

```bash
python -m agent.cli inspect models/example_full.agent --print-ir
python -m agent.cli generate models/example_full.agent --output generated_agent.py
python -m agent.cli run models/example_full.agent "Compare the REST and GraphQL APIs of GitHub"
python -m agent.cli compile models/data_visualizer/data_visualizer.agent --model openai/gpt-5.4-nano
python -m agent.cli run models/example_full.agent --dspy "Compare the REST and GraphQL APIs of GitHub"
```

Legacy inspection mode is still supported:

```bash
python -m agent.cli models/example_full.agent --print-ir
```

### 9. Example DSL Files

Located in `models/`:

- `example_minimal.agent` — minimal valid file (one executor, no skills or rules)
- `example_full.agent` — full-featured example (two executors, rules, skills, multiple examples)
- `data_visualizer/` — end-to-end runnable demo for environment setup, visualization, preprocessing, and structured
  failure reporting
- `invalid_example_*.agent` — invalid files used to test validation errors

### 10. Testing

```bash
pytest
```

Test coverage includes:

- Grammar loading and metamodel construction
- Parsing valid DSL files (minimal and full)
- Default inheritance and per-executor overrides
- Rule `negative` attribute computation (`do`/`dont`)
- Skill argument conversion (`params` → `skillArguments`)
- Cross-reference resolution (rules and skills referenced by executors)
- Validation failure cases (duplicates, empty commands, missing required fields)
- IR structure and field values
- Code generation
- Runtime SDK wiring, shell tool execution, and structured output coercion
- CLI generation and run flows

## Usage

### Parse and Validate a DSL File

```python
from agent.parser import parse_model

system = parse_model("models/example_full.agent")
print(system.planner.llm)  # "gpt-5.4-nano"
print(system.executors[0].persona)  # "research agent"
```

### Build Prompt IR

```python
from agent.ir import build_prompt_ir

ir = build_prompt_ir(system)
# ir["planner"], ir["executors"], ir["rules"], ir["skills"]
```

### Generate Runnable Code

```bash
python -m agent.cli generate models/example_full.agent --output generated_agent.py
python generated_agent.py "Compare the REST and GraphQL APIs of GitHub"
```

### Run Without Saving Generated Code

```bash
python -m agent.cli run models/example_full.agent "Compare the REST and GraphQL APIs of GitHub"
```

### Data Visualizer Demo

```bash
python -m agent.cli run models/data_visualizer/data_visualizer.agent \
  "Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json"
```
