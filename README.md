# Agent DSL

## Overview

This project implements a Domain-Specific Language (DSL) for specifying agent-based systems and generating structured inputs for Large Language Model (LLM) agents.

The DSL provides a structured and systematic way to define:

- Agent roles (Planner and Executors)
- Tasks and their behavior
- Operational rules and constraints
- Few-shot examples
- External tools (skills)

The goal is to reduce the complexity of prompt engineering and enable users to describe agent behavior at a higher level of abstraction.

## DSL Syntax

A `.agent` file begins with two global defaults, followed by any number of executor declarations, rules, and skills in any order.

```text
llm: "<model-id>"
reasoning: "<strategy>"

TaskName : "persona" {
    llm: "<override>"          // optional — inherits global if omitted
    reasoning: "<override>"    // optional — inherits global if omitted
    input: "<description>"
    behavior: "<description>"
    output: "<schema>"         // optional — defaults to "string"
    skills: [skill1, skill2]   // optional
    rules: [rule1, rule2]      // optional

    example {
        input: "..."
        commands: ["cmd1", "cmd2"]
        output: "..."
    }
}

do rule_name: "positive constraint description"
dont rule_name: "negative constraint description"

skill_name {
    command: "<shell command template>"
    description: "<description>"
    param param_name: "<description>"
}
```

**Key design decisions:**

- The **Planner** is implicit — it is auto-created from the global `llm` and `reasoning` defaults.
- **Executor + Task** are merged into a single declaration using `TaskName : "persona" { ... }`.
- **Rules** are prefixed with `do` (positive) or `dont` (negative).
- **Skills** are bare identifier blocks — no keyword prefix required.
- Per-executor `llm` and `reasoning` override the global defaults when specified.

## Implementation

This repository includes a complete DSL frontend pipeline:

### 1. Grammar Definition (textX)

- Formal grammar defined in `grammar/agent.tx` using textX
- Core constructs: `Model`, `ExecutorSyntax`, `Rule`, `Skill`, `Example`, `Param`

### 2. Metamodel

- Plain Python classes defined in `agent/metamodel.py`
- Provides a stable, fixed object interface (`System`, `Planner`, `Executor`, `Task`, `SkillArgument`) shared by parsing, validation, and IR generation

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
- Structural constraints: each `example` must contain at least one command
- Default value handling (`outputSchema` defaults to `"string"`)

### 6. Prompt IR (Intermediate Representation)

- Implemented in `agent/ir.py`
- Converts the `System` object into a Jinja2-friendly dictionary
- Flattened structure, consistent snake_case naming, cross-references resolved

### 7. CLI Interface

```bash
python -m agent.cli models/example_full.agent
python -m agent.cli models/example_full.agent --print-model-info
python -m agent.cli models/example_full.agent --print-ir
```

### 8. Example DSL Files

Located in `models/`:

- `example_minimal.agent` — minimal valid file (one executor, no skills or rules)
- `example_full.agent` — full-featured example (two executors, rules, skills, multiple examples)
- `invalid_example_*.agent` — invalid files used to test validation errors

### 9. Testing

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

## Usage

### Parse and Validate a DSL File

```python
from agent.parser import parse_model

system = parse_model("models/example_full.agent")
print(system.planner.llm)           # "gpt-5"
print(system.executors[0].persona)  # "research agent"
```

### Build Prompt IR

```python
from agent.ir import build_prompt_ir

ir = build_prompt_ir(system)
# ir["planner"], ir["executors"], ir["rules"], ir["skills"]
```

### Run CLI

```bash
python -m agent.cli models/example_full.agent --print-ir
```
