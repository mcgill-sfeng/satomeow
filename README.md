# Agent DSL

## Overview

This project implements a Domain-Specific Language (DSL) for specifying agent-based systems and generating structured inputs for Large Language Model (LLM) agents.

The DSL provides a structured and systematic way to define:

* Agent roles (Planner and Executor)
* Tasks and their behavior
* Operational rules and constraints
* Few-shot examples
* External tools (skills)

The goal is to reduce the complexity of prompt engineering and enable users to describe agent behavior at a higher level of abstraction.


## Implementation

This repository currently includes a **complete DSL frontend pipeline**, consisting of:

### 1. Grammar Definition (textX)

* A formal grammar (`agent.tx`) defined using textX
* Covers all core DSL constructs:

  * `System`
  * `Planner`
  * `Executor`
  * `Task`
  * `Rule`
  * `Skill`
  * `TaskExample`
  * `SkillArgument`


### 2. Parsing (Metamodel & Model)

* DSL files (`.agent`) are parsed into Python object models
* Implemented using `textx.metamodel_from_file` and `model_from_file`
* Supports:

  * Nested structures
  * Cross-references (e.g., Rule, Skill)
  * Optional and repeated elements


### 3. Semantic Validation

* Custom validation layer implemented in `agent/validation.py`
* Ensures correctness beyond syntax

Validation includes:

* Required field checks
* Duplicate detection:

  * Rule names
  * Skill names
  * SkillArgument names (within a skill)
* Structural constraints:

  * TaskExample must contain at least one command
* Default value handling (e.g., `outputSchema`)


### 4. Model Contract

* A stable object structure is defined for downstream usage
* Documented in `models/README.md`
* Ensures transformation logic does not depend on internal parsing details


### 5. Prompt IR (Intermediate Representation)

* A transformation-friendly intermediate representation is implemented in `agent/ir.py`
* Converts textX model into a clean dictionary structure

Features:

* Flattened structure
* Consistent naming (snake_case)
* Cross-references resolved
* Ready for Jinja2 templates


### 6. CLI Interface

* A simple command-line interface is provided (`agent/cli.py`)
* Allows users to:

```bash
python -m agent.cli models/example_full.agent
```

Options:

* `--print-model-info`
* `--print-ir`


### 7. Example DSL Files

Located in `models/`:

* `example_minimal.agent`
* `example_full.agent`
* Several invalid examples for testing validation

These examples cover all core language constructs.


### 8. Testing

Implemented using `pytest`.

Test coverage includes:

* Grammar loading
* Parsing valid DSL files
* Validation failure cases
* Cross-reference resolution
* IR generation

Run tests:

```bash
pytest
```

## Usage

### Parse and Validate a DSL File

```python
from agent.parser import parse_model

system = parse_model("models/example_full.agent")
```

---

### Build Prompt IR

```python
from agent.ir import build_prompt_ir

ir = build_prompt_ir(system)
```

---

### Run CLI

```bash
python -m agent.cli models/example_full.agent --print-ir
```


