# Model Contract for the Agent DSL

This document defines the object structure produced by textX after parsing a `.agent` file.

It is intended for the transformation stage, so the Jinja2/code generation team can reliably access model attributes
without guessing the parsed structure.

## Root Object

The root parsed object is:

- `system: System`

---

## Object Structure

### System

Represents the top-level system definition.

Attributes:

- `planner: Planner`
- `executors: list[Executor]`
- `rules: list[Rule]`
- `skills: list[Skill]`

---

### Agent

Abstract concept in the language design.

In the current textX grammar implementation, `Agent` is represented concretely as either:

- `Planner`
- `Executor`

Shared conceptual fields:

- `reasoningStrategy`  // stores the DSL `reasoning` field, i.e. SDK reasoning effort
- `llm`
- `persona`
- `rules`

These fields are implemented directly inside `Planner` and `Executor` in the grammar.

---

### Planner

Represents the planner agent.

Attributes:

- `reasoningStrategy: str`  // one of: none | minimal | low | medium | high | xhigh
- `llm: str`
- `persona: str`
- `rules: list[Rule]`  
  Cross-references to globally defined `Rule` objects.

---

### Executor

Represents an executor agent.

Attributes:

- `reasoningStrategy: str`  // one of: none | minimal | low | medium | high | xhigh
- `llm: str`
- `persona: str`
- `rules: list[Rule]`  
  Cross-references to globally defined `Rule` objects.
- `task: Task | None`  
  Contained object, not a cross-reference.

---

### Task

Represents the task owned by an `Executor`.

Attributes:

- `name: str`
- `inputDescription: str`
- `behavior: str`
- `outputSchema: str | None`
- `examples: list[TaskExample]`
- `skills: list[Skill]`  
  Cross-references to globally defined `Skill` objects.

Notes:

- `Task` is contained inside `Executor`
- `Task` is not globally defined at the root level

---

### Rule

Represents a reusable rule.

Attributes:

- `name: str`
- `negative: bool`
- `description: str`

Notes:

- `Rule` is globally defined under `model.rules`
- Other elements reference `Rule` objects by name

---

### TaskExample

Represents one few-shot example for a task.

Attributes:

- `input: str`
- `commands: list[str]`
- `output: str`

---

### Skill

Represents a reusable skill.

Attributes:

- `name: str`
- `command: str`
- `description: str`
- `skillArguments: list[SkillArgument]`

Notes:

- `Skill` is globally defined under `model.skills`
- Other elements reference `Skill` objects by name

---

### SkillArgument

Represents one argument of a skill.

Attributes:

- `name: str`
- `description: str`

---

## Containment vs Cross-Reference

### Containment

These objects are structurally contained inside their parent:

- `model.system`
- `system.planner`
- `system.executors`
- `executor.task`
- `task.examples`
- `skill.skillArguments`

### Cross-Reference

These fields refer to globally defined objects:

- `system.rules -> Rule`
- `system.skills -> Skill`
- `planner.rules -> Rule`
- `executor.rules -> Rule`
- `task.skills -> Skill`

---

## Example Access Patterns

```python
from textx import metamodel_from_file

mm = metamodel_from_file("grammar/agent.tx")
system = mm.model_from_file("models/example.agent")

planner_persona = system.planner.persona
executor_count = len(system.executors)
task_name = system.executors[0].task.name
system_rule_description = system.rules[0].description
system_skill_command = system.skills[0].command
```

## Parsing and Validation

The DSL front-end provides a stable entry point for parsing and validating `.agent` files.

Example:

```python
from agent.parser import parse_model

system = parse_model("models/example_full.agent")
print(system.planner.persona)
```

