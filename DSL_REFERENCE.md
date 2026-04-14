# Agent DSL Reference

This document is the authoritative reference for the `.agent` file format. It covers every construct, every constraint, and every output type. The grammar source of truth is `grammar/agent.tx`.

---

## Table of Contents

1. [File Structure](#file-structure)
2. [Global Header](#global-header)
3. [Executor Declaration](#executor-declaration)
4. [Chat Agent](#chat-agent)
5. [Output Specification](#output-specification)
6. [Rules](#rules)
7. [Skills](#skills)
8. [Examples](#examples)
9. [Field Types](#field-types)
10. [Multi-Executor Files and Routing](#multi-executor-files-and-routing)
11. [Constraints and Common Errors](#constraints-and-common-errors)
12. [Complete Annotated Example](#complete-annotated-example)

---

## File Structure

A `.agent` file has exactly this top-level shape:

```
<global header>          -- required, always first
<items>*                 -- any order: executors, rules, skills
```

Items can appear in any order after the header. There is no required ordering between executors, rules, and skills, but by convention executors come first, rules second, skills last.

---

## Global Header

The file starts with a required `llm` line and may optionally include a `reasoning` line:

```
llm: "model-name"
reasoning: "effort"
```

| Field | Type | Description |
|---|---|---|
| `llm` | string | Default model ID for all executors. Example: `"gpt-5.4-nano"`. |
| `reasoning` | enum | Optional default reasoning effort passed to the OpenAI Agents SDK. Allowed values: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`. |

`llm` is required and must appear before any item declarations. `reasoning` is optional; if omitted, the runtime leaves reasoning effort unset.

---

## Executor Declaration

An executor is the primary working unit of the agent system. Syntax:

```
TaskName : "persona string" {
    llm: "override-model"           -- optional
    reasoning: "override-effort"    -- optional
    input: "what this executor accepts"
    behavior: "what this executor does"
    output: <OutputSpec>            -- optional, defaults to string
    skills: [skill1, skill2]        -- optional
    rules: [rule1, rule2]           -- optional

    example { ... }                 -- optional, repeatable
}
```

### Required fields

| Field | Description |
|---|---|
| `TaskName` | A bare identifier (no quotes). Used as the executor's canonical name. |
| `: "persona string"` | A quoted string describing the executor's role. **The colon and quoted persona are required.** Omitting them causes the parser to treat the block as a Skill, producing an `Expected 'command'` error. |
| `input` | A quoted string describing what the executor accepts as input. |
| `behavior` | A quoted string describing what the executor does. This becomes the system prompt. |

### Optional fields

| Field | Description |
|---|---|
| `llm` | Override the global model for this executor only. |
| `reasoning` | Override the global reasoning effort for this executor only. Must use the same enum. |
| `output` | Output format specification. See [Output Specification](#output-specification). Defaults to `string`. |
| `skills` | Comma-separated list of skill names defined elsewhere in the file. |
| `rules` | Comma-separated list of rule names defined elsewhere in the file. |

### Skills and rules are optional

If an executor only needs the LLM to produce text (Q&A, summarization, conversion, writing), omit the `skills` block entirely. Only include skills when the executor truly needs shell tools to complete its task.

---

## Chat Agent

A `chat` block defines an **interactive intake agent**. It is a different kind of construct from an executor: instead of processing a single input and returning output, it drives a structured multi-turn conversation to collect information from the user, then hands off to an executor for one-shot execution.

**`agent cli chat`** requires a `chat` block in the model — it fails with a clear error if none is present.

### Syntax

```
chat AgentName : "persona" {
    llm: "override-model"       // optional
    reasoning: "medium"         // optional
    goal: "one-sentence description of what task will be performed"
    questions: [
        "First question to ask the user?",
        "Second question?",
        "Third question?"
    ]
    executor: TargetExecutorName  // optional — omit to let the planner route
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `AgentName` | Yes | Bare identifier. Used as the agent's canonical name. |
| `: "persona"` | Yes | Quoted string describing the agent's role. Required (same rule as executors). |
| `goal` | Yes | Describes the task that will be performed once all information is collected. Used by the LLM to generate the confirmation message. |
| `questions` | Yes | Ordered list of quoted strings. The runtime asks them one at a time. At least one question is required. |
| `executor` | No | Name of the executor to hand off to after confirmation. If omitted, the planner routes based on the collected input. |
| `llm` / `reasoning` | No | Override the global model / reasoning effort for the confirmation LLM call. |

### Runtime behaviour

1. The CLI (`agent cli chat`) presents each question in order and reads the user's answer.
2. Once all questions are answered, a single LLM call generates a natural confirmation message summarizing what will be done.
3. The user is asked to confirm ("Proceed? [Y/n]").
4. On **yes**: the collected Q&A is bundled into a structured input and sent to the executor (or planner if `executor:` is omitted) as a one-shot run.
5. On **no**: the session is cancelled.

### Example

```
chat Intake : "requirements analyst" {
    goal: "Generate a complete .agent file for the described task"
    questions: [
        "What should the agent do?",
        "What input does it accept?",
        "What output format do you need?",
        "Does it need shell tools? If so, briefly describe them.",
        "Any rules or constraints? (optional)"
    ]
    executor: Composer
}
```

---

## Output Specification

The `output:` field accepts one of three forms.

### Text formats (no fields)

```
output: string
output: markdown
```

- `string` — plain text output, no validation.
- `markdown` — Markdown-formatted text, no validation.

### Structured formats (with typed field block)

```
output: json {
    field1: str,
    field2: int,
    field3: bool,
    field4: list[str]
}
```

Replace `json` with `toml` or `yaml` for those formats.

| Format | Behavior |
|---|---|
| `json` | LLM is instructed to return a JSON object. Output is validated against the field schema. A Pydantic model is enforced by the SDK. |
| `toml` | LLM is instructed to return valid TOML. Output is parsed with `tomllib` and field presence is checked. |
| `yaml` | LLM is instructed to return valid YAML. Output is parsed with `yaml` and field presence is checked. |

Fields within the block are comma-separated `name: type` pairs. See [Field Types](#field-types) for allowed types.

### Default

If `output:` is omitted, the executor defaults to `string`.

---

## Rules

Rules constrain executor behavior. They are declared at the file level and referenced by name in executor `rules:` lists.

```
do rule_name: "positive constraint description"
dont rule_name: "negative constraint description"
```

| Prefix | Meaning |
|---|---|
| `do` | Positive constraint — the executor should always do this. |
| `dont` | Negative constraint — the executor must never do this. |

**Naming:** `rule_name` is a bare identifier. Names must be unique across the file.

**Referencing:** List the rule name (without `do`/`dont`) in the executor's `rules: [...]` array.

**Examples:**

```
do verify_sources: "always verify sources before citing"
dont no_hallucination: "do not make up facts"
dont skip_validation: "always run inspect after writing a .agent file"
```

---

## Skills

Skills expose shell commands as tools available to executors. They are declared at the file level and referenced by name in executor `skills:` lists.

```
skill_name {
    command: "shell command with <param_name> placeholders"
    description: "what this skill does"
    param param1: "description of param1"
    param param2: "description of param2"
}
```

### Constraints — MUST follow or the parser will fail

1. **`command:` must be a single-line string.** No newlines, no heredocs, no multi-line strings. Use `python scripts/foo.py <arg>` — not `python -c "..."` with embedded newlines.

2. **`<paramName>` placeholders in `command:` must exactly match defined `param` names.** The runtime substitutes each `<name>` with the value the LLM provides for that parameter.

3. **`description:` must come immediately after `command:`.** Swapping them causes a parse error.

4. **Skill names must be unique** across the file.

5. **Skills are optional.** For pure LLM tasks (text conversion, analysis, Q&A, writing), do not include a `skills` block in the executor at all. Skills are only needed when the executor must execute shell commands.

**Example:**

```
fetchPage {
    command: "curl -s <url>"
    description: "Fetch the raw HTML of a web page"
    param url: "The URL to fetch"
}

writeFile {
    command: "python scripts/write_file.py --path <path> --content <content>"
    description: "Write text content to a file at the given path"
    param path: "Destination file path"
    param content: "The full text content to write"
}
```

---

## Examples

Examples are optional few-shot demonstrations declared inside an executor block. The runtime uses them for prompt enrichment.

```
example {
    input: "example user input string"
    commands: [
        someTool(arg1: "value", arg2: "value")
    ]
    output: "expected output string"
}
```

| Field | Description |
|---|---|
| `input` | The example user input. Quoted string. |
| `commands` | List of structured tool calls. Use `[]` for no commands. Each entry must name a skill referenced by the executor and provide all of that skill's parameters. |
| `output` | The expected output. Quoted string. For JSON output, embed the JSON as a string. |

### Tool-call syntax

Each example command uses function-call syntax:

```
toolName(arg1: "value", arg2: "value")
```

- `toolName` must match a skill available to that executor
- Every skill parameter must be present exactly once
- Values are strings, because current runtime tool arguments are string-typed

Multiple `example` blocks can appear inside one executor. They are listed after `rules:` and before the closing `}`.

---

## Field Types

Used inside structured output blocks (`output: json { ... }`, `output: toml { ... }`, `output: yaml { ... }`).

| Type | Description |
|---|---|
| `str` | UTF-8 string |
| `int` | Integer |
| `float` | Floating-point number |
| `bool` | Boolean (`true`/`false`) |
| `list[str]` | List of strings |
| `list[int]` | List of integers |
| `list[float]` | List of floats |
| `list[bool]` | List of booleans |

---

## Multi-Executor Files and Routing

A `.agent` file can declare any number of executors. When multiple executors are present, the runtime auto-builds a planner agent that routes each user input to the most appropriate executor using SDK-native handoffs.

**The planner is implicit** — it is not declared in the DSL. It is auto-generated at runtime from the executor names, personas, and input descriptions.

**Single-executor files** bypass the planner entirely.

**Routing:** The planner reads each executor's `name`, `persona`, and `input` description to decide which executor handles a given user message. Write `input:` descriptions that clearly distinguish what each executor accepts.

---

## Constraints and Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Expected 'command'` | Executor block missing `: "persona"` — parsed as a Skill | Add `: "persona string"` after the executor name |
| `Expected 'description'` | `command:` and `description:` are swapped | Put `command:` first, then `description:` |
| Parse error on newline in string | `command:` or any STRING field contains a newline | Rewrite as a single-line string |
| `Skill 'x' referenced but not defined` | Executor `skills: [x]` but no `x { ... }` block exists | Define the skill block or remove the reference |
| `Rule 'x' referenced but not defined` | Executor `rules: [x]` but no `do`/`dont x:` line exists | Define the rule or remove the reference |
| `Duplicate executor name` | Two executor blocks with the same `TaskName` | Rename one |
| `Duplicate skill name` | Two skill blocks with the same name | Rename one |
| `Duplicate rule name` | Two `do`/`dont` declarations with the same name | Rename one |

---

## Complete Annotated Example

```
// Global defaults — required, always first
llm: "gpt-5.4-nano"
reasoning: "medium"

// --- Executor 1: pure LLM, no skills needed ---
Summarizer : "text summarization agent" {
    input: "A block of text or a topic to summarize"
    behavior: "Read the provided text carefully and return a concise summary"
    output: markdown
    rules: [be_concise]
}

// --- Executor 2: tool-using agent ---
DataFetcher : "data retrieval agent" {
    llm: "gpt-5.4-nano"         // per-executor model override
    reasoning: "medium"
    input: "A URL or search query for data to retrieve"
    behavior: "Fetch the requested data using available tools and return it verbatim"
    output: json {
        url: str,
        status_code: int,
        body: str
    }
    skills: [fetchPage]
    rules: [no_hallucination]

    example {
        input: "Fetch https://example.com"
        commands: [
            fetchPage(url: "https://example.com")
        ]
        output: "{\"url\": \"https://example.com\", \"status_code\": 200, \"body\": \"<html>...\"}"
    }
}

// --- Rules ---
do be_concise: "keep summaries under 150 words"
dont no_hallucination: "do not invent facts not present in the source"

// --- Skills ---
fetchPage {
    command: "curl -s <url>"
    description: "Fetch the raw content of a URL"
    param url: "The URL to fetch"
}
```

### Validation

After writing a `.agent` file, validate it with:

```bash
python -m agent.cli inspect path/to/file.agent --print-ir
```

A clean run prints the Prompt IR as JSON. Any parse or semantic error is reported with a line number and message.
