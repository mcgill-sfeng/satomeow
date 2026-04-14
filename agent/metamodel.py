"""
Metamodel classes for the Agent DSL.

These plain Python classes define the fixed object structure that the parser,
validator, and IR generator all operate on. The v1 grammar creates equivalent
objects dynamically via textX; the v2 grammar uses object processors to
construct these explicitly.
"""


class System:
    def __init__(self):
        self.planner: Planner = None
        self.executors: list[Executor] = []
        self.rules: list = []
        self.skills: list = []
        self.chatAgent: ChatModeAgent | None = None


class Planner:
    def __init__(self):
        self.llm: str = None
        self.reasoningStrategy: str = None
        self.persona: str = None
        self.rules: list = []


class Executor:
    def __init__(self):
        self.llm: str = None
        self.reasoningStrategy: str = None
        self.persona: str = None
        self.rules: list = []
        self.task: Task = None


class Task:
    def __init__(self):
        self.name: str = None
        self.inputDescription: str = None
        self.behavior: str = None
        # outputSpec is an OutputSpec instance; None means default ("string" text mode).
        self.outputSpec: OutputSpec = None
        self.examples: list = []
        self.skills: list = []


class OutputSpec:
    """Parsed output specification from an executor block.

    format: one of 'json', 'toml', 'yaml', 'markdown', 'string'
    fields: list of OutputField — non-empty only for structured formats
    """

    def __init__(self, format: str = "string", fields: list | None = None):
        self.format: str = format
        self.fields: list[OutputField] = fields or []

    @property
    def is_structured(self) -> bool:
        return self.format in ("json", "toml", "yaml")

    @property
    def is_text(self) -> bool:
        return not self.is_structured


class OutputField:
    """A single field in a structured output spec."""

    def __init__(self, name: str = "", type: str = "str"):
        self.name: str = name
        self.type: str = type


class ChatModeAgent:
    """Interactive chat intake agent defined with the 'chat' keyword.

    Drives a structured Q&A loop, then hands off to an executor for one-shot execution.
    """

    def __init__(self):
        self.name: str = None
        self.persona: str = None
        self.llm: str = None
        self.reasoningStrategy: str = None
        self.goal: str = None
        self.questions: list = []
        self.executor_ref: str | None = None  # optional: name of executor to hand off to


class SkillArgument:
    def __init__(self, name=None, description=None):
        self.name: str = name
        self.description: str = description


class ExampleCommandArgument:
    def __init__(self, name=None, value=None):
        self.name: str = name
        self.value: str = value


class ExampleCommand:
    def __init__(self, toolName=None, arguments=None):
        self.toolName: str = toolName
        self.arguments: list[ExampleCommandArgument] = arguments or []
