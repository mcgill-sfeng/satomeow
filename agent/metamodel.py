"""
Metamodel classes for the Agent DSL.

These plain Python classes define the fixed object structure that the parser,
validator, and IR generator all operate on. The v1 grammar creates equivalent
objects dynamically via textX; the v2 grammar uses object processors to
construct these explicitly.
"""


class System:
    def __init__(self):
        self.planner = None
        self.executors = []
        self.rules = []
        self.skills = []
        self.chat_agent = None  # ChatModeAgent | None


class Planner:
    def __init__(self):
        self.llm = None
        self.reasoningStrategy = None
        self.persona = None
        self.rules = []


class Executor:
    def __init__(self):
        self.llm = None
        self.reasoningStrategy = None
        self.persona = None
        self.rules = []
        self.task = None


class Task:
    def __init__(self):
        self.name = None
        self.inputDescription = None
        self.behavior = None
        # outputSpec is an OutputSpec instance; None means default ("string" text mode).
        self.outputSpec = None
        self.examples = []
        self.skills = []


class OutputSpec:
    """Parsed output specification from an executor block.

    format: one of 'json', 'toml', 'yaml', 'markdown', 'string'
    fields: list of OutputField — non-empty only for structured formats
    """

    def __init__(self, format: str = "string", fields: list | None = None):
        self.format = format
        self.fields = fields or []

    @property
    def is_structured(self) -> bool:
        return self.format in ("json", "toml", "yaml")

    @property
    def is_text(self) -> bool:
        return not self.is_structured


class OutputField:
    """A single field in a structured output spec."""

    def __init__(self, name: str = "", type: str = "str"):
        self.name = name
        self.type = type


class ChatModeAgent:
    """Interactive chat intake agent defined with the 'chat' keyword.

    Drives a structured Q&A loop, then hands off to an executor for one-shot execution.
    """

    def __init__(self):
        self.name = None
        self.persona = None
        self.llm = None
        self.reasoningStrategy = None
        self.goal = None
        self.questions = []
        self.executor_ref = None  # optional: name of executor to hand off to


class SkillArgument:
    def __init__(self, name=None, description=None):
        self.name = name
        self.description = description


class ExampleCommandArgument:
    def __init__(self, name=None, value=None):
        self.name = name
        self.value = value


class ExampleCommand:
    def __init__(self, toolName=None, arguments=None):
        self.toolName = toolName
        self.arguments = arguments or []
