"""
Metamodel classes for the Agent DSL.

These plain Python classes define the fixed object structure that the parser
and validator operate on, and that a generated module's ``_build_system()``
instantiates directly.
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
    def __init__(self, llm=None, reasoningStrategy=None, persona=None, rules=None, task=None):
        self.llm: str = llm
        self.reasoningStrategy: str = reasoningStrategy
        self.persona: str = persona
        self.rules: list = rules if rules is not None else []
        self.task: Task = task


class Task:
    def __init__(
        self,
        name=None,
        inputDescription=None,
        behavior=None,
        outputSpec=None,
        examples=None,
        skills=None,
    ):
        self.name: str = name
        self.inputDescription: str = inputDescription
        self.behavior: str = behavior
        # outputSpec is an OutputSpec instance; None means default ("string" text mode).
        self.outputSpec: OutputSpec = outputSpec
        self.examples: list = examples if examples is not None else []
        self.skills: list = skills if skills is not None else []


class OutputSpec:
    """Parsed output specification from an executor block.

    format: one of 'json', 'toml', 'yaml', 'markdown', 'string'
    fields: list of OutputField — non-empty only for structured formats
    """

    def __init__(self, format: str = "string", fields: list | None = None):
        self.format: str = format
        self.fields: list[OutputField] = fields or []


class OutputField:
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
        self.executor_ref: str | None = None


class Rule:
    def __init__(self, name=None, negative=False, description=None):
        self.name: str = name
        self.negative: bool = negative
        self.description: str = description


class Skill:
    """A shell skill with interpolated arguments."""

    def __init__(self, name=None, command=None, description=None, skillArguments=None):
        self.name: str = name
        self.command: str = command
        self.description: str = description
        self.skillArguments: list[SkillArgument] = skillArguments if skillArguments is not None else []


class Example:
    """A task example (input → command trajectory → output)."""

    def __init__(self, input=None, output=None, commands=None):
        self.input: str = input
        self.output: str = output
        self.commands: list[ExampleCommand] = commands if commands is not None else []


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
