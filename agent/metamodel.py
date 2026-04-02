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
        self.outputSchema = "string"
        self.examples = []
        self.skills = []


class SkillArgument:
    def __init__(self, name=None, description=None):
        self.name = name
        self.description = description
