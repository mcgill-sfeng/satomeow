"""
Object processors and model transformation for the Agent DSL v2 grammar.

Processors (process_rule, process_skill) are registered with the textX
metamodel and run automatically during parsing, modifying objects in-place.

build_system_from_model() is called explicitly after model_from_file()
because textX does not propagate a replaced root object back through
model_from_file — returning a new object from a root processor has no effect.

Processing order (textX runs processors bottom-up):
  Param    (no-op, consumed by Skill processor)
  Rule     → adds `negative` boolean from `ruleType`
  Skill    → renames `params` to `skillArguments`
  Example  (no-op, structure already matches TaskExample interface)
  ExecutorSyntax (no-op; consumed by build_system_from_model)
  Model    (no-op at processor level; transformed by build_system_from_model)
"""

from agent.metamodel import (
    System,
    Planner,
    Executor,
    Task,
    OutputSpec,
    OutputField,
    SkillArgument,
    ChatModeAgent,
    ExampleCommand,
    ExampleCommandArgument,
)


def build_system_from_model(model):
    """
    Transform a parsed Model into a System object.

    Called explicitly after model_from_file() returns the Model.  By this
    point, all Rule and Skill processors have already run, so Rule objects
    carry a `negative` attribute and Skill objects carry `skillArguments`.

    Skills and rules are collected first so that executor skill/rule references
    (now plain ID lists) can be resolved by name lookup.  The raw ID lists are
    preserved on executor/task as `_rule_refs` / `_skill_refs` so that
    validation can report unknown-reference errors with a useful location.

    Args:
        model: textX Model with llm, reasoningStrategy, and items list.

    Returns:
        System: Complete metamodel object ready for validation.
    """
    rules = [item for item in model.items if item.__class__.__name__ == "Rule"]
    skills = [item for item in model.items if item.__class__.__name__ == "Skill"]
    chat_agents = [item for item in model.items if item.__class__.__name__ == "ChatAgent"]

    # First-occurrence wins; duplicates are caught by validate_skills/validate_rules.
    rules_by_name = {}
    for r in rules:
        if r.name not in rules_by_name:
            rules_by_name[r.name] = r

    skills_by_name = {}
    for s in skills:
        if s.name not in skills_by_name:
            skills_by_name[s.name] = s

    planner = Planner()
    planner.llm = model.llm
    planner.reasoningStrategy = model.reasoningStrategy
    planner.persona = "planner"
    planner.rules = []

    executors = []
    for item in model.items:
        if item.__class__.__name__ == "ExecutorSyntax":
            rule_refs = list(item.ruleRefs) if item.ruleRefs else []
            skill_refs = list(item.skillRefs) if item.skillRefs else []

            executor = Executor()
            executor.llm = item.llm or model.llm
            executor.reasoningStrategy = item.reasoningStrategy or model.reasoningStrategy
            executor.persona = item.persona
            executor.rules = [rules_by_name[r] for r in rule_refs if r in rules_by_name]
            executor._source_obj = item
            executor._rule_refs = rule_refs

            task = Task()
            task.name = item.name
            task.inputDescription = item.inputDescription
            task.behavior = item.behavior
            task.outputSpec = _build_output_spec(item.outputSpec)
            task.examples = [_build_example(example) for example in item.examples]
            task.skills = [skills_by_name[s] for s in skill_refs if s in skills_by_name]
            task._source_obj = item
            task._skill_refs = skill_refs

            executor.task = task
            executors.append(executor)

    chat_agent = None
    if chat_agents:
        raw = chat_agents[0]
        chat_agent = ChatModeAgent()
        chat_agent.name = raw.name
        chat_agent.persona = raw.persona
        chat_agent.llm = raw.llm or model.llm
        chat_agent.reasoningStrategy = raw.reasoningStrategy or model.reasoningStrategy
        chat_agent.goal = raw.goal
        chat_agent.questions = list(raw.questions) if raw.questions else []
        chat_agent.executor_ref = raw.executorRef if raw.executorRef else None
        chat_agent._source_obj = raw

    system = System()
    system.planner = planner
    system.executors = executors
    system.rules = rules
    system.skills = skills
    system.chat_agent = chat_agent

    return system


def _build_output_spec(raw_spec) -> OutputSpec:
    """Convert a textX OutputSpec object (or None) into a metamodel OutputSpec."""
    if raw_spec is None:
        return OutputSpec(format="string", fields=[])

    cls = raw_spec.__class__.__name__

    if cls == "TextOutputSpec":
        return OutputSpec(format=str(raw_spec.format), fields=[])

    if cls == "StructuredOutputSpec":
        fields = [OutputField(name=f.name, type=str(f.type)) for f in (raw_spec.fields or [])]
        return OutputSpec(format=str(raw_spec.format), fields=fields)

    # Fallback: unknown spec type — treat as plain string
    return OutputSpec(format="string", fields=[])


def _build_example(raw_example):
    raw_example.commands = [_build_example_command(command) for command in (raw_example.commands or [])]
    return raw_example


def _build_example_command(raw_command) -> ExampleCommand:
    command = ExampleCommand(toolName=raw_command.toolName)
    command._source_obj = raw_command
    command.arguments = []
    for raw_argument in raw_command.arguments or []:
        argument = ExampleCommandArgument(name=raw_argument.name, value=raw_argument.value)
        argument._source_obj = raw_argument
        command.arguments.append(argument)
    return command


def process_rule(rule):
    """
    Convert ruleType string ('do' or 'dont') to a boolean `negative` attribute.

    Args:
        rule: Parsed Rule with ruleType attribute.

    Returns:
        Rule: Same object with `negative` attribute added.
    """
    rule.negative = rule.ruleType == "dont"
    return rule


def process_skill(skill):
    """
    Rename `params` to `skillArguments`, converting Param objects to
    SkillArgument instances to match the metamodel interface.

    Args:
        skill: Parsed Skill with `params` list.

    Returns:
        Skill: Same object with `skillArguments` attribute added.
    """
    skill.skillArguments = []
    for p in skill.params:
        skill_arg = SkillArgument(name=p.name, description=p.description)
        skill_arg._source_obj = p
        skill.skillArguments.append(skill_arg)
    return skill
