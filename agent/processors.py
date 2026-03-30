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

from agent.metamodel import System, Planner, Executor, Task, SkillArgument


def build_system_from_model(model):
    """
    Transform a parsed Model into a System object.

    Called explicitly after model_from_file() returns the Model.  By this
    point, all Rule and Skill processors have already run, so Rule objects
    carry a `negative` attribute and Skill objects carry `skillArguments`.

    Args:
        model: textX Model with llm, reasoningStrategy, and items list.

    Returns:
        System: Complete metamodel object ready for validation.
    """
    planner = Planner()
    planner.llm = model.llm
    planner.reasoningStrategy = model.reasoningStrategy
    planner.persona = "planner"
    planner.rules = []

    executors = []
    for item in model.items:
        if item.__class__.__name__ == "ExecutorSyntax":
            executor = Executor()
            executor.llm = item.llm or model.llm
            executor.reasoningStrategy = item.reasoningStrategy or model.reasoningStrategy
            executor.persona = item.persona
            executor.rules = list(item.rules) if item.rules else []
            executor._source_obj = item

            task = Task()
            task.name = item.name
            task.inputDescription = item.inputDescription
            task.behavior = item.behavior
            task.outputSchema = item.outputSchema if item.outputSchema else "string"
            task.examples = list(item.examples)
            task.skills = list(item.skills) if item.skills else []
            task._source_obj = item

            executor.task = task
            executors.append(executor)

    rules = [item for item in model.items if item.__class__.__name__ == "Rule"]
    skills = [item for item in model.items if item.__class__.__name__ == "Skill"]

    system = System()
    system.planner = planner
    system.executors = executors
    system.rules = rules
    system.skills = skills

    return system


def process_rule(rule):
    """
    Convert ruleType string ('do' or 'dont') to a boolean `negative` attribute.

    Args:
        rule: Parsed Rule with ruleType attribute.

    Returns:
        Rule: Same object with `negative` attribute added.
    """
    rule.negative = (rule.ruleType == "dont")
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
