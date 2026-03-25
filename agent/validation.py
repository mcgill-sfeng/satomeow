from textx import TextXSemanticError


def validate_system(system):
    validate_rules(system.rules)
    validate_skills(system.skills)
    validate_planner(system.planner)
    for executor in system.executors:
        validate_executor(executor)


def validate_planner(planner):
    check_required(planner, "reasoningStrategy")
    check_required(planner, "llm")
    check_required(planner, "persona")


def validate_executor(executor):
    check_required(executor, "reasoningStrategy")
    check_required(executor, "llm")
    check_required(executor, "persona")

    if executor.task:
        validate_task(executor.task)


def validate_task(task):
    check_required(task, "name")
    check_required(task, "inputDescription")
    check_required(task, "behavior")

    if task.outputSchema is None:
        task.outputSchema = "string"

    for example in task.examples:
        validate_task_example(example)


def validate_task_example(example):
    check_required(example, "input")
    check_required(example, "output")

    if not example.commands:
        raise TextXSemanticError(
            "TaskExample must contain at least one command"
        )


def validate_rules(rules):
    seen = set()
    for rule in rules:
        check_required(rule, "name")
        check_required(rule, "description")

        if rule.name in seen:
            raise TextXSemanticError(
                f"Duplicate rule name: {rule.name}"
            )
        seen.add(rule.name)


def validate_skills(skills):
    seen = set()
    for skill in skills:
        check_required(skill, "name")
        check_required(skill, "command")
        check_required(skill, "description")

        if skill.name in seen:
            raise TextXSemanticError(
                f"Duplicate skill name: {skill.name}"
            )
        seen.add(skill.name)

        validate_skill_arguments(skill)


def validate_skill_arguments(skill):
    seen = set()

    for arg in skill.skillArguments:
        check_required(arg, "name")
        check_required(arg, "description")

        if arg.name in seen:
            raise TextXSemanticError(
                f"Duplicate SkillArgument name '{arg.name}' in skill '{skill.name}'"
            )
        seen.add(arg.name)


def check_required(obj, field):
    value = getattr(obj, field, None)

    if value is None:
        raise TextXSemanticError(
            f"{obj.__class__.__name__}.{field} is required"
        )

    if isinstance(value, str) and value.strip() == "":
        raise TextXSemanticError(
            f"{obj.__class__.__name__}.{field} is required"
        )