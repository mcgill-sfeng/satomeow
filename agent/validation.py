from textx import TextXSemanticError, get_location


def validate_system(system):
    validate_rules(system.rules)
    validate_skills(system.skills)
    validate_planner(system.planner)
    validate_unique_executor_names(system.executors)
    known_skill_names = {s.name for s in system.skills}
    known_rule_names = {r.name for r in system.rules}
    for executor in system.executors:
        validate_executor(executor)
        validate_executor_refs(executor, known_skill_names, known_rule_names)


def validate_executor_refs(executor, known_skill_names, known_rule_names):
    """Check that all skill/rule names referenced by an executor are defined."""
    for ref in getattr(executor, "_rule_refs", []):
        if ref not in known_rule_names:
            _raise_semantic(f"Unknown rule reference: '{ref}'", executor)
    if executor.task:
        for ref in getattr(executor.task, "_skill_refs", []):
            if ref not in known_skill_names:
                _raise_semantic(f"Unknown skill reference: '{ref}'", executor.task)


def validate_unique_executor_names(executors):
    """Ensure task names are unique across all executors."""
    seen = set()
    for executor in executors:
        if executor.task:
            name = executor.task.name
            if name in seen:
                _raise_semantic(
                    f"Duplicate executor/task name: {name}",
                    executor.task,
                )
            seen.add(name)


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


def validate_rules(rules):
    seen = set()
    for rule in rules:
        check_required(rule, "name")
        check_required(rule, "description")

        if rule.name in seen:
            _raise_semantic(f"Duplicate rule name: {rule.name}", rule)
        seen.add(rule.name)


def validate_skills(skills):
    seen = set()
    for skill in skills:
        check_required(skill, "name")
        check_required(skill, "command")
        check_required(skill, "description")

        if skill.name in seen:
            _raise_semantic(f"Duplicate skill name: {skill.name}", skill)
        seen.add(skill.name)

        validate_skill_arguments(skill)


def validate_skill_arguments(skill):
    seen = set()

    for arg in skill.skillArguments:
        check_required(arg, "name")
        check_required(arg, "description")

        if arg.name in seen:
            _raise_semantic(
                f"Duplicate SkillArgument name '{arg.name}' in skill '{skill.name}'",
                arg,
            )
        seen.add(arg.name)


def check_required(obj, field):
    value = getattr(obj, field, None)

    if value is None:
        _raise_semantic(f"{obj.__class__.__name__}.{field} is required", obj)

    if isinstance(value, str) and value.strip() == "":
        _raise_semantic(f"{obj.__class__.__name__}.{field} is required", obj)


def _raise_semantic(message, model_obj):
    location_obj = _get_location_obj(model_obj)
    if location_obj is None:
        raise TextXSemanticError(message)

    location = get_location(location_obj)
    raise TextXSemanticError(
        message,
        line=location["line"],
        col=location["col"],
        nchar=location["nchar"],
        filename=location["filename"],
    )


def _get_location_obj(model_obj):
    if hasattr(model_obj, "_tx_position"):
        return model_obj

    source_obj = getattr(model_obj, "_source_obj", None)
    if source_obj is not None and hasattr(source_obj, "_tx_position"):
        return source_obj

    return None