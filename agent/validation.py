from textx import TextXSemanticError, get_location

ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


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
    if system.chatAgent is not None:
        executor_names = {e.task.name for e in system.executors if e.task}
        validate_chat_agent(system.chatAgent, executor_names)


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
    check_required(planner, "llm")
    check_required(planner, "persona")
    validate_reasoning_effort(planner, "reasoningStrategy")


def validate_executor(executor):
    check_required(executor, "llm")
    check_required(executor, "persona")
    validate_reasoning_effort(executor, "reasoningStrategy")

    if executor.task:
        validate_task(executor.task)


def validate_task(task):
    check_required(task, "name")
    check_required(task, "inputDescription")
    check_required(task, "behavior")

    if task.outputSpec is None:
        from agent.metamodel import OutputSpec

        task.outputSpec = OutputSpec(format="string", fields=[])

    for example in task.examples:
        validate_task_example(example, task)


def validate_task_example(example, task):
    check_required(example, "input")
    check_required(example, "output")
    known_skills = {skill.name: skill for skill in task.skills}
    for command in getattr(example, "commands", []) or []:
        check_required(command, "toolName")
        if command.toolName not in known_skills:
            _raise_semantic(
                f"Unknown example command tool: '{command.toolName}'",
                command,
            )
        validate_example_command_arguments(command, known_skills[command.toolName])


def validate_example_command_arguments(command, skill):
    seen = set()
    known_args = {arg.name for arg in skill.skillArguments}
    for argument in command.arguments or []:
        check_required(argument, "name")
        check_required(argument, "value")
        if argument.name in seen:
            _raise_semantic(
                f"Duplicate example command argument '{argument.name}' for tool '{command.toolName}'",
                argument,
            )
        seen.add(argument.name)
        if argument.name not in known_args:
            _raise_semantic(
                f"Unknown example command argument '{argument.name}' for tool '{command.toolName}'",
                argument,
            )
    missing = known_args - seen
    if missing:
        _raise_semantic(
            f"Missing example command arguments for tool '{command.toolName}': {', '.join(sorted(missing))}",
            command,
        )


def validate_chat_agent(chat_agent, executor_names: set):
    check_required(chat_agent, "name")
    check_required(chat_agent, "persona")
    check_required(chat_agent, "goal")
    validate_reasoning_effort(chat_agent, "reasoningStrategy")
    if not chat_agent.questions:
        _raise_semantic(
            f"ChatModeAgent '{chat_agent.name}' must define at least one question",
            chat_agent,
        )
    if chat_agent.executor_ref is not None and chat_agent.executor_ref not in executor_names:
        _raise_semantic(
            f"ChatModeAgent '{chat_agent.name}' references unknown executor: '{chat_agent.executor_ref}'",
            chat_agent,
        )


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


def validate_reasoning_effort(obj, field):
    value = getattr(obj, field, None)
    if value is None:
        return
    if value == "":
        setattr(obj, field, None)
        return
    if isinstance(value, str) and len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
        setattr(obj, field, value)
    if value not in ALLOWED_REASONING_EFFORTS:
        _raise_semantic(
            f"{obj.__class__.__name__}.{field} must be one of: " f"{', '.join(sorted(ALLOWED_REASONING_EFFORTS))}",
            obj,
        )


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
