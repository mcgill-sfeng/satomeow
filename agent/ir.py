def serialize_system_to_dict(system) -> dict:
    """Serialize a System into a flat, JSON-friendly dictionary for the
    ``inspect --print-ir`` debug command."""

    global_skills = {
        skill.name: {
            "name": skill.name,
            "command": skill.command,
            "description": skill.description,
            "arguments": [{"name": arg.name, "description": arg.description} for arg in skill.skillArguments],
        }
        for skill in system.skills
    }

    global_rules = {
        rule.name: {
            "name": rule.name,
            "negative": rule.negative,
            "description": rule.description,
        }
        for rule in system.rules
    }

    planner = {
        "reasoning_strategy": system.planner.reasoningStrategy,
        "llm": system.planner.llm,
        "persona": system.planner.persona,
        "rules": [global_rules[rule.name] for rule in system.planner.rules],
    }

    executors = []

    for executor in system.executors:
        task_ir = None

        if executor.task is not None:
            spec = executor.task.outputSpec
            task_ir = {
                "name": executor.task.name,
                "input_description": executor.task.inputDescription,
                "behavior": executor.task.behavior,
                "output_format": spec.format if spec else "string",
                "output_fields": [{"name": f.name, "type": f.type} for f in (spec.fields if spec else [])],
                "examples": [
                    {
                        "input": ex.input,
                        "commands": [
                            {
                                "tool_name": command.toolName,
                                "arguments": [
                                    {"name": argument.name, "value": argument.value} for argument in command.arguments
                                ],
                            }
                            for command in ex.commands
                        ],
                        "output": ex.output,
                    }
                    for ex in executor.task.examples
                ],
                "skills": [global_skills[skill.name] for skill in executor.task.skills],
            }

        executors.append(
            {
                "name": executor.task.name if executor.task is not None else executor.persona,
                "reasoning_strategy": executor.reasoningStrategy,
                "llm": executor.llm,
                "persona": executor.persona,
                "rules": [global_rules[rule.name] for rule in executor.rules],
                "task": task_ir,
            }
        )

    chat_agent_ir = None
    if system.chatAgent is not None:
        ca = system.chatAgent
        chat_agent_ir = {
            "name": ca.name,
            "persona": ca.persona,
            "llm": ca.llm,
            "reasoning_strategy": ca.reasoningStrategy,
            "goal": ca.goal,
            "questions": list(ca.questions),
            "executor_ref": ca.executor_ref,
        }

    system_ir = {
        "planner": planner,
        "executors": executors,
        "rules": [global_rules[rule.name] for rule in system.rules],
        "skills": [global_skills[skill.name] for skill in system.skills],
        "chatAgent": chat_agent_ir,
    }

    return system_ir
