def build_prompt_ir(system) -> dict:
    """
    Convert textX model into a Jinja2-friendly dictionary IR.
    """

    # ---- Global skills ----
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

    # ---- Planner ----
    planner = {
        "reasoning_strategy": system.planner.reasoningStrategy,
        "llm": system.planner.llm,
        "persona": system.planner.persona,
        "rules": [global_rules[rule.name] for rule in system.planner.rules],
    }

    # ---- Executors ----
    executors = []

    for executor in system.executors:
        task_ir = None

        if executor.task is not None:
            task_ir = {
                "name": executor.task.name,
                "input_description": executor.task.inputDescription,
                "behavior": executor.task.behavior,
                "output_schema": executor.task.outputSchema,
                "examples": [
                    {
                        "input": ex.input,
                        "commands": list(ex.commands),
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

    # ---- System-level ----
    system_ir = {
        "planner": planner,
        "executors": executors,
        "rules": [global_rules[rule.name] for rule in system.rules],
        "skills": [global_skills[skill.name] for skill in system.skills],
    }

    return system_ir
