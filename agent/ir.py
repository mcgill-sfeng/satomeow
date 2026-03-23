def build_prompt_ir(model) -> dict:
    """
    Convert textX model into a Jinja2-friendly dictionary IR.
    """

    # ---- Global skills ----
    global_skills = {
        skill.name: {
            "name": skill.name,
            "command": skill.command,
            "description": skill.description,
            "arguments": [
                {
                    "name": arg.name,
                    "description": arg.description
                }
                for arg in skill.skillArguments
            ]
        }
        for skill in model.skills
    }

    # ---- Planner ----
    planner = {
        "reasoning_strategy": model.system.planner.reasoningStrategy,
        "llm": model.system.planner.llm,
        "persona": model.system.planner.persona,
        "rules": [rule.description for rule in model.system.planner.rules],
    }

    # ---- Executors ----
    executors = []

    for executor in model.system.executors:
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
                "skills": [
                    global_skills[skill.name]
                    for skill in executor.task.skills
                ],
            }

        executors.append({
            "reasoning_strategy": executor.reasoningStrategy,
            "llm": executor.llm,
            "persona": executor.persona,
            "rules": [rule.description for rule in executor.rules],
            "task": task_ir,
        })

    # ---- System-level ----
    system_ir = {
        "planner": planner,
        "executors": executors,
        "rules": [rule.description for rule in model.system.rules],
        "skills": [
            global_skills[skill.name]
            for skill in model.system.skills
        ],
    }

    return system_ir