import json
import os
from pathlib import Path

from agent.parser import parse_model
from agent.runtime import (
    AgentSystemRuntime,
    ShellToolExecutor,
    build_executor_system_prompt,
    build_function_tool,
    build_output_type,
    build_planner_prompt,
)

SNAPSHOT_ROOT = Path(__file__).resolve().parent / "snapshots" / "runtime_characterization"
MODEL_PATHS = {
    "example_full": "models/example_full.agent",
    "example_minimal": "models/example_minimal.agent",
    "data_visualizer": "models/data_visualizer/data_visualizer.agent",
    "example_chat": "models/example_chat.agent",
}
SAMPLE_INPUT = "snapshot input"


def _tool_schema_payload(skill):
    tool = build_function_tool(skill, tool_executor=ShellToolExecutor())
    return {
        "name": tool.name,
        "description": tool.description,
        "params_json_schema": tool.params_json_schema,
    }


def _capture_model_snapshot(model_path: str) -> dict:
    system = parse_model(model_path)
    runtime = AgentSystemRuntime(system)

    executors_payload = []
    for executor in system.executors:
        output_type = build_output_type(executor.task.outputSpec, executor.task.name)
        executors_payload.append(
            {
                "name": executor.task.name,
                "system_prompt": build_executor_system_prompt(executor),
                "tools": [_tool_schema_payload(skill) for skill in executor.task.skills],
                "output_type_json_schema": output_type.model_json_schema() if output_type is not None else None,
                "prompt_dump": runtime.build_prompt_dump(SAMPLE_INPUT, executor_name=executor.task.name),
            }
        )

    planner_prompt = None
    planner_dump = None
    if len(system.executors) > 1:
        planner_prompt = build_planner_prompt(system.executors)
        planner_dump = runtime.build_prompt_dump(SAMPLE_INPUT, planner=True)

    return {
        "model_path": model_path,
        "executors": executors_payload,
        "planner_prompt": planner_prompt,
        "planner_prompt_dump": planner_dump,
    }


def test_runtime_characterization_snapshots():
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    update = os.environ.get("UPDATE_SNAPSHOTS") == "1"

    for model_name, model_path in MODEL_PATHS.items():
        actual = _capture_model_snapshot(model_path)
        snapshot_path = SNAPSHOT_ROOT / f"{model_name}.json"

        if update or not snapshot_path.exists():
            snapshot_path.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")

        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert actual == expected
