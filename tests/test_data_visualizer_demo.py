import asyncio
import os
import shutil
from pathlib import Path

import pytest

from agent.parser import parse_model
from agent.runtime import ShellToolExecutor, build_openai_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = PROJECT_ROOT / "models" / "data_visualizer"
MODEL_PATH = DEMO_ROOT / "data_visualizer.agent"
PYTHON_BIN = "models/data_visualizer/.venv/Scripts/python.exe" if os.name == "nt" else "models/data_visualizer/.venv/bin/python"


@pytest.fixture(scope="module", autouse=True)
def cleanup_demo_venv():
    shutil.rmtree(DEMO_ROOT / ".venv", ignore_errors=True)
    yield
    shutil.rmtree(DEMO_ROOT / ".venv", ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_demo_outputs():
    shutil.rmtree(DEMO_ROOT / "generated", ignore_errors=True)
    shutil.rmtree(DEMO_ROOT / "output", ignore_errors=True)
    yield
    shutil.rmtree(DEMO_ROOT / "generated", ignore_errors=True)
    shutil.rmtree(DEMO_ROOT / "output", ignore_errors=True)


@pytest.mark.skipif(os.name == "nt", reason="Data visualizer skill commands assume POSIX shell semantics.")
def test_aligned_data_demo_tools():
    tools = _tool_map()
    asyncio.run(
        tools["preparePythonEnv"].on_invoke_tool(
            None,
            '{"demo_root":"models/data_visualizer"}',
        )
    )
    asyncio.run(
        tools["runPythonScript"].on_invoke_tool(
            None,
            (
                f'{{"python_bin":"{PYTHON_BIN}",'
                '"script_path":"models/data_visualizer/scripts/visualize_data.py",'
                '"input_path":"models/data_visualizer/data/aligned_sales.json",'
                '"output_path":"models/data_visualizer/output/aligned_chart.svg"}'
            ),
        )
    )

    expected_python = DEMO_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert expected_python.exists()
    assert (DEMO_ROOT / "output" / "aligned_chart.svg").exists()


@pytest.mark.skipif(os.name == "nt", reason="Data visualizer skill commands assume POSIX shell semantics.")
def test_convertible_data_demo_tools():
    tools = _tool_map()
    asyncio.run(tools["preparePythonEnv"].on_invoke_tool(None, '{"demo_root":"models/data_visualizer"}'))
    asyncio.run(
        tools["writePreprocessor"].on_invoke_tool(
            None,
            (
                f'{{"python_bin":"{PYTHON_BIN}",'
                '"writer_script_path":"models/data_visualizer/scripts/write_preprocessor.py",'
                '"template_name":"monthly_csv_to_json",'
                '"output_script_path":"models/data_visualizer/generated/monthly_csv_to_json.py"}'
            ),
        )
    )
    asyncio.run(
        tools["runPythonScript"].on_invoke_tool(
            None,
            (
                f'{{"python_bin":"{PYTHON_BIN}",'
                '"script_path":"models/data_visualizer/generated/monthly_csv_to_json.py",'
                '"input_path":"models/data_visualizer/data/monthly_revenue.csv",'
                '"output_path":"models/data_visualizer/generated/monthly_revenue_converted.json"}'
            ),
        )
    )
    asyncio.run(
        tools["runPythonScript"].on_invoke_tool(
            None,
            (
                f'{{"python_bin":"{PYTHON_BIN}",'
                '"script_path":"models/data_visualizer/scripts/visualize_data.py",'
                '"input_path":"models/data_visualizer/generated/monthly_revenue_converted.json",'
                '"output_path":"models/data_visualizer/output/converted_chart.svg"}'
            ),
        )
    )

    assert (DEMO_ROOT / "generated" / "monthly_csv_to_json.py").exists()
    assert (DEMO_ROOT / "generated" / "monthly_revenue_converted.json").exists()
    assert (DEMO_ROOT / "output" / "converted_chart.svg").exists()


def test_demo_prompt_includes_unsupported_example():
    system = parse_model(MODEL_PATH)
    executor = system.executors[0]
    prompt = build_openai_agent(executor, tool_executor=ShellToolExecutor(), use_dspy=False).instructions
    assert "unsupported_input" in prompt
    assert "cannot be converted" in prompt


def _tool_map():
    system = parse_model(MODEL_PATH)
    executor = system.executors[0]
    agent = build_openai_agent(executor, tool_executor=ShellToolExecutor(), use_dspy=False)
    return {tool.name: tool for tool in agent.tools}
