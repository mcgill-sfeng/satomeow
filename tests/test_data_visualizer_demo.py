import importlib.util
import shutil
from pathlib import Path

import pytest

from agent.codegen import generate_agent_module
from agent.runtime import ExampleDrivenModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = PROJECT_ROOT / "models" / "data_visualizer"
MODEL_PATH = DEMO_ROOT / "data_visualizer.agent"


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


def test_aligned_data_demo_branch(tmp_path):
    module = _load_generated_demo_module(tmp_path)
    result = module.run_agent(
        "Visualize the aligned monthly sales data in models/data_visualizer/data/aligned_sales.json",
        model_client=ExampleDrivenModel(),
    )

    assert result.output["status"] == "success"
    assert result.output["artifact_path"] == "models/data_visualizer/output/aligned_chart.svg"
    assert result.output["preprocessed_data_path"] == ""
    assert result.output["preprocessor_script_path"] == ""
    assert (DEMO_ROOT / "output" / "aligned_chart.svg").exists()


def test_convertible_data_demo_branch(tmp_path):
    module = _load_generated_demo_module(tmp_path)
    result = module.run_agent(
        "Visualize the monthly revenue CSV in models/data_visualizer/data/monthly_revenue.csv",
        model_client=ExampleDrivenModel(),
    )

    assert result.output["status"] == "success"
    assert result.output["artifact_path"] == "models/data_visualizer/output/converted_chart.svg"
    assert result.output["preprocessed_data_path"] == "models/data_visualizer/generated/monthly_revenue_converted.json"
    assert result.output["preprocessor_script_path"] == "models/data_visualizer/generated/monthly_csv_to_json.py"
    assert (DEMO_ROOT / "generated" / "monthly_csv_to_json.py").exists()
    assert (DEMO_ROOT / "generated" / "monthly_revenue_converted.json").exists()
    assert (DEMO_ROOT / "output" / "converted_chart.svg").exists()


def test_unsupported_data_demo_branch(tmp_path):
    module = _load_generated_demo_module(tmp_path)
    result = module.run_agent(
        "Visualize the unsupported dataset in models/data_visualizer/data/unsupported_nested.json",
        model_client=ExampleDrivenModel(),
    )

    assert result.output["status"] == "unsupported_input"
    assert "cannot be converted" in result.output["message"]
    assert result.output["artifact_path"] == ""
    assert result.output["preprocessed_data_path"] == ""
    assert result.output["preprocessor_script_path"] == ""
    assert not (DEMO_ROOT / "generated").exists()
    assert not (DEMO_ROOT / "output").exists()


def _load_generated_demo_module(tmp_path: Path):
    module_path = tmp_path / "generated_data_visualizer.py"
    generate_agent_module(MODEL_PATH, module_path)
    spec = importlib.util.spec_from_file_location("generated_data_visualizer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
