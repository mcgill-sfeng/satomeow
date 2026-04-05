import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.runtime import load_openai_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "agent_composer" / "agent_composer.agent"


def _has_provider() -> bool:
    return load_openai_config(str(MODEL_PATH)) is not None


pytestmark = pytest.mark.skipif(not _has_provider(), reason="requires OPENAI provider configuration")


def _run_cli_json(prompt: str) -> dict:
    env = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.cli",
            "run",
            str(MODEL_PATH),
            "--json",
            prompt,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return json.loads(completed.stdout)


def _inspect_agent(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "agent.cli",
            "inspect",
            str(path),
            "--print-ir",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_e2e_agent_composer_creates_new_agent_file(tmp_path):
    target = tmp_path / "summarizer.agent"

    result = _run_cli_json(
        f"I need an agent that summarizes web pages. Save it to {target}"
    )

    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert 'llm: "gpt-5.4-nano"' in text
    assert "Summar" in text
    inspect_result = _inspect_agent(target)
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert '"executors"' in inspect_result.stdout
    assert result["executor_name"] == "Composer"


def test_e2e_agent_composer_routes_to_validator_for_existing_agent():
    result = _run_cli_json(
        "Validate models/data_visualizer/data_visualizer.agent"
    )

    assert result["executor_name"] == "Validator"
    assert result["output"]["valid"] is True
    assert result["output"]["errors"] == ""


def test_e2e_agent_composer_routes_to_validator_for_invalid_agent(tmp_path):
    target = tmp_path / "bad.agent"
    target.write_text("this is not valid DSL", encoding="utf-8")

    result = _run_cli_json(f"Validate {target}")

    assert result["executor_name"] == "Validator"
    assert result["output"]["valid"] is False
    assert result["output"]["errors"]
