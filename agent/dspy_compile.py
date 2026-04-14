"""DSPy compile-time integration for the Agent DSL.

Runs DSPy BootstrapFewShot on each executor that declares examples, then saves
the optimised demonstrations to a sidecar JSON file next to the source model.

Usage (CLI):
    python -m agent.cli compile models/my_agent.agent

The sidecar is written to:
    models/my_agent.agent.compiled.json

At inference time, ``AgentSystemRuntime`` loads the sidecar automatically when
it is present and ``use_dspy=True``.  The compiled demonstrations replace the
hand-written examples in the executor system prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_system_spec(
    system_spec: dict[str, Any],
    source_path: str | Path,
    *,
    lm=None,
) -> Path:
    """Run DSPy BootstrapFewShot on executors that have examples.

    Args:
        system_spec: Prompt IR dict (output of ``build_prompt_ir``).
        source_path: Path to the source ``.agent`` file.  The sidecar is
                     written alongside it as ``<path>.compiled.json``.
        lm: Optional DSPy LM instance.  If ``None``, the currently configured
            DSPy default LM is used.  Must be set before calling if no default
            is configured.

    Returns:
        Path to the written sidecar file.

    Raises:
        RuntimeError: If ``dspy`` is not installed.
    """
    dspy = _require_dspy()

    if lm is not None:
        dspy.configure(lm=lm)

    compiled: dict[str, Any] = {"executors": {}}

    for executor in system_spec["executors"]:
        examples = executor["task"].get("examples", [])
        if not examples:
            continue
        name = executor["name"]
        compiled["executors"][name] = _compile_executor(executor, examples, dspy)

    sidecar = _sidecar_path(source_path)
    sidecar.write_text(json.dumps(compiled, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def load_compiled_sidecar(source_path: str | Path) -> dict[str, Any] | None:
    """Load the compiled sidecar for a source ``.agent`` file.

    Returns the parsed dict, or ``None`` if the sidecar does not exist or
    cannot be read.
    """
    sidecar = _sidecar_path(source_path)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return None


def get_compiled_examples(
    sidecar: dict[str, Any] | None,
    executor_name: str,
) -> list[dict[str, Any]] | None:
    """Extract compiled examples for a named executor from a loaded sidecar.

    Returns ``None`` if the sidecar is absent or has no entry for the executor.
    """
    if sidecar is None:
        return None
    entry = sidecar.get("executors", {}).get(executor_name)
    if entry is None:
        return None
    return entry.get("compiled_examples") or None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sidecar_path(source_path: str | Path) -> Path:
    p = Path(source_path)
    return p.with_suffix(p.suffix + ".compiled.json")


def _require_dspy():
    try:
        import dspy

        return dspy
    except ImportError as exc:
        raise RuntimeError("dspy is not installed. Run: pip install dspy") from exc


def _compile_executor(
    executor: dict[str, Any],
    examples: list[dict[str, Any]],
    dspy,
) -> dict[str, Any]:
    """Compile a single executor's examples with DSPy BootstrapFewShot.

    Falls back to the original examples verbatim if compilation fails (e.g.
    no LM configured, metric always false, or API error).
    """
    input_desc = executor["task"].get("input_description", "user input")
    behavior = executor["task"].get("behavior", "")

    try:
        return _run_bootstrap(input_desc, behavior, examples, dspy)
    except Exception as exc:
        # Graceful fallback — original examples preserved as-is.
        return {
            "compiled_examples": [{"input": ex["input"], "output": ex["output"]} for ex in examples],
            "bootstrap_error": str(exc),
        }


def _run_bootstrap(
    input_desc: str,
    behavior: str,
    examples: list[dict[str, Any]],
    dspy,
) -> dict[str, Any]:
    """Inner function that actually runs DSPy BootstrapFewShot."""

    # Build a two-field string signature: task_input -> task_output
    class _TaskSig(dspy.Signature):
        task_input: str = dspy.InputField(desc=input_desc)
        task_output: str = dspy.OutputField(desc="expected output given the behavior")

    _TaskSig.__doc__ = behavior or "Perform the task."

    module = dspy.Predict(_TaskSig)

    trainset = [
        dspy.Example(
            task_input=ex["input"],
            task_output=ex["output"],
        ).with_inputs("task_input")
        for ex in examples
    ]

    def _metric(example, pred, trace=None):
        # Accept any non-empty prediction as a positive signal.
        return bool(getattr(pred, "task_output", "").strip())

    optimizer = dspy.BootstrapFewShot(
        metric=_metric,
        max_bootstrapped_demos=len(examples),
        max_labeled_demos=len(examples),
    )
    compiled_module = optimizer.compile(module, trainset=trainset)

    # Extract bootstrapped demonstrations from the compiled predictor.
    demos: list[dict[str, Any]] = []
    for _, predictor in compiled_module.named_predictors():
        for demo in getattr(predictor, "demos", []):
            inp = demo.get("task_input") or demo.get("input", "")
            out = demo.get("task_output") or demo.get("output", "")
            if inp or out:
                demos.append({"input": inp, "output": out})

    # If bootstrap produced nothing, fall back to the raw examples.
    if not demos:
        demos = [{"input": ex["input"], "output": ex["output"]} for ex in examples]

    return {"compiled_examples": demos}
