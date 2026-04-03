from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

from agent.codegen import generate_agent_module
from agent.ir import build_prompt_ir
from agent.parser import parse_model
from agent.runtime import ExampleDrivenModel


def main(argv=None):
    argv = list(argv) if argv is not None else None
    if argv is None:
        import sys

        argv = sys.argv[1:]

    if _looks_like_legacy_invocation(argv):
        return _run_inspect(_build_legacy_parser().parse_args(argv))

    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser():
    parser = argparse.ArgumentParser(description="Inspect, generate, or run an agent defined in a .agent file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Parse and inspect a .agent model",
    )
    _add_model_argument(inspect_parser)
    inspect_parser.add_argument(
        "--print-model-info",
        action="store_true",
        help="Print a small summary of the parsed model",
    )
    inspect_parser.add_argument(
        "--print-ir",
        action="store_true",
        help="Print the generated Prompt IR as JSON",
    )
    inspect_parser.set_defaults(func=_run_inspect)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a runnable Python module from a .agent model",
    )
    _add_model_argument(generate_parser)
    generate_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the generated Python module",
    )
    generate_parser.set_defaults(func=_run_generate)

    run_parser = subparsers.add_parser(
        "run",
        help="Generate and run a Python agent module from a .agent model",
    )
    _add_model_argument(run_parser)
    run_parser.add_argument(
        "prompt",
        help="User input to send to the generated agent",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full run result as JSON",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print executor selection, each tool call, and the raw LLM response",
    )
    run_parser.add_argument(
        "--dspy",
        action="store_true",
        help="Use a DSPy few-shot client built from the task examples",
    )
    run_parser.add_argument(
        "--use-examples",
        action="store_true",
        help="Use the deterministic example-driven local model instead of the provider client",
    )
    run_parser.set_defaults(func=_run_generated_agent)

    return parser


def _build_legacy_parser():
    parser = argparse.ArgumentParser(description="Parse an .agent file and build Prompt IR.")
    _add_model_argument(parser)
    parser.add_argument(
        "--print-model-info",
        action="store_true",
        help="Print a small summary of the parsed model",
    )
    parser.add_argument(
        "--print-ir",
        action="store_true",
        help="Print the generated Prompt IR as JSON",
    )
    return parser


def _add_model_argument(parser):
    parser.add_argument("model_path", help="Path to the .agent model file")


def _looks_like_legacy_invocation(argv: list[str]) -> bool:
    return bool(argv) and argv[0] not in {"inspect", "generate", "run"}


def _run_inspect(args):
    system = parse_model(args.model_path)

    if args.print_model_info:
        print("Model parsed successfully.")
        print("Planner persona:", system.planner.persona)
        print("Executor count:", len(system.executors))

        if system.executors and system.executors[0].task:
            print("First task name:", system.executors[0].task.name)

    if args.print_ir:
        prompt_ir = build_prompt_ir(system)
        print(json.dumps(prompt_ir, indent=2, ensure_ascii=False))

    if not args.print_model_info and not args.print_ir:
        print("Model parsed and validated successfully.")


def _run_generate(args):
    output_path = generate_agent_module(args.model_path, args.output)
    print(output_path)


def _run_generated_agent(args):
    model_client = ExampleDrivenModel() if args.use_examples else None
    with tempfile.TemporaryDirectory(prefix="agent_codegen_") as temp_dir:
        module_path = Path(temp_dir) / "generated_agent.py"
        generate_agent_module(args.model_path, module_path)
        module = _load_generated_module(module_path)
        result = module.run_agent(
            args.prompt,
            model_client=model_client,
            require_provider=not args.use_examples,
            use_dspy=args.dspy,
        )

    if args.verbose:
        print(f"[executor] {result.executor_name} — {result.planner_reason}", flush=True)
        for tool in result.tool_results:
            print(f"\n[tool_call] $ {tool.command}", flush=True)
            print(f"[exit_code] {tool.exit_code}", flush=True)
            if tool.stdout.strip():
                print(f"[stdout]\n{tool.stdout.rstrip()}", flush=True)
            if tool.stderr.strip():
                print(f"[stderr]\n{tool.stderr.rstrip()}", flush=True)
        print(f"\n[raw_response]\n{result.raw_response}", flush=True)
        print()

    if args.json:
        print(
            json.dumps(
                {
                    "executor_name": result.executor_name,
                    "planner_reason": result.planner_reason,
                    "output": result.output,
                    "output_schema": result.output_schema,
                    "raw_response": result.raw_response,
                    "tool_results": [
                        {
                            "command": tool.command,
                            "stdout": tool.stdout,
                            "stderr": tool.stderr,
                            "exit_code": tool.exit_code,
                        }
                        for tool in result.tool_results
                    ],
                },
                indent=2,
            )
        )
        return

    if isinstance(result.output, dict):
        print(json.dumps(result.output, indent=2))
        return

    print(result.output)


def _load_generated_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("generated_agent_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
