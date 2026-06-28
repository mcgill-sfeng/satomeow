from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from agent.codegen import generate_agent_module, generate_portable_agent_bundle
from agent.ir import serialize_system_to_dict
from agent.parser import parse_model


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

    portable_parser = subparsers.add_parser(
        "portable",
        help="Generate an experimental portable bundle directory with an agent.sh entrypoint",
    )
    _add_model_argument(portable_parser)
    portable_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for the portable bundle",
    )
    portable_parser.set_defaults(func=_run_portable)

    compile_parser = subparsers.add_parser(
        "compile",
        help="Run DSPy BootstrapFewShot on a .agent model and save optimised examples to a sidecar",
    )
    _add_model_argument(compile_parser)
    compile_parser.add_argument(
        "--model",
        metavar="LM",
        default=None,
        help="DSPy LM string to use (e.g. 'openai/gpt-4o'). Uses the environment default if omitted.",
    )
    compile_parser.set_defaults(func=_run_compile)

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
        help="Use the compile-time DSPy-style example prompt enrichment path",
    )
    run_parser.add_argument(
        "--call-graph",
        choices=["text", "dot"],
        metavar="FORMAT",
        dest="call_graph",
        help="Print the agent call graph after the run (text or dot)",
    )
    run_parser.set_defaults(func=_run_generated_agent)

    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Print the SDK prompt payload that would be sent for a given input",
    )
    _add_model_argument(prompt_parser)
    prompt_parser.add_argument("prompt", help="User input to compose against the model")
    prompt_parser.add_argument(
        "--executor",
        help="Executor name to target when the model has multiple executors",
    )
    prompt_parser.add_argument(
        "--planner",
        action="store_true",
        help="Dump the planner prompt instead of an executor prompt",
    )
    prompt_parser.add_argument(
        "--dspy",
        action="store_true",
        help="Use compiled DSPy examples if available when composing the prompt",
    )
    prompt_parser.set_defaults(func=_run_prompt_dump)

    chat_parser = subparsers.add_parser(
        "chat",
        help="Run an interactive chat intake agent defined by a 'chat' block in the .agent model",
    )
    _add_model_argument(chat_parser)
    chat_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print executor selection, tool calls, and the bundled input after confirmation",
    )
    chat_parser.set_defaults(func=_run_chat)

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


def _run_compile(args):
    from agent.dspy_compile import compile_system_spec
    from agent.parser import parse_model

    system = parse_model(args.model_path)

    executors_with_examples = [e.task.name for e in system.executors if e.task.examples]
    if not executors_with_examples:
        print("No executors with examples found — nothing to compile.")
        return

    lm = None
    if args.model:
        try:
            import dspy  # type: ignore[import-untyped]

            lm = dspy.LM(args.model)
        except Exception as exc:
            print(f"[error] Could not initialise DSPy LM '{args.model}': {exc}")
            return

    print(f"Compiling {len(executors_with_examples)} executor(s): {', '.join(executors_with_examples)}")

    try:
        sidecar = compile_system_spec(system, args.model_path, lm=lm)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return
    except Exception as exc:
        print(f"[error] Compilation failed: {exc}")
        return

    print(f"Sidecar written to: {sidecar}")


def _looks_like_legacy_invocation(argv: list[str]) -> bool:
    return bool(argv) and argv[0] not in {"inspect", "generate", "portable", "run", "chat", "compile", "prompt"}


def _run_inspect(args):
    system = parse_model(args.model_path)

    if args.print_model_info:
        print("Model parsed successfully.")
        print("Planner persona:", system.planner.persona)
        print("Executor count:", len(system.executors))

        if system.executors and system.executors[0].task:
            print("First task name:", system.executors[0].task.name)

    if args.print_ir:
        prompt_ir = serialize_system_to_dict(system)
        print(json.dumps(prompt_ir, indent=2, ensure_ascii=False))

    if not args.print_model_info and not args.print_ir:
        print("Model parsed and validated successfully.")


def _run_generate(args):
    output_path = generate_agent_module(args.model_path, args.output)
    print(output_path)


def _run_portable(args):
    output_path = generate_portable_agent_bundle(args.model_path, args.output)
    print(
        "[warning] portable is experimental/TODO: repo-relative assets and hard-coded skill paths are not bundled yet.",
        flush=True,
    )
    print(output_path)


def _run_prompt_dump(args):
    from agent.parser import parse_model
    from agent.runtime import AgentSystemRuntime

    system = parse_model(args.model_path)
    runtime = AgentSystemRuntime(
        system,
        source_model_path=args.model_path,
        use_dspy=args.dspy,
    )

    try:
        payload = runtime.build_prompt_dump(
            args.prompt,
            executor_name=args.executor,
            planner=args.planner,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_generated_agent(args):
    with tempfile.TemporaryDirectory(prefix="agent_codegen_") as temp_dir:
        module_path = Path(temp_dir) / "generated_agent.py"
        generate_agent_module(args.model_path, module_path)
        module = _load_generated_module(module_path)
        result = module.run_agent(
            args.prompt,
            use_dspy=args.dspy,
        )

    if args.verbose:
        print(f"[executor] {result.executor_name} — {result.planner_reason}", flush=True)
        print(f"\n[system_prompt]\n{result.system_prompt}", flush=True)
        print(f"\n[user_input]\n{result.user_input}", flush=True)
        for tool in result.tool_results:
            print(f"\n[tool_call] $ {tool.command}", flush=True)
            print(f"[exit_code] {tool.exit_code}", flush=True)
            if tool.stdout.strip():
                print(f"[stdout]\n{tool.stdout.rstrip()}", flush=True)
            if tool.stderr.strip():
                print(f"[stderr]\n{tool.stderr.rstrip()}", flush=True)
        for index, raw_response in enumerate(result.raw_responses or (result.raw_response,), start=1):
            print(f"\n[raw_response {index}]\n{raw_response}", flush=True)
        print()

    # When DOT output is requested, the graph must be the only thing on stdout
    # so the result can be piped directly to `dot`. Everything else goes to stderr.
    dot_mode = getattr(args, "call_graph", None) == "dot"
    out = sys.stderr if dot_mode else sys.stdout

    if args.json:
        print(
            json.dumps(
                {
                    "executor_name": result.executor_name,
                    "planner_reason": result.planner_reason,
                    "output": result.output,
                    "output_format": result.output_format,
                    "raw_response": result.raw_response,
                    "system_prompt": result.system_prompt,
                    "user_input": result.user_input,
                    "raw_responses": list(result.raw_responses),
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
            ),
            file=out,
        )
        if not dot_mode:
            return

    elif isinstance(result.output, dict):
        print(json.dumps(result.output, indent=2), file=out)
    else:
        print(result.output, file=out)

    if getattr(args, "call_graph", None):
        if dot_mode:
            print(result.call_graph.render_dot(), flush=True)
        else:
            print(flush=True)
            print(result.call_graph.render_text(), flush=True)


def _run_chat(args):
    from agent.parser import parse_model
    from agent.runtime import AgentSystemRuntime

    system = parse_model(args.model_path)
    chat_agent = system.chatAgent

    if chat_agent is None:
        print(
            "Error: No 'chat' agent defined in this model.\n"
            "Add a 'chat' block to use interactive chat mode. See DSL_REFERENCE.md.",
            flush=True,
        )
        return

    runtime = AgentSystemRuntime(
        system,
        source_model_path=args.model_path,
    )

    questions = chat_agent.questions
    goal = chat_agent.goal
    executor_ref = chat_agent.executor_ref

    _print_logo_v2()

    # Banner
    print("─" * 60, flush=True)
    print(f"  {chat_agent.name} ({chat_agent.persona})", flush=True)
    print(f"  Goal: {goal}", flush=True)
    print("─" * 60, flush=True)
    print("  Type 'quit' or Ctrl-D to exit at any time.", flush=True)
    print("─" * 60, flush=True)
    print(flush=True)

    # Q&A loop — runtime-driven, one question at a time
    answers: list[str] = []
    for i, question in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] {question}", flush=True)
        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return
        if answer.lower() in {"quit", "exit"}:
            return
        answers.append(answer)

    print(flush=True)
    print("Thinking...", flush=True)

    # LLM-generated confirmation summary
    try:
        confirmation = runtime.generate_confirmation(goal, questions, answers)
    except Exception as exc:
        print(f"[error] Failed to generate confirmation: {exc}", flush=True)
        return

    print(flush=True)
    print(confirmation, flush=True)
    print(flush=True)

    try:
        proceed = input("Proceed? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(flush=True)
        return

    if proceed in {"n", "no"}:
        print("Cancelled.", flush=True)
        return

    # Build bundled one-shot input and execute
    bundled_input = AgentSystemRuntime._build_chat_input(goal, questions, answers)

    if args.verbose:
        print(f"\n[bundled input]\n{bundled_input}", flush=True)

    try:
        result = runtime.run(bundled_input, executor_name=executor_ref)
    except Exception as exc:
        print(f"[error] {exc}", flush=True)
        return

    if args.verbose:
        print(f"[executor] {result.executor_name} — {result.planner_reason}", flush=True)
        for tool in result.tool_results:
            print(f"[tool_call] $ {tool.command}", flush=True)
            print(f"[exit_code] {tool.exit_code}", flush=True)
            if tool.stdout.strip():
                print(f"[stdout]\n{tool.stdout.rstrip()}", flush=True)
            if tool.stderr.strip():
                print(f"[stderr]\n{tool.stderr.rstrip()}", flush=True)
        print(flush=True)

    if isinstance(result.output, dict):
        print(json.dumps(result.output, indent=2), flush=True)
    else:
        print(result.output, flush=True)


def _print_logo_v2():
    logo_path = Path(__file__).resolve().parent.parent / "logo-v2.txt"
    if not logo_path.exists():
        return
    print(logo_path.read_text(encoding="utf-8").rstrip(), flush=True)
    print(flush=True)


def _load_generated_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("generated_agent_module", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
