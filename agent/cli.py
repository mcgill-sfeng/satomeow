import argparse
import json

from agent.parser import parse_model
from agent.ir import build_prompt_ir


def main():
    parser = argparse.ArgumentParser(
        description="Parse an .agent file and build Prompt IR."
    )
    parser.add_argument(
        "model_path",
        help="Path to the .agent model file"
    )
    parser.add_argument(
        "--print-model-info",
        action="store_true",
        help="Print a small summary of the parsed model"
    )
    parser.add_argument(
        "--print-ir",
        action="store_true",
        help="Print the generated Prompt IR as JSON"
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()