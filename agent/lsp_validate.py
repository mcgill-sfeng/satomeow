import argparse
import json
import traceback

from textx import TextXError

from agent.parser import parse_model, parse_model_text


def _to_diagnostic(error):
    line = getattr(error, "line", None) or 1
    col = getattr(error, "col", None) or 1
    nchar = getattr(error, "nchar", None) or 1

    end_col = col + max(1, nchar)

    return {
        "message": str(error),
        "line": line,
        "col": col,
        "endLine": line,
        "endCol": end_col,
        "severity": "error",
    }


def main():
    parser = argparse.ArgumentParser(description="Validate .agent source for editor diagnostics.")
    parser.add_argument("--path", required=True, help="Document file path for source mapping.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read document content from stdin and validate in-memory text.",
    )
    args = parser.parse_args()

    try:
        if args.stdin:
            model_text = input_stream_read_all()
            parse_model_text(model_text, source_name=args.path)
        else:
            parse_model(args.path)

        print(json.dumps({"ok": True, "diagnostics": []}))
        return

    except TextXError as error:
        print(json.dumps({"ok": False, "diagnostics": [_to_diagnostic(error)]}))
        return

    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "diagnostics": [
                        {
                            "message": f"Internal validation error: {error}",
                            "line": 1,
                            "col": 1,
                            "endLine": 1,
                            "endCol": 2,
                            "severity": "error",
                        }
                    ],
                    "debug": traceback.format_exc(),
                }
            )
        )
        return


def input_stream_read_all():
    try:
        import sys

        return sys.stdin.read()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
