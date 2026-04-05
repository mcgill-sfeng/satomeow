#!/usr/bin/env python3
"""Helper script: write text content to a file, creating parent dirs as needed."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Write text content to a file.")
    parser.add_argument("--path", required=True, help="Destination file path")
    parser.add_argument("--content", required=True, help="Text content to write")
    args = parser.parse_args()

    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args.content, encoding="utf-8")
    print(f"Wrote {len(args.content)} chars to {path}")


if __name__ == "__main__":
    main()
