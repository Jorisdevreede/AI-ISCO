"""Reading and writing the pipeline's JSON files."""

import json
import os
import sys


def load_json(path):
    """Load a JSON file, exiting with a clear message when it is missing."""
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data, indent=2):
    """Write JSON as UTF-8 without escaping non-ASCII characters."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
