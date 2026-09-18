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


def write_compact(path, data):
    """Write one of the fetched files: no spaces, real UTF-8, stable order."""
    payload = json.dumps(data, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(payload)
    return payload


def write_shard_files(directory, files):
    """Write one compact file per entry, creating the directory; return the bytes.

    ``files`` maps the whole file name, suffix and all, to what goes in it, so
    one caller can shard by id and another by ISCO unit group.
    """
    os.makedirs(directory, exist_ok=True)
    return [write_compact(os.path.join(directory, name), data)
            for name, data in sorted(files.items())]
