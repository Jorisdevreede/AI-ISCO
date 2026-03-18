"""Merge narrative shard files into a single occupation_narratives.json."""
import json
import glob
import sys

def main():
    shard_files = sorted(glob.glob("data/occupation_narratives_shard_*.json"))

    # Also include the original file if it exists
    merged = {}

    original = "data/occupation_narratives.json"
    try:
        with open(original) as f:
            for entry in json.load(f):
                merged[entry["uri"]] = entry
        print(f"Loaded {len(merged)} from {original}")
    except FileNotFoundError:
        print(f"No existing {original}")

    for path in shard_files:
        try:
            with open(path) as f:
                entries = json.load(f)
            count = 0
            for entry in entries:
                if entry["uri"] not in merged:
                    count += 1
                merged[entry["uri"]] = entry
            print(f"Loaded {len(entries)} from {path} ({count} new)")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading {path}: {e}")

    result = list(merged.values())
    with open(original, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nMerged total: {len(result)} narratives -> {original}")

if __name__ == "__main__":
    main()
