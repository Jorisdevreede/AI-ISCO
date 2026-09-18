"""Merge narrative shard files into a single occupation_narratives.json."""
import glob
import json

from aiisco.checkpoint import index_by_uri, save_checkpoint

SHARD_GLOB = "data/occupation_narratives_shard_*.json"
OUTPUT_FILE = "data/occupation_narratives.json"


def load_existing(path):
    """Load the narratives merged by an earlier run, empty when there are none."""
    try:
        with open(path) as f:
            merged = index_by_uri(json.load(f))
    except FileNotFoundError:
        print(f"No existing {path}")
        return {}
    print(f"Loaded {len(merged)} from {path}")
    return merged


def merge_shard(path, merged):
    """Add one shard's narratives to merged, reporting how many were new."""
    try:
        with open(path) as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {path}: {e}")
        return
    count = 0
    for entry in entries:
        if entry["uri"] not in merged:
            count += 1
        merged[entry["uri"]] = entry
    print(f"Loaded {len(entries)} from {path} ({count} new)")


def main():
    merged = load_existing(OUTPUT_FILE)
    for path in sorted(glob.glob(SHARD_GLOB)):
        merge_shard(path, merged)
    save_checkpoint(OUTPUT_FILE, merged)
    print(f"\nMerged total: {len(merged)} narratives -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
