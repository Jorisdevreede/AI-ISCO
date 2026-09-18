"""Resuming a long scoring run from the file it writes as it goes.

The three scoring scripts each spend hours calling a model, so they write their
results after every batch and skip on restart whatever is already there. The
writer here deliberately keeps json.dump's default ASCII escaping: the pipeline's
committed files are escaped that way, and switching to aiisco.jsonio.write_json
would rewrite every non-ASCII title.
"""

import json
import os

from aiisco.jsonio import load_json


def index_by_uri(entries):
    """Index entries by their uri, later entries winning."""
    return {entry["uri"]: entry for entry in entries}


def load_checkpoint(path, force=False):
    """Return the entries already written to path, keyed by uri.

    Empty when the file does not exist yet, or when force asks for a fresh run.
    """
    if force or not os.path.exists(path):
        return {}
    return index_by_uri(load_json(path))


def pending(items, done):
    """The items whose uri is not in done, in their original order."""
    return [item for item in items if item["uri"] not in done]


def save_checkpoint(path, entries, indent=2):
    """Write the collected entries as a JSON array, escaped as the pipeline is."""
    with open(path, "w") as f:
        json.dump(list(entries.values()), f, indent=indent)


def save_checkpoint_atomically(path, entries, indent=2):
    """Write through a temporary file so a killed run cannot truncate the file."""
    tmp = path + ".tmp"
    save_checkpoint(tmp, entries, indent)
    os.replace(tmp, path)
