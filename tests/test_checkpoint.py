"""Behaviour of the resume helpers: indexing, resuming and byte-stable writes."""

import json

import pytest

from aiisco.checkpoint import (
    index_by_uri,
    load_checkpoint,
    pending,
    save_checkpoint,
    save_checkpoint_atomically,
)

ENTRIES = [{"uri": "u/a", "title": "brew coffee"},
           {"uri": "u/b", "title": "wash dishes"}]


def test_index_by_uri_keys_entries_and_lets_the_last_one_win():
    indexed = index_by_uri([*ENTRIES, {"uri": "u/a", "title": "newer"}])
    assert list(indexed) == ["u/a", "u/b"]
    assert indexed["u/a"]["title"] == "newer"


def test_load_checkpoint_is_empty_when_there_is_no_file(tmp_path):
    assert load_checkpoint(str(tmp_path / "missing.json")) == {}


def test_load_checkpoint_reads_the_entries_of_an_earlier_run(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(ENTRIES))
    assert list(load_checkpoint(str(path))) == ["u/a", "u/b"]


def test_load_checkpoint_ignores_the_file_when_force_is_set(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(ENTRIES))
    assert load_checkpoint(str(path), force=True) == {}


def test_pending_keeps_the_input_order_and_drops_finished_items():
    assert pending(ENTRIES, {"u/a": {}}) == [ENTRIES[1]]
    assert pending(ENTRIES, {}) == ENTRIES


def test_save_checkpoint_escapes_non_ascii_as_the_pipeline_files_do(tmp_path):
    path = tmp_path / "scores.json"
    save_checkpoint(str(path), {"u/c": {"uri": "u/c", "title": "café"}})
    assert path.read_text() == (
        "[\n  {\n    \"uri\": \"u/c\",\n    \"title\": \"caf\\u00e9\"\n  }\n]"
    )


def test_save_checkpoint_honours_the_indent(tmp_path):
    path = tmp_path / "scores.json"
    save_checkpoint(str(path), {"u/a": {"uri": "u/a"}}, indent=1)
    assert path.read_text() == "[\n {\n  \"uri\": \"u/a\"\n }\n]"


def test_save_checkpoint_writes_the_values_in_insertion_order(tmp_path):
    path = tmp_path / "scores.json"
    save_checkpoint(str(path), {"u/b": ENTRIES[1], "u/a": ENTRIES[0]})
    assert [e["uri"] for e in json.loads(path.read_text())] == ["u/b", "u/a"]


def test_save_checkpoint_atomically_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "scores.json"
    save_checkpoint_atomically(str(path), {"u/a": ENTRIES[0]})
    assert json.loads(path.read_text()) == [ENTRIES[0]]
    assert not (tmp_path / "scores.json.tmp").exists()


def test_save_checkpoint_atomically_replaces_an_earlier_file(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text("[]")
    save_checkpoint_atomically(str(path), {"u/a": ENTRIES[0]}, indent=1)
    assert json.loads(path.read_text()) == [ENTRIES[0]]


def test_load_checkpoint_exits_when_the_file_is_unreadable(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text("{ not json")
    with pytest.raises(json.JSONDecodeError):
        load_checkpoint(str(path))
