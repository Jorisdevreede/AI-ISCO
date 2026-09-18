"""Behaviour of the shard merger: shard order, new counts, unreadable files."""

import json
import runpy

import pytest

import merge_narrative_shards


def entry(uri, story="a"):
    """A narrative record as the generator writes it."""
    return {"uri": uri, "title": uri, "evolution_story": story}


@pytest.fixture
def data(tmp_path, monkeypatch):
    """An empty data/ directory as the working directory."""
    directory = tmp_path / "data"
    directory.mkdir()
    monkeypatch.chdir(tmp_path)
    return directory


def write(path, entries):
    """Write a narrative file the way the pipeline writes them."""
    path.write_text(json.dumps(entries, indent=2))


def test_merging_nothing_reports_no_existing_file(data, capsys):
    merge_narrative_shards.main()

    out = capsys.readouterr().out
    assert "No existing data/occupation_narratives.json" in out
    assert "\nMerged total: 0 narratives -> data/occupation_narratives.json" in out
    assert (data / "occupation_narratives.json").read_text() == "[]"


def test_shards_are_merged_in_sorted_order_after_the_existing_file(data, capsys):
    write(data / "occupation_narratives.json", [entry("occ/1", "old")])
    write(data / "occupation_narratives_shard_2.json", [entry("occ/3")])
    write(data / "occupation_narratives_shard_1.json",
          [entry("occ/1", "new"), entry("occ/2")])

    merge_narrative_shards.main()

    merged = json.loads((data / "occupation_narratives.json").read_text())
    assert [item["uri"] for item in merged] == ["occ/1", "occ/2", "occ/3"]
    assert merged[0]["evolution_story"] == "new"
    out = capsys.readouterr().out
    assert "Loaded 1 from data/occupation_narratives.json" in out
    assert "Loaded 2 from data/occupation_narratives_shard_1.json (1 new)" in out
    assert "Loaded 1 from data/occupation_narratives_shard_2.json (1 new)" in out
    assert "Merged total: 3 narratives" in out


def test_a_duplicated_uri_inside_one_shard_counts_once(data, capsys):
    write(data / "occupation_narratives_shard_1.json",
          [entry("occ/1", "first"), entry("occ/1", "second")])

    merge_narrative_shards.main()

    assert "Loaded 2 from data/occupation_narratives_shard_1.json (1 new)" in (
        capsys.readouterr().out)
    merged = json.loads((data / "occupation_narratives.json").read_text())
    assert merged == [entry("occ/1", "second")]


def test_an_unreadable_shard_is_reported_and_skipped(data, capsys):
    (data / "occupation_narratives_shard_1.json").write_text("{ not json")
    write(data / "occupation_narratives_shard_2.json", [entry("occ/2")])

    merge_narrative_shards.main()

    out = capsys.readouterr().out
    assert "Error reading data/occupation_narratives_shard_1.json:" in out
    merged = json.loads((data / "occupation_narratives.json").read_text())
    assert [item["uri"] for item in merged] == ["occ/2"]


def test_the_merged_file_keeps_the_pipeline_indentation_and_escaping(data):
    write(data / "occupation_narratives_shard_1.json",
          [{"uri": "occ/1", "title": "café manager"}])

    merge_narrative_shards.main()

    assert (data / "occupation_narratives.json").read_text() == (
        "[\n"
        "  {\n"
        '    "uri": "occ/1",\n'
        '    "title": "caf\\u00e9 manager"\n'
        "  }\n"
        "]"
    )


def test_a_corrupt_existing_file_is_not_swallowed(data):
    (data / "occupation_narratives.json").write_text("{ not json")
    with pytest.raises(json.JSONDecodeError):
        merge_narrative_shards.main()


def test_the_script_runs_as_a_program(data, capsys):
    write(data / "occupation_narratives_shard_1.json", [entry("occ/1")])

    runpy.run_module("merge_narrative_shards", run_name="__main__")

    assert "Merged total: 1 narratives" in capsys.readouterr().out
