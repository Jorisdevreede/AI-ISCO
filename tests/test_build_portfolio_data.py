"""Tests for build_portfolio_data.py.

The first tests are characterisation tests: they run main() over a small
synthetic dataset in a tmp directory and compare the whole output file and the
whole printed summary with golden copies.

The skills object of the output is keyed in set-iteration order, which CPython
randomises per process, so the comparison sorts those keys on both sides; the
values, the occupations array and the compact separators are compared verbatim.
"""

import json
import runpy
import shutil
import sys
from pathlib import Path

import build_portfolio_data

FIXTURES = Path(__file__).parent / "fixtures" / "aggregation"
EXPECTED = FIXTURES / "expected"
INPUTS = ("esco_occupations.json", "skill_scores.json",
          "skill_scores_typesafe.json", "occupation_narratives.json")


def run_main(tmp_path, monkeypatch, argv, inputs=INPUTS):
    """Run build_portfolio_data.main() over the fixture data inside tmp_path."""
    (tmp_path / "data").mkdir(exist_ok=True)
    for name in inputs:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_portfolio_data.py", *argv])
    build_portfolio_data.main()


def golden(name):
    """Text of a golden file captured from the behaviour we must keep."""
    return (EXPECTED / name).read_text(encoding="utf-8")


def normalised(text):
    """The output with its skills keys sorted, so hash order cannot fail a test."""
    # BUG: the skills object is built by iterating a set of URIs, so its key order
    # changes between runs (CPython randomises str hashing per process). Two runs
    # over identical input therefore write byte-different files with identical
    # content, which makes `git diff site/portfolio_data.json` noisy. Sorting the
    # URIs would fix it but would reorder the published file, so it is left alone.
    data = json.loads(text)
    data["skills"] = dict(sorted(data["skills"].items()))
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def test_gemini_run_writes_the_expected_portfolio_data(tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, [])
    printed = capsys.readouterr().out

    written = (tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8")
    assert normalised(written) == normalised(golden("portfolio_data.json"))
    assert printed == golden("portfolio_stdout.txt")


def test_output_is_compact_json_without_escaped_unicode(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    written = (tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8")
    assert written == json.dumps(json.loads(written), ensure_ascii=False,
                                 separators=(",", ":"))
    assert "négocier contracts" in written
    assert written.startswith('{"skills":{')


def test_typesafe_run_writes_a_suffixed_file_and_leaves_the_gemini_one_alone(
        tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, ["--scorer", "typesafe"])
    printed = capsys.readouterr().out

    written = (tmp_path / "site/portfolio_data_typesafe.json").read_text(encoding="utf-8")
    assert normalised(written) == normalised(golden("portfolio_data_typesafe.json"))
    assert printed == golden("portfolio_typesafe_stdout.txt")
    assert not (tmp_path / "site/portfolio_data.json").exists()


def test_narratives_are_compacted_and_empty_ones_are_dropped(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    data = json.loads((tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8"))
    by_title = {occ["t"]: occ for occ in data["occupations"]}
    assert by_title["Data entry clerk"]["n"]["ts"] == 30
    assert by_title["Data entry clerk"]["n"]["story"].startswith("Keying gives way")
    assert "n" not in by_title["Café manager"]
    assert by_title["Software developer"]["n"] == {"ts": 0}


def test_adjacency_lists_only_point_at_higher_evolution_potential(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    data = json.loads((tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8"))
    by_title = {occ["t"]: occ for occ in data["occupations"]}
    developer = by_title["Software developer"]
    assert [adj["t"] for adj in developer["adj"]] == ["Data entry clerk", "Café manager"]
    assert developer["adj"][0]["ov"] == 0.4
    assert all(adj["e"] > developer["e"] for adj in developer["adj"])
    assert by_title["Data entry clerk"]["adj"] == []


def test_gap_skills_are_the_most_amplifiable_missing_ones(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    data = json.loads((tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8"))
    by_title = {occ["t"]: occ for occ in data["occupations"]}
    gap = by_title["Software developer"]["adj"][0]["gap"]
    assert [data["skills"][short]["t"] for short in gap] == \
        ["train staff", "café service", "copy ledgers", "sort post", "file records"]


def test_unscored_skills_keep_null_scores_in_the_output(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    data = json.loads((tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8"))
    by_title = {entry["t"]: entry for entry in data["skills"].values()}
    assert by_title["unscored task"] == {"t": "unscored task", "a": None, "m": None}
    assert by_title["type documents"]["r"] == "Templates and dictation cover most of it."
    assert "r" not in by_title["file records"]


def strip_rationales(path):
    """Make a score file look like a numbers-only scorer wrote it."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    for entry in entries:
        entry.pop("rationale", None)
    path.write_text(json.dumps(entries), encoding="utf-8")


def run_numbers_only(tmp_path, monkeypatch, inputs=INPUTS):
    """A typesafe run whose scores carry no rationale, as the real ones do."""
    (tmp_path / "data").mkdir()
    shutil.copy(FIXTURES / "skill_scores_typesafe.json", tmp_path / "data")
    strip_rationales(tmp_path / "data/skill_scores_typesafe.json")
    run_main(tmp_path, monkeypatch, ["--scorer", "typesafe"],
             inputs=[name for name in inputs if name != "skill_scores_typesafe.json"])
    written = (tmp_path / "site/portfolio_data_typesafe.json").read_text(encoding="utf-8")
    return {entry["t"]: entry for entry in json.loads(written)["skills"].values()}


def test_a_numbers_only_scorer_borrows_the_published_rationale(tmp_path, monkeypatch, capsys):
    by_title = run_numbers_only(tmp_path, monkeypatch)

    typed = by_title["type documents"]
    assert typed["r"] == "Templates and dictation cover most of it."
    assert typed["rf"]["s"] == "gemini"
    assert "Rationales borrowed from the gemini scores: 5" in capsys.readouterr().out


def test_a_borrowed_rationale_says_which_scores_it_was_written_for(tmp_path, monkeypatch):
    by_title = run_numbers_only(tmp_path, monkeypatch)
    published = {entry["title"]: entry for entry in
                 json.loads((FIXTURES / "skill_scores.json").read_text(encoding="utf-8"))}

    source = by_title["type documents"]["rf"]
    assert source["a"] == float(published["type documents"]["automation_risk"])
    assert source["m"] == float(published["type documents"]["amplification_potential"])
    assert "rf" not in by_title["file records"]  # nothing published to borrow
    assert "r" not in by_title["file records"]


def test_nothing_is_borrowed_when_the_published_scores_are_absent(
        tmp_path, monkeypatch, capsys):
    by_title = run_numbers_only(tmp_path, monkeypatch,
                                inputs=("esco_occupations.json", "skill_scores_typesafe.json"))

    assert all("r" not in entry and "rf" not in entry for entry in by_title.values())
    assert "Rationales borrowed" not in capsys.readouterr().out


def test_the_published_scorer_never_marks_a_rationale_as_borrowed(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    data = json.loads((tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8"))
    assert all("rf" not in entry for entry in data["skills"].values())


def test_missing_narratives_file_is_only_a_warning(tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, [], inputs=("esco_occupations.json", "skill_scores.json"))
    printed = capsys.readouterr().out

    data = json.loads((tmp_path / "site/portfolio_data.json").read_text(encoding="utf-8"))
    assert "WARNING: data/occupation_narratives.json not found" in printed
    assert "Occupations with narratives" not in printed
    assert all("n" not in occ for occ in data["occupations"])


def test_running_the_module_as_a_script_builds_the_same_dataset(
        tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    for name in INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_portfolio_data.py"])

    runpy.run_module("build_portfolio_data", run_name="__main__")

    assert capsys.readouterr().out == golden("portfolio_stdout.txt")


def test_a_run_without_any_scored_skill_still_writes_an_empty_dataset(
        tmp_path, monkeypatch, capsys):
    (tmp_path / "data").mkdir()
    for name in INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    (tmp_path / "data/skill_scores.json").write_text("[]", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_portfolio_data.py"])

    build_portfolio_data.main()

    printed = capsys.readouterr().out
    assert json.loads((tmp_path / "site/portfolio_data.json").read_text()) == \
        {"skills": {}, "occupations": []}
    assert "  Occupations with adjacency: 0 / 0" in printed
    assert "    TRANSFORM   :    0 (  0.0%)" in printed


# --- what the pages fetch, per scorer ---------------------------------------

def shard_names(directory, pattern="*.json"):
    """The names of the matching JSON files in a shard directory, sorted."""
    return sorted(path.name for path in directory.glob(pattern)) \
        if directory.exists() else []


def test_the_published_scorer_writes_the_units_map_and_one_file_per_unit_group(
        tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    jobs = tmp_path / "site/jobs"
    assert "units.json" in shard_names(jobs)
    assert shard_names(jobs, "[0-9]*.json") == ["1412.json", "2423.json",
                                                "2512.json", "2519.json",
                                                "4132.json", "4412.json"]
    assert json.loads((jobs / "units.json").read_text())["software-developer"] == "2512"


def test_the_published_scorer_writes_the_rationale_shards(tmp_path, monkeypatch):
    run_main(tmp_path, monkeypatch, [])

    notes = tmp_path / "site/skill_notes"
    assert shard_names(notes)
    every = {}
    for name in shard_names(notes):
        every.update(json.loads((notes / name).read_text()))
    data = json.loads((tmp_path / "site/portfolio_data.json").read_text())
    with_text = {sid for sid, skill in data["skills"].items() if skill.get("r")}
    assert set(every) == with_text


def test_a_shard_never_carries_a_suffix_the_switch_does_not_offer(tmp_path,
                                                                  monkeypatch):
    run_main(tmp_path, monkeypatch, ["--scorer", "typesafe"])

    assert shard_names(tmp_path / "site/jobs") == []
    assert shard_names(tmp_path / "site/skill_notes") == []
    assert (tmp_path / "site/portfolio_data_typesafe.json").exists()


def test_the_run_reports_what_the_pages_will_fetch(tmp_path, monkeypatch, capsys):
    run_main(tmp_path, monkeypatch, [])
    printed = capsys.readouterr().out

    assert "site/jobs/units.json" in printed
    assert "jobs " in printed and "skill_notes " in printed
    assert "files" in printed
