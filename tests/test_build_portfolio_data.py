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
