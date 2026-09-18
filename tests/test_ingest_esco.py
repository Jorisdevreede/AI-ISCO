"""End-to-end characterisation of ingest_esco.py.

The golden files under tests/fixtures/esco/ were produced by the script itself
from the synthetic CSVs in the same directory. They pin the exact bytes and the
exact printed summary, so any refactoring that shifts key order, sort order,
indentation or rounding fails here.
"""

import csv
import json
import os
import runpy
import shutil

import pytest

import ingest_esco

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "esco")
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ingest_esco.py")

CSV_NAMES = (
    "occupations_en.csv",
    "skills_en.csv",
    "occupationSkillRelations_en.csv",
    "ISCOGroups_en.csv",
    "broaderRelationsOccPillar_en.csv",
)


def golden(name):
    """Read a golden fixture as text."""
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def esco_run(tmp_path, monkeypatch):
    """Copy the synthetic CSVs into tmp_path/data/esco and work from there."""
    esco_dir = tmp_path / "data" / "esco"
    esco_dir.mkdir(parents=True)
    for name in CSV_NAMES:
        shutil.copyfile(os.path.join(FIXTURES, name), esco_dir / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_csv(path, header, rows):
    """Write a small CSV shaped like the ESCO downloads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_main_writes_byte_identical_occupations_json(esco_run, capsys):
    ingest_esco.main()
    capsys.readouterr()
    written = (esco_run / "data" / "esco_occupations.json").read_text(encoding="utf-8")
    assert written == golden("expected_esco_occupations.json")


def test_main_writes_byte_identical_skills_json(esco_run, capsys):
    ingest_esco.main()
    capsys.readouterr()
    written = (esco_run / "data" / "esco_skills.json").read_text(encoding="utf-8")
    assert written == golden("expected_esco_skills.json")


def test_main_prints_the_expected_summary(esco_run, capsys):
    ingest_esco.main()
    assert capsys.readouterr().out == golden("expected_stdout.txt")


def test_main_is_deterministic_across_runs(esco_run, capsys):
    ingest_esco.main()
    first = capsys.readouterr().out
    occupations = (esco_run / "data" / "esco_occupations.json").read_bytes()
    skills = (esco_run / "data" / "esco_skills.json").read_bytes()

    ingest_esco.main()
    assert capsys.readouterr().out == first
    assert (esco_run / "data" / "esco_occupations.json").read_bytes() == occupations
    assert (esco_run / "data" / "esco_skills.json").read_bytes() == skills


def test_skill_usage_counts_are_written_as_integers(esco_run, capsys):
    ingest_esco.main()
    capsys.readouterr()
    skills = json.loads((esco_run / "data" / "esco_skills.json").read_text(encoding="utf-8"))
    counts = {s["uri"]: (s["essential_for_count"], s["optional_for_count"]) for s in skills}
    assert all(
        isinstance(n, int) and not isinstance(n, bool)
        for pair in counts.values()
        for n in pair
    )
    # A duplicated relation row is counted twice: occ-05 lists skill s10 twice.
    assert counts["http://example.org/esco/skill/s10"] == (2, 0)
    # A skill nothing points at keeps a zero, it is not dropped.
    assert counts["http://example.org/esco/skill/s05"] == (0, 0)


def test_running_the_script_ingests_the_data(esco_run, capsys):
    """README documents `uv run python ingest_esco.py`, so __main__ must keep working."""
    runpy.run_path(SCRIPT, run_name="__main__")
    assert capsys.readouterr().out == golden("expected_stdout.txt")
    written = (esco_run / "data" / "esco_skills.json").read_text(encoding="utf-8")
    assert written == golden("expected_esco_skills.json")


def test_main_exits_when_the_data_directory_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        ingest_esco.main()
    assert exit_info.value.code == 1
    assert "ERROR: data/esco/ directory not found." in capsys.readouterr().out


def test_main_exits_when_the_data_directory_holds_no_csv(tmp_path, monkeypatch, capsys):
    (tmp_path / "data" / "esco").mkdir(parents=True)
    (tmp_path / "data" / "esco" / "readme.txt").write_text("not a csv")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        ingest_esco.main()
    assert exit_info.value.code == 1
    assert "ERROR: No CSV files found in data/esco/" in capsys.readouterr().out


def test_main_exits_when_occupations_cannot_be_read(tmp_path, monkeypatch, capsys):
    write_csv(tmp_path / "data" / "esco" / "skills_en.csv",
              ["conceptUri", "skillType", "reuseLevel", "preferredLabel", "description"],
              [["s1", "knowledge", "transversal", "a skill", "described"]])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        ingest_esco.main()
    out = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert "WARNING: Could not find Occupations CSV in data/esco/" in out
    assert "ERROR: Cannot proceed without occupations data." in out


def test_main_skips_the_statistics_when_nothing_survives_the_join(tmp_path, monkeypatch, capsys):
    """Rows without a conceptUri are dropped, which can leave both outputs empty."""
    write_csv(tmp_path / "data" / "esco" / "occupations_en.csv",
              ["conceptType", "conceptUri", "iscoGroup", "preferredLabel", "description"],
              [["Occupation", "", "1111", "row without a uri", "dropped"]])
    monkeypatch.chdir(tmp_path)
    ingest_esco.main()
    out = capsys.readouterr().out
    assert "Total occupations: 0" in out
    assert "Total skills:      0" in out
    assert "Skills per occupation" not in out
    assert "Skill type breakdown" not in out
    assert json.loads((tmp_path / "data" / "esco_occupations.json").read_text()) == []
    assert json.loads((tmp_path / "data" / "esco_skills.json").read_text()) == []
