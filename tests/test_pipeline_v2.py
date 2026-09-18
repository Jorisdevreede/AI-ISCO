"""The three pipeline steps end to end for --scorer v2, against golden outputs.

A characterisation test: it runs the whole chain over a small synthetic dataset in
a tmp directory and compares every file and everything printed with a golden copy,
so a change in a threshold, a key, a rounding or a heading shows up as a failure.
The same run also proves the older scorers' files are not touched.
"""

import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path

import pytest

import aggregate_scores
import build_portfolio_data
import build_site_indexes
from aiisco import rubric_v2

FIXTURES = Path(__file__).parent / "fixtures" / "aggregation"
EXPECTED = FIXTURES / "expected"
INPUTS = ("esco_occupations.json", "esco_skills.json", "skill_scores.json",
          "skill_scores_v2.json", "occupation_narratives.json")

DATA_FILES = ("occupation_scores_v2.json", "site_data_v2.json")
SITE_FILES = ("data_v2.json", "portfolio_data_v2.json", "search_index_v2.json",
              "groups_v2.json", "stats_v2.json", "skill_index_v2.json",
              "skill_occupations_v2.json", "rubric_v2.json")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A data/ and site/ pair holding the synthetic inputs, as the real tree has."""
    (tmp_path / "data" / "esco").mkdir(parents=True)
    (tmp_path / "site").mkdir()
    for name in INPUTS:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    for csv in (FIXTURES / "esco").glob("*.csv"):
        shutil.copy(csv, tmp_path / "data" / "esco" / csv.name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def step(monkeypatch, module, argv):
    """Run one pipeline step with the given command line."""
    monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py", *argv])
    return module.main()


def run_pipeline(monkeypatch, scorer="v2"):
    """Aggregate, build the portfolio dataset, build the five index files."""
    step(monkeypatch, aggregate_scores, ["--scorer", scorer])
    step(monkeypatch, build_portfolio_data, ["--scorer", scorer])
    return build_site_indexes.main(["--scorer", scorer, "--date", "2026-09-18"])


def golden(name):
    """Text of a golden file captured from the behaviour we must keep."""
    return (EXPECTED / name).read_text(encoding="utf-8")


def normalised(text):
    """The portfolio dataset with its skills keys sorted, so hash order cannot fail a test.

    The skills object is built by iterating a set of URIs, so its key order
    changes between runs; every file derived from it is sorted, so only this one
    needs normalising.
    """
    data = json.loads(text)
    data["skills"] = dict(sorted(data["skills"].items()))
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def written(workspace, relative):
    """Text of one file the pipeline wrote."""
    return (workspace / relative).read_text(encoding="utf-8")


# --- the whole chain --------------------------------------------------------

def test_the_pipeline_writes_every_v2_file(workspace, monkeypatch):
    assert run_pipeline(monkeypatch) == 0

    for name in DATA_FILES:
        assert (workspace / "data" / name).exists()
    for name in SITE_FILES:
        assert (workspace / "site" / name).exists()


@pytest.mark.parametrize("name", DATA_FILES)
def test_the_data_files_match_their_golden_copy(workspace, monkeypatch, name):
    run_pipeline(monkeypatch)
    assert written(workspace, f"data/{name}") == golden(name)


@pytest.mark.parametrize("name", ["search_index_v2.json", "groups_v2.json",
                                  "stats_v2.json", "skill_index_v2.json",
                                  "skill_occupations_v2.json", "rubric_v2.json"])
def test_the_index_files_match_their_golden_copy(workspace, monkeypatch, name):
    run_pipeline(monkeypatch)
    assert written(workspace, f"site/{name}") == golden(name)


def test_every_answers_shard_matches_its_golden_copy(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    built = sorted((workspace / "site" / "skill_answers_v2").glob("*.json"))
    expected = sorted((EXPECTED / "skill_answers_v2").glob("*.json"))

    assert [p.name for p in built] == [p.name for p in expected]
    assert [p.read_text(encoding="utf-8") for p in built] == \
        [p.read_text(encoding="utf-8") for p in expected]


def test_the_portfolio_dataset_matches_its_golden_copy(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    assert normalised(written(workspace, "site/portfolio_data_v2.json")) == \
        normalised(golden("portfolio_data_v2.json"))


def test_the_site_data_is_published_where_the_frontend_reads_it(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    assert written(workspace, "site/data_v2.json") == \
        written(workspace, "data/site_data_v2.json")


def test_each_step_prints_what_its_golden_says(workspace, monkeypatch, capsys):
    step(monkeypatch, aggregate_scores, ["--scorer", "v2"])
    assert capsys.readouterr().out == golden("aggregate_v2_stdout.txt")
    step(monkeypatch, build_portfolio_data, ["--scorer", "v2"])
    assert capsys.readouterr().out == golden("portfolio_v2_stdout.txt")
    build_site_indexes.main(["--scorer", "v2", "--date", "2026-09-18"])
    assert capsys.readouterr().out == golden("site_indexes_v2_stdout.txt")


def test_a_v2_run_leaves_the_older_scorers_files_alone(workspace, monkeypatch):
    run_pipeline(monkeypatch)

    for name in ("occupation_scores.json", "site_data.json"):
        assert not (workspace / "data" / name).exists()
    for name in ("data.json", "portfolio_data.json", "stats.json",
                 "portfolio_data_typesafe.json", "stats_typesafe.json"):
        assert not (workspace / "site" / name).exists()


# --- what the contract asks the outputs to carry ----------------------------

def test_the_aggregate_prints_both_complementarity_cut_offs(workspace, monkeypatch,
                                                            capsys):
    step(monkeypatch, aggregate_scores, ["--scorer", "v2"])
    printed = capsys.readouterr().out

    assert "Distribution by type, COMP_LEVEL = 3 (published):" in printed
    assert "Distribution by type, COMP_LEVEL = 2:" in printed
    assert "Distribution by quadrant" not in printed


def test_an_occupation_record_carries_the_shares_the_spread_and_the_type(
        workspace, monkeypatch):
    run_pipeline(monkeypatch)
    records = json.loads(written(workspace, "data/occupation_scores_v2.json"))
    clerk = next(occ for occ in records if occ["title"] == "Data entry clerk")

    assert clerk["quadrant"] == "INSULATED_PEOPLE"
    assert sum(Decimal(str(v)) for v in clerk["shares"].values()) == 1
    assert set(clerk["shares"]) == {"substituted", "assisted", "mechanised",
                                    "insulated"}
    assert clerk["why_insulated"] == "people"
    assert clerk["near_line"] is True
    assert 0 <= clerk["sigma"] <= 1
    assert clerk["mechanical_automation"] == 2.8


def test_the_site_data_leads_with_the_shares_and_the_machine_score(workspace,
                                                                   monkeypatch):
    run_pipeline(monkeypatch)
    rows = json.loads(written(workspace, "site/data_v2.json"))
    trainer = next(row for row in rows if row["title"] == "AI trainer")

    assert trainer["quadrant"] == "AUTOMATION_HEAVY"
    assert trainer["shares"]["substituted"] == 1.0
    assert "mechanical_automation" in trainer


def test_the_portfolio_dataset_says_which_scheme_and_model_produced_it(
        workspace, monkeypatch):
    run_pipeline(monkeypatch)
    data = json.loads(written(workspace, "site/portfolio_data_v2.json"))

    assert data["scheme"] == "shares"
    assert data["model"] == "jev-test"


def test_a_portfolio_occupation_carries_the_compact_share_fields(workspace,
                                                                 monkeypatch):
    run_pipeline(monkeypatch)
    data = json.loads(written(workspace, "site/portfolio_data_v2.json"))
    clerk = next(o for o in data["occupations"] if o["t"] == "Data entry clerk")

    assert clerk["q"] == "INSULATED_PEOPLE"
    assert len(clerk["sh"]) == 4
    assert clerk["nl"] is True
    assert clerk["why"] == "people"
    assert clerk["ak"] == 2.8


def test_a_portfolio_skill_carries_its_class_and_its_three_probabilities(
        workspace, monkeypatch):
    run_pipeline(monkeypatch)
    data = json.loads(written(workspace, "site/portfolio_data_v2.json"))
    by_title = {entry["t"]: entry for entry in data["skills"].values()}

    assert by_title["type documents"]["c"] == "S"
    assert by_title["type documents"]["p"] == [0.76, 0.25, 0.02]
    assert by_title["type documents"]["k"] == 2.3


def test_a_numbers_only_v2_scorer_borrows_the_published_rationale(workspace,
                                                                  monkeypatch):
    run_pipeline(monkeypatch)
    data = json.loads(written(workspace, "site/portfolio_data_v2.json"))
    by_title = {entry["t"]: entry for entry in data["skills"].values()}

    assert by_title["type documents"]["r"] == \
        "Templates and dictation cover most of it."
    assert by_title["type documents"]["rf"]["s"] == "gemini"


def test_an_unscored_skill_carries_no_class_at_all(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    data = json.loads(written(workspace, "site/portfolio_data_v2.json"))
    by_title = {entry["t"]: entry for entry in data["skills"].values()}

    assert by_title["unscored task"] == {"t": "unscored task", "a": None, "m": None}


def test_the_stats_file_describes_the_share_scheme(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    stats = json.loads(written(workspace, "site/stats_v2.json"))

    assert stats["scheme"] == "shares"
    assert stats["model"] == "jev-test"
    assert "quadrants" not in stats
    assert len(stats["types"]["order"]) == 7
    assert sum(stats["types"]["counts"].values()) == stats["occupations"]
    assert set(stats["skill_classes"]["counts"]) == {"S", "A", "M", "I"}
    assert stats["near_line"] == {"count": 2, "share": 0.2857}


def test_a_search_row_carries_the_machine_score_shares_and_near_line(workspace,
                                                                     monkeypatch):
    run_pipeline(monkeypatch)
    rows = json.loads(written(workspace, "site/search_index_v2.json"))
    trainer = next(row for row in rows if row["s"] == "ai-trainer")

    assert trainer["q"] == "AUTOMATION_HEAVY"
    assert trainer["sh"] == [1.0, 0.0, 0.0, 0.0]
    assert trainer["nl"] is False
    assert trainer["k"] == 2.3


def test_a_group_carries_the_machine_distribution_and_the_mean_shares(workspace,
                                                                      monkeypatch):
    run_pipeline(monkeypatch)
    groups = json.loads(written(workspace, "site/groups_v2.json"))

    assert set(groups["all"]["mech"]) == {"mean", "p10", "p50", "p90"}
    assert len(groups["all"]["sh"]) == 4
    assert groups["all"]["q"]["AUTOMATION_HEAVY"] == 2


def test_a_skill_index_row_carries_its_class_and_probabilities(workspace,
                                                               monkeypatch):
    run_pipeline(monkeypatch)
    rows = json.loads(written(workspace, "site/skill_index_v2.json"))
    documents = next(row for row in rows if row["t"] == "type documents")

    assert documents["c"] == "S"
    assert documents["p"] == [0.76, 0.25, 0.02]
    assert documents["k"] == 2.3


# --- the scoring itself, published for the pages ----------------------------

def test_the_rubric_is_generated_from_the_question_module(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    rubric = json.loads(written(workspace, "site/rubric_v2.json"))

    assert rubric["model"] == "jev-test"
    assert rubric["preamble"] == rubric_v2.PREAMBLE
    assert rubric["display"] == "1.5 + 2.0 * position"
    assert [q["id"] for q in rubric["questions"]] == list(rubric_v2.QUESTION_IDS)
    assert [q["label"] for q in rubric["questions"]] == [
        "Digital output", "AI substitution", "Machine automation", "AI assistance",
        "How it is exercised", "Deployment"]
    assert [c["code"] for c in rubric["classes"]] == ["S", "A", "M", "I"]
    assert rubric["classes"][0]["rule"] == "SUB >= 0.5"


def test_each_question_publishes_the_shape_its_kind_calls_for(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    by_id = {q["id"]: q for q in
             json.loads(written(workspace, "site/rubric_v2.json"))["questions"]}

    assert by_id["digital_output"]["kind"] == "yesno"
    assert [o["name"] for o in by_id["digital_output"]["options"]] == ["true", "false"]
    assert by_id["ai_substitution"]["kind"] == "levels"
    assert by_id["ai_substitution"]["levels"] == rubric_v2.SUBSTITUTION_LEVELS
    assert by_id["mode"]["kind"] == "choice"
    assert [o["name"] for o in by_id["mode"]["options"]] == \
        list(rubric_v2.options_for("mode"))


def test_only_the_two_questions_with_a_variant_publish_one(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    by_id = {q["id"]: q for q in
             json.loads(written(workspace, "site/rubric_v2.json"))["questions"]}

    assert by_id["mechanical"]["knowledge"]["levels"] == \
        rubric_v2.MECHANICAL_KNOWLEDGE_LEVELS
    assert by_id["ai_substitution"]["knowledge"]["instructions"] == \
        rubric_v2.KNOWLEDGE_SUBSTITUTION_INSTRUCTIONS
    assert by_id["complementarity"]["knowledge"] is None
    assert by_id["mode"]["knowledge"] is None


def shard_of(workspace, skill_id):
    """The answers shard one skill id lives in."""
    path = workspace / "site" / "skill_answers_v2" / f"{skill_id[:2]}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def skill_id_of(workspace, title):
    """The short id the portfolio gave one skill."""
    skills = json.loads(written(workspace, "site/portfolio_data_v2.json"))["skills"]
    return next(sid for sid, skill in skills.items() if skill["t"] == title)


def test_an_answers_shard_carries_the_stored_answers_of_its_skills(workspace,
                                                                   monkeypatch):
    run_pipeline(monkeypatch)
    sid = skill_id_of(workspace, "type documents")

    record = shard_of(workspace, sid)[sid]
    assert record == {
        "ty": "s", "d": 0.95,
        "s": [0.0, 0.05, 0.15, 0.5, 0.3],
        "k": [0.7, 0.2, 0.08, 0.02, 0.0],
        "c": [0.05, 0.25, 0.45, 0.2, 0.05],
        "mo": {"through_software": 0.7, "on_things": 0.07, "with_people": 0.07,
               "on_paper_in_place": 0.07, "directing_others": 0.07},
        "dp": {"routine": 0.7, "available": 0.1, "demonstrated": 0.1,
               "not_shown": 0.1},
        "cf": {"s": 0.8, "k": 0.75, "c": 0.7, "mo": 0.9, "dp": 0.5},
    }


def test_a_knowledge_item_says_so_in_its_shard(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    sid = skill_id_of(workspace, "sort post")
    assert shard_of(workspace, sid)[sid]["ty"] == "k"


def test_every_shard_holds_only_ids_that_start_with_its_name(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    for path in (workspace / "site" / "skill_answers_v2").glob("*.json"):
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert all(sid.startswith(path.stem) for sid in shard)
        assert list(shard) == sorted(shard)


def test_a_skill_the_scorer_never_saw_gets_no_shard_entry(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    sid = skill_id_of(workspace, "unscored task")
    published = set()
    for path in (workspace / "site" / "skill_answers_v2").glob("*.json"):
        published.update(json.loads(path.read_text(encoding="utf-8")))
    assert sid not in published


def test_a_skill_index_row_says_what_the_skill_is_and_how_it_is_exercised(
        workspace, monkeypatch):
    run_pipeline(monkeypatch)
    rows = {r["t"]: r for r in json.loads(written(workspace, "site/skill_index_v2.json"))}

    assert (rows["type documents"]["ty"], rows["type documents"]["mo"]) == ("s", "s")
    assert (rows["sort post"]["ty"], rows["sort post"]["mo"]) == ("k", "t")
    assert (rows["advise clients"]["ty"], rows["advise clients"]["mo"]) == ("s", "p")
    assert "ty" not in rows["unscored task"]


def test_the_build_stops_with_a_clear_message_when_the_scores_are_missing(
        workspace, monkeypatch):
    step(monkeypatch, aggregate_scores, ["--scorer", "v2"])
    step(monkeypatch, build_portfolio_data, ["--scorer", "v2"])
    (workspace / "data" / "skill_scores_v2.json").unlink()

    with pytest.raises(SystemExit) as stopped:
        build_site_indexes.main(["--scorer", "v2", "--date", "2026-09-18"])
    assert "publishes every skill's stored answers" in str(stopped.value)


def test_a_scored_skill_without_answers_stops_the_build(workspace, monkeypatch):
    step(monkeypatch, aggregate_scores, ["--scorer", "v2"])
    step(monkeypatch, build_portfolio_data, ["--scorer", "v2"])
    scores = json.loads(written(workspace, "data/skill_scores_v2.json"))
    (workspace / "data" / "skill_scores_v2.json").write_text(json.dumps(scores[:-6]))

    with pytest.raises(SystemExit) as stopped:
        build_site_indexes.main(["--scorer", "v2", "--date", "2026-09-18"])
    assert "have no stored answers" in str(stopped.value)


# --- the older scorers are untouched ---------------------------------------

def test_a_gemini_run_carries_none_of_the_share_fields(workspace, monkeypatch):
    step(monkeypatch, aggregate_scores, [])
    step(monkeypatch, build_portfolio_data, [])

    records = json.loads(written(workspace, "data/occupation_scores.json"))
    assert all("shares" not in occ for occ in records)
    data = json.loads(written(workspace, "site/portfolio_data.json"))
    assert "scheme" not in data
    assert all("sh" not in occ for occ in data["occupations"])
    assert all("c" not in skill for skill in data["skills"].values())


# --- what the pages actually fetch ------------------------------------------

def shard_files(directory, pattern="*.json"):
    """The names of the matching JSON files in one shard directory, sorted.

    The golden directory holds both scorers' shards, as site/ does; a v2 run
    writes the ``_v2`` half of them and the units map.
    """
    return sorted(path.name for path in directory.glob(pattern))


def assert_shards_match_golden(workspace, directory, pattern):
    """Every matching shard is byte-identical to its golden copy."""
    built, expected = workspace / "site" / directory, EXPECTED / directory
    assert shard_files(built, pattern) == shard_files(expected, pattern)
    for name in shard_files(built, pattern):
        assert (built / name).read_text(encoding="utf-8") == \
            (expected / name).read_text(encoding="utf-8")


def test_the_job_shards_and_the_units_map_match_their_golden_copies(workspace,
                                                                    monkeypatch):
    run_pipeline(monkeypatch)
    assert_shards_match_golden(workspace, "jobs", "*_v2.json")
    assert_shards_match_golden(workspace, "jobs", "units.json")


def test_the_rationale_shards_match_their_golden_copies(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    assert_shards_match_golden(workspace, "skill_notes", "*_v2.json")


def test_the_units_map_places_every_job_in_a_unit_group(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    units = json.loads(written(workspace, "site/jobs/units.json"))
    occupations = json.loads(
        written(workspace, "site/portfolio_data_v2.json"))["occupations"]

    placed = {occ["s"] for occ in occupations if occ["c"]}
    assert set(units) == placed
    assert all(len(code) == 4 and code.isdigit() for code in units.values())


def test_the_units_map_carries_no_scorer_suffix(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    assert (workspace / "site" / "jobs" / "units.json").exists()
    assert not (workspace / "site" / "jobs" / "units_v2.json").exists()


def test_a_job_shard_is_the_dataset_restricted_to_one_unit_group(workspace,
                                                                 monkeypatch):
    run_pipeline(monkeypatch)
    shard = json.loads(written(workspace, "site/jobs/2512_v2.json"))
    whole = json.loads(written(workspace, "site/portfolio_data_v2.json"))

    assert list(shard) == ["skills", "occupations", "scheme", "model", "neighbours"]
    assert [o["t"] for o in shard["occupations"]] == ["Software developer"]
    assert all(o["c"] != "2512" for o in shard["neighbours"])
    assert shard["scheme"] == whole["scheme"]
    assert shard["model"] == whole["model"]


def test_a_job_shard_carries_the_skills_of_its_jobs_and_their_neighbours(workspace,
                                                                         monkeypatch):
    run_pipeline(monkeypatch)
    shard = json.loads(written(workspace, "site/jobs/2512_v2.json"))

    needed = {sid for occ in shard["occupations"] + shard["neighbours"]
              for sid in occ["se"] + occ["so"]}
    assert set(shard["skills"]) == needed


def test_a_job_shard_record_is_the_one_the_dataset_publishes(workspace, monkeypatch):
    run_pipeline(monkeypatch)
    shard = json.loads(written(workspace, "site/jobs/2512_v2.json"))
    whole = json.loads(written(workspace, "site/portfolio_data_v2.json"))
    by_slug = {occ["s"]: occ for occ in whole["occupations"]}

    assert shard["occupations"][0] == by_slug["software-developer"]
    assert all(neighbour == by_slug[neighbour["s"]]
               for neighbour in shard["neighbours"])


def test_a_rationale_shard_holds_the_notes_of_the_ids_it_is_named_for(workspace,
                                                                      monkeypatch):
    run_pipeline(monkeypatch)
    skills = json.loads(written(workspace, "site/portfolio_data_v2.json"))["skills"]

    for path in (workspace / "site" / "skill_notes").glob("*_v2.json"):
        shard = json.loads(path.read_text(encoding="utf-8"))
        prefix = path.stem.removesuffix("_v2")
        assert all(sid.startswith(prefix) for sid in shard)
        for sid, note in shard.items():
            assert note["r"] == skills[sid]["r"]


def test_every_group_says_how_many_of_its_jobs_are_near_a_cut_off(workspace,
                                                                  monkeypatch):
    run_pipeline(monkeypatch)
    groups = json.loads(written(workspace, "site/groups_v2.json"))
    stats = json.loads(written(workspace, "site/stats_v2.json"))

    assert groups["all"]["near"] == stats["near_line"]["count"]
    assert all("near" in group for group in groups.values())
