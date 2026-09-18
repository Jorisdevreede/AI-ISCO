"""Behaviour of the scoring v2 skill scorer: answers in, records out, resuming.

Every number here is invented, no request leaves the machine, and nothing is
written outside tmp_path. The real run's output is data/skill_scores_v2.json.
"""

import json
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typesafe_sdk import TypeSafeError

import score_skills_v2 as scorer
from aiisco import rubric_v2

FIXTURES = Path(__file__).parent / "fixtures" / "scoring"

SKILL = {"uri": "u/a", "title": "brew coffee", "description": " Prepare. ",
         "type": "skill"}

# The distributions of the worked example in docs/scoring-v2.md.
SUBSTITUTION_PROBS = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.5, 4: 0.1}
MECHANICAL_PROBS = {0: 0.85, 1: 0.1, 2: 0.05, 3: 0.0, 4: 0.0}
COMPLEMENTARITY_PROBS = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.4, 4: 0.27}


def noul(yes):
    """A NoulAnswer stand-in: the probability of yes, and no other field at all."""
    return SimpleNamespace(type="noul", noul=yes)


def score(position, confidence, probs):
    """A ScoreAnswer stand-in over the five rubric levels."""
    return SimpleNamespace(type="score", score=position, confidence=confidence,
                           probabilities=probs)


def choice(picked, confidence, probs):
    """A ChoiceAnswer stand-in over named options."""
    return SimpleNamespace(type="choice", choice=picked, confidence=confidence,
                           probabilities=probs)


def answers(**overrides):
    """One answer per question of the rubric, in a deliberately jumbled order."""
    given = {
        "mode": choice("through_software", 0.8,
                       {"through_software": 0.8, "with_people": 0.2}),
        "deployment": choice("routine", 0.6, {"routine": 0.6, "available": 0.4}),
        "complementarity": score(2.9, 0.66, COMPLEMENTARITY_PROBS),
        "mechanical": score(0.2, 0.8, MECHANICAL_PROBS),
        "ai_substitution": score(2.6, 0.61, SUBSTITUTION_PROBS),
        "digital_output": noul(0.93),
    }
    given.update(overrides)
    return given


def answers_dict():
    """The same answers in the stored form, as the scores file holds them."""
    return scorer.answers_of(SimpleNamespace(answers=answers()))


class FakeClient:
    """A TypeSafeClient whose system_one answers from a script."""

    def __init__(self, failures=(), crashes=()):
        self.failures = set(failures)
        self.crashes = set(crashes)
        self.calls = []

    def system_one(self, state, questions, model):
        title = state["skill"]["title"]
        self.calls.append(SimpleNamespace(state=state, questions=questions,
                                          model=model))
        if title in self.failures:
            raise TypeSafeError(f"refused {title}")
        if title in self.crashes:
            raise RuntimeError(f"crashed on {title}")
        return SimpleNamespace(answers=answers(), model="jev-test",
                               usage=SimpleNamespace(input_tokens=17))


class NullLimiter:
    """A rate limiter that never waits."""

    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def workspace(tmp_path, monkeypatch, *names):
    """Copy the named scoring fixtures into tmp_path/data and work there."""
    (tmp_path / "data").mkdir(exist_ok=True)
    for name in names:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)


def run_main(monkeypatch, *argv):
    """Run main() with the given command line."""
    monkeypatch.setattr(sys, "argv", ["score_skills_v2.py", *argv])
    scorer.main()


def install(monkeypatch, client):
    """Make main() build the given fake client instead of a real one."""
    monkeypatch.setattr(scorer, "TypeSafeClient", lambda api_key: client)


def written(tmp_path):
    """The scores file main() wrote, parsed."""
    return json.loads((tmp_path / "data" / "skill_scores_v2.json").read_text())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """A synthetic key, a frozen clock and no keychain or network access."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("a test reached the keychain"),
    )
    monkeypatch.setattr(
        scorer, "TypeSafeClient",
        lambda **kwargs: pytest.fail("a test built a real TypeSafe client"),
    )


# --- the contract the script promises ---------------------------------------

def test_the_module_says_where_its_output_lives_and_claims_no_affiliation():
    assert "data/skill_scores_v2.json" in scorer.__doc__
    assert "endorsed by TypeSafe AI, Inc." in scorer.__doc__


def test_constants_pin_the_cli_contract():
    assert scorer.MODEL == "jev-1.13.0"
    assert scorer.INPUT_FILE == "data/esco_skills.json"
    assert scorer.OUTPUT_FILE == "data/skill_scores_v2.json"
    assert scorer.CHECKPOINT_EVERY == 250


def test_the_model_is_pinned_rather_than_a_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["score_skills_v2.py"])
    assert not hasattr(scorer.parse_args(), "model")


def test_the_pool_defaults_to_six_workers_at_ten_requests_a_second(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["score_skills_v2.py"])
    args = scorer.parse_args()
    assert (args.workers, args.rps) == (6, 10.0)


# --- serialising the three answer types -------------------------------------

def test_a_noul_answer_carries_only_the_probability_of_yes():
    assert scorer.serialise("digital_output", noul(0.9312)) == {"yes": 0.9312}


def test_a_noul_answer_is_read_without_touching_a_confidence_field():
    bare = SimpleNamespace(type="noul", noul=0.5)  # a NoulAnswer has nothing else
    assert scorer.serialise("digital_output", bare) == {"yes": 0.5}


def test_a_graded_answer_keeps_the_whole_distribution_in_level_order():
    jumbled = score(2.6, 0.6123, {3: 0.5, 0: 0.0, 4: 0.1, 1: 0.1, 2: 0.3})
    assert scorer.serialise("ai_substitution", jumbled) == {
        "probs": [0.0, 0.1, 0.3, 0.5, 0.1], "position": 2.6, "confidence": 0.612}


def test_a_choice_answer_lists_every_option_it_was_offered():
    picked = choice("routine", 0.6, {"routine": 0.6, "available": 0.4})
    assert scorer.serialise("deployment", picked) == {
        "choice": "routine",
        "probs": {"routine": 0.6, "available": 0.4, "demonstrated": 0.0,
                  "not_shown": 0.0},
        "confidence": 0.6,
    }


def test_a_choice_answer_keeps_the_order_the_options_were_offered_in():
    picked = choice("on_things", 1.0, {"on_things": 1.0})
    assert tuple(scorer.serialise("mode", picked)["probs"]) == \
        rubric_v2.options_for("mode")


def test_the_answers_are_written_in_the_rubric_order():
    response = SimpleNamespace(answers=answers())
    assert tuple(scorer.answers_of(response)) == rubric_v2.QUESTION_IDS


# --- scoring one skill ------------------------------------------------------

def test_score_skill_returns_the_published_record_shape():
    client = FakeClient()
    limiter = NullLimiter()

    entry = scorer.score_skill(client, limiter, SKILL)

    assert entry == {
        "uri": "u/a",
        "title": "brew coffee",
        "type": "skill",
        "answers": {
            "digital_output": {"yes": 0.93},
            "ai_substitution": {"probs": [0.0, 0.1, 0.3, 0.5, 0.1],
                                "position": 2.6, "confidence": 0.61},
            "mechanical": {"probs": [0.85, 0.1, 0.05, 0.0, 0.0],
                           "position": 0.2, "confidence": 0.8},
            "complementarity": {"probs": [0.0, 0.03, 0.3, 0.4, 0.27],
                                "position": 2.9, "confidence": 0.66},
            "mode": {"choice": "through_software",
                     "probs": {"through_software": 0.8, "on_things": 0.0,
                               "with_people": 0.2, "on_paper_in_place": 0.0,
                               "directing_others": 0.0},
                     "confidence": 0.8},
            "deployment": {"choice": "routine",
                           "probs": {"routine": 0.6, "available": 0.4,
                                     "demonstrated": 0.0, "not_shown": 0.0},
                           "confidence": 0.6},
        },
        "sub": 0.558,
        "comp": 0.67,
        "comp_part": 0.97,
        "mech": 0.0,
        "class": "S",
        "automation_risk": 6.7,
        "amplification_potential": 7.3,
        "mechanical_automation": 1.9,
        "model": "jev-test",
        "input_tokens": 17,
    }
    assert limiter.waits == 1


def test_score_skill_sends_the_skill_as_trimmed_state_and_the_pinned_model():
    client = FakeClient()
    scorer.score_skill(client, NullLimiter(), SKILL)
    assert client.calls[0].state == {
        "skill": {"title": "brew coffee", "description": "Prepare.",
                  "type": "skill", "reuse_level": ""},
    }
    assert client.calls[0].model == "jev-1.13.0"


def test_score_skill_asks_a_plain_skill_the_skill_wording():
    client = FakeClient()
    scorer.score_skill(client, NullLimiter(), SKILL)
    assert client.calls[0].questions is rubric_v2.questions_for("skill")


def test_score_skill_asks_a_knowledge_item_the_knowledge_variants():
    client = FakeClient()
    knowledge = {"uri": "u/b", "title": "thermodynamics", "type": "knowledge"}

    scorer.score_skill(client, NullLimiter(), knowledge)

    asked = client.calls[0].questions
    assert asked is rubric_v2.questions_for("knowledge")
    assert list(asked["mechanical"].criteria) == rubric_v2.MECHANICAL_KNOWLEDGE_LEVELS
    assert list(asked["ai_substitution"].criteria) == \
        rubric_v2.KNOWLEDGE_SUBSTITUTION_LEVELS


def test_score_skill_asks_all_six_questions_in_one_request():
    client = FakeClient()
    scorer.score_skill(client, NullLimiter(), SKILL)
    assert len(client.calls) == 1
    assert tuple(client.calls[0].questions) == rubric_v2.QUESTION_IDS


# --- the file it writes -----------------------------------------------------

def test_save_writes_indented_json_and_removes_its_temporary_file(tmp_path,
                                                                  monkeypatch):
    workspace(tmp_path, monkeypatch)
    scorer.save({"u/a": {"uri": "u/a", "title": "café"}})

    output = tmp_path / "data" / "skill_scores_v2.json"
    assert output.read_text() == (
        "[\n {\n  \"uri\": \"u/a\",\n  \"title\": \"caf\\u00e9\"\n }\n]"
    )
    assert not (tmp_path / "data" / "skill_scores_v2.json.tmp").exists()


def test_main_writes_one_record_per_skill_in_input_order(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "2", "--workers", "1")

    assert [entry["uri"] for entry in written(tmp_path)] == [
        "http://example.org/skill/a", "http://example.org/skill/b"]
    assert written(tmp_path)[0]["class"] == "S"


def test_main_prints_the_run_summary(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "1", "--workers", "1")

    assert capsys.readouterr().out == (
        "Scoring 1 skills with jev-1.13.0\n"
        "Already scored: 0\n"
        "Remaining to score: 1\n"
        "\nDone in 0s. Skills scored: 1, errors: 0.\n"
        "Input tokens this run: 17\n"
        "\nSkill classes across 1 skills:\n"
        "  S:      1 (100.0%)\n"
        "  A:      0 (  0.0%)\n"
        "  M:      0 (  0.0%)\n"
        "  I:      0 (  0.0%)\n"
    )


def test_a_run_that_scored_nothing_says_so_instead_of_dividing_by_zero(
        tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient(failures=["brew coffee"]))

    run_main(monkeypatch, "--end", "1", "--workers", "1")

    out = capsys.readouterr().out
    assert "Skills scored: 0, errors: 1." in out
    assert "Nothing was scored, so there is no distribution to show." in out


# --- resuming, sampling, stopping -------------------------------------------

def test_main_resumes_from_the_checkpoint_without_asking_again(tmp_path,
                                                               monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    (tmp_path / "data" / "skill_scores_v2.json").write_text(json.dumps(
        [{"uri": "http://example.org/skill/a", "title": "brew coffee",
          "class": "I", "input_tokens": 1}]))
    client = FakeClient()
    install(monkeypatch, client)

    run_main(monkeypatch, "--end", "2", "--workers", "1")

    assert [call.state["skill"]["title"] for call in client.calls] == \
        ["types of sugars"]
    assert {entry["uri"] for entry in written(tmp_path)} == {
        "http://example.org/skill/a", "http://example.org/skill/b"}


def test_main_stops_when_everything_is_already_scored(tmp_path, monkeypatch,
                                                      capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    (tmp_path / "data" / "skill_scores_v2.json").write_text(json.dumps(
        [{"uri": "http://example.org/skill/a"}]))

    run_main(monkeypatch, "--end", "1")

    assert "Nothing to score. Use --force to re-score all." in capsys.readouterr().out


def test_main_force_rescores_everything(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    (tmp_path / "data" / "skill_scores_v2.json").write_text(json.dumps(
        [{"uri": "http://example.org/skill/a", "class": "I"}]))
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "1", "--force", "--workers", "1")

    assert [entry["class"] for entry in written(tmp_path)] == ["S"]


def test_main_samples_a_fixed_subset_for_a_given_seed(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--sample", "2", "--seed", "7", "--workers", "1")

    assert sorted(entry["uri"] for entry in written(tmp_path)) == [
        "http://example.org/skill/a", "http://example.org/skill/c"]


def test_main_checkpoints_while_the_pool_runs(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    monkeypatch.setattr(scorer, "CHECKPOINT_EVERY", 1)
    clock = iter(1000.0 + step for step in range(1000))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "2", "--workers", "1")

    out = capsys.readouterr().out
    assert "  1/2 (" in out
    assert "34 tokens, errors 0)" in out


def many_skills(tmp_path, count):
    """A synthetic input file big enough to trip the failure guard."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "esco_skills.json").write_text(json.dumps(
        [{"uri": f"u/{i}", "title": f"skill {i}", "type": "skill"}
         for i in range(count)]))


def test_main_stops_cleanly_after_twenty_consecutive_failures(tmp_path,
                                                              monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    many_skills(tmp_path, 60)
    install(monkeypatch, FakeClient(failures=[f"skill {i}" for i in range(60)]))

    run_main(monkeypatch, "--workers", "1")

    out = capsys.readouterr().out
    assert "Stopped early: 20 requests in a row failed. Re-run to resume." in out
    assert "Skills scored: 0, errors: 20." in out


def test_main_writes_the_checkpoint_even_when_the_pool_raises(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    (tmp_path / "data" / "skill_scores_v2.json").write_text(json.dumps(
        [{"uri": "http://example.org/skill/a", "input_tokens": 1}]))
    install(monkeypatch, FakeClient(crashes=["types of sugars"]))

    with pytest.raises(RuntimeError, match="crashed on types of sugars"):
        run_main(monkeypatch, "--end", "2", "--workers", "1")

    assert [entry["uri"] for entry in written(tmp_path)] == [
        "http://example.org/skill/a"]


# --- rederiving from the answers already on disk ----------------------------

STALE = {
    "uri": "u/a", "title": "brew coffee", "type": "skill", "answers": answers_dict(),
    "sub": 0.0, "comp": 0.0, "comp_strict": 0.9, "mech": 0.0, "class": "I",
    "automation_risk": 1.5, "amplification_potential": 1.5,
    "mechanical_automation": 1.5, "model": "jev-1.13.0", "input_tokens": 17,
}


def with_scores(tmp_path, monkeypatch, entries):
    """A workspace whose scores file holds the given entries."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "skill_scores_v2.json").write_text(json.dumps(entries))
    monkeypatch.chdir(tmp_path)


def test_rederive_refreshes_the_derived_fields_from_the_stored_answers(
        tmp_path, monkeypatch):
    with_scores(tmp_path, monkeypatch, [STALE])

    run_main(monkeypatch, "--rederive")

    entry = written(tmp_path)[0]
    assert (entry["sub"], entry["comp"], entry["comp_part"], entry["class"]) == \
        (0.558, 0.67, 0.97, "S")
    assert entry["automation_risk"] == 6.7


def test_rederive_drops_a_field_a_rule_change_has_retired(tmp_path, monkeypatch):
    with_scores(tmp_path, monkeypatch, [STALE])
    run_main(monkeypatch, "--rederive")
    assert "comp_strict" not in written(tmp_path)[0]


def test_rederive_leaves_the_answers_and_the_provenance_untouched(tmp_path,
                                                                  monkeypatch):
    with_scores(tmp_path, monkeypatch, [STALE])

    run_main(monkeypatch, "--rederive")

    entry = written(tmp_path)[0]
    assert entry["answers"] == STALE["answers"]
    assert (entry["uri"], entry["title"], entry["type"]) == ("u/a", "brew coffee",
                                                             "skill")
    assert (entry["model"], entry["input_tokens"]) == ("jev-1.13.0", 17)


def test_rederive_writes_the_contract_key_order(tmp_path, monkeypatch):
    with_scores(tmp_path, monkeypatch, [STALE])
    run_main(monkeypatch, "--rederive")
    assert list(written(tmp_path)[0]) == [
        "uri", "title", "type", "answers", "sub", "comp", "comp_part", "mech",
        "class", "automation_risk", "amplification_potential",
        "mechanical_automation", "model", "input_tokens"]


def test_rederive_keeps_the_order_the_file_already_has(tmp_path, monkeypatch):
    entries = [dict(STALE, uri=f"u/{i}", title=f"skill {i}") for i in range(5)]
    with_scores(tmp_path, monkeypatch, entries)

    run_main(monkeypatch, "--rederive")

    assert [e["uri"] for e in written(tmp_path)] == [f"u/{i}" for i in range(5)]


def test_rederive_is_idempotent(tmp_path, monkeypatch):
    with_scores(tmp_path, monkeypatch, [STALE])
    run_main(monkeypatch, "--rederive")
    once = (tmp_path / "data" / "skill_scores_v2.json").read_text()
    run_main(monkeypatch, "--rederive")
    assert (tmp_path / "data" / "skill_scores_v2.json").read_text() == once


def test_rederive_needs_no_key_and_asks_the_model_nothing(tmp_path, monkeypatch,
                                                          capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY")  # the keychain is already a test failure
    with_scores(tmp_path, monkeypatch, [STALE])

    run_main(monkeypatch, "--rederive")

    out = capsys.readouterr().out
    assert "Rederived 1 skills in data/skill_scores_v2.json" in out
    assert "  S:      1 (100.0%)" in out


def test_the_script_runs_as_a_program(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["score_skills_v2.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("score_skills_v2", run_name="__main__")
    assert "usage: score_skills_v2.py" in capsys.readouterr().out
