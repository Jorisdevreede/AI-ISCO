"""Behaviour of the TypeSafe skill scorer: rubric, keys, pool, checkpoints.

Every number here is invented, no request leaves the machine, and nothing is
written outside tmp_path. The real run's output is data/skill_scores_typesafe.json.
"""

import hashlib
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

import score_skills_typesafe as typesafe

FIXTURES = Path(__file__).parent / "fixtures" / "scoring"

AUTOMATION_PROBS = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.25, 4: 0.15}
AMPLIFICATION_PROBS = {0: 0.05, 1: 0.05, 2: 0.4, 3: 0.3, 4: 0.2}

SKILL = {"uri": "u/a", "title": "brew coffee", "description": " Prepare. ",
         "type": "skill"}


def answers(auto, amp):
    """Two ScoreAnswer-shaped stand-ins, one per axis."""
    return {
        "automation": SimpleNamespace(score=auto, confidence=0.875,
                                      probabilities=AUTOMATION_PROBS),
        "amplification": SimpleNamespace(score=amp, confidence=0.5,
                                         probabilities=AMPLIFICATION_PROBS),
    }


class FakeClient:
    """A TypeSafeClient whose system_one answers from a script."""

    def __init__(self, scores=None, failures=(), crashes=()):
        self.scores = scores or {}
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
        auto, amp = self.scores.get(title, (2.0, 3.0))
        return SimpleNamespace(answers=answers(auto, amp), model="jev-test",
                               usage=SimpleNamespace(input_tokens=11))


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
    monkeypatch.setattr(sys, "argv", ["score_skills_typesafe.py", *argv])
    typesafe.main()


def install(monkeypatch, client):
    """Make main() build the given fake client instead of a real one."""
    monkeypatch.setattr(typesafe, "TypeSafeClient", lambda api_key: client)


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
        typesafe, "TypeSafeClient",
        lambda **kwargs: pytest.fail("a test built a real TypeSafe client"),
    )


def test_the_module_says_where_its_output_lives_and_claims_no_affiliation():
    assert "data/skill_scores_typesafe.json" in typesafe.__doc__
    assert "endorsed by TypeSafe AI, Inc." in typesafe.__doc__


def test_constants_pin_the_cli_contract():
    assert typesafe.DEFAULT_MODEL == "jev-1.13.0"
    assert typesafe.INPUT_FILE == "data/esco_skills.json"
    assert typesafe.OUTPUT_FILE == "data/skill_scores_typesafe.json"
    assert typesafe.CHECKPOINT_EVERY == 250


def digest(value):
    """A stable hash of one rubric or instruction string."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize(("name", "expected"), [
    ("AUTOMATION_LEVELS",
     "5e62d9a1789087cafb1f37e89a2903497f4383d836709940239050e69c2ace71"),
    ("AMPLIFICATION_LEVELS",
     "d5b4ed8b885731a707c934a98708eeb1f903f067c9fd3c1c2e77e8daf3106656"),
    ("KNOWLEDGE_AUTOMATION_LEVELS",
     "dc3e3830cefafcefaed4b7af95ec0d2c339dc3e86528f08d4a5ea03df720b7f3"),
])
def test_rubric_levels_are_unchanged(name, expected):
    assert digest(getattr(typesafe, name)) == expected


@pytest.mark.parametrize(("question", "expected"), [
    (lambda: typesafe.KNOWLEDGE_AUTOMATION.instructions,
     "7408784ba27d3673bd1f111de894c328ae88e803ba7d56452f2e0204092abc88"),
    (lambda: typesafe.QUESTIONS["automation"].instructions,
     "2460be771123a7297d892ca126176ad153b6a4bcc3d305adcc9fe09e588760a3"),
    (lambda: typesafe.QUESTIONS["amplification"].instructions,
     "339875293f1505886123fd4b3892670f410da02b9ebb8c8e70be6014d2f88e70"),
])
def test_question_instructions_are_unchanged(question, expected):
    assert digest(question()) == expected


def test_questions_use_the_matching_rubric():
    assert typesafe.QUESTIONS["automation"].criteria == typesafe.AUTOMATION_LEVELS
    assert typesafe.QUESTIONS["amplification"].criteria == (
        typesafe.AMPLIFICATION_LEVELS)
    assert typesafe.KNOWLEDGE_AUTOMATION.criteria == (
        typesafe.KNOWLEDGE_AUTOMATION_LEVELS)


@pytest.mark.parametrize(("level", "expected"), [
    (0, 1.5), (1, 3.5), (2, 5.5), (3, 7.5), (4, 9.5), (1.234, 3.97),
])
def test_to_ten_scale_maps_bands_onto_the_rubric_scale(level, expected):
    assert typesafe.to_ten_scale(level) == expected


def test_load_api_key_prefers_the_environment():
    assert typesafe.load_api_key() == "test-key"


def test_load_api_key_falls_back_to_the_login_keychain(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="keychain-key\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert typesafe.load_api_key() == "keychain-key"
    assert calls[0][0] == ["security", "find-generic-password", "-a",
                           "typesafe", "-s", "typesafe-api-key", "-w"]
    assert calls[0][1] == {"capture_output": True, "text": True, "check": False}


def test_load_api_key_ignores_an_empty_environment_variable(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="from-keychain"),
    )
    assert typesafe.load_api_key() == "from-keychain"


def test_load_api_key_exits_when_there_is_no_key_anywhere(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=44, stdout="", stderr="no"),
    )
    with pytest.raises(SystemExit) as exit_info:
        typesafe.load_api_key()
    assert "No TypeSafe key: set TYPESAFE_API_KEY or add the keychain item" in (
        str(exit_info.value))


def test_rate_limiter_spaces_starts_and_skips_the_wait_when_it_is_late(
        monkeypatch):
    clock = [100.0]
    recorded = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", recorded.append)

    limiter = typesafe.RateLimiter(4.0)
    limiter.wait()
    limiter.wait()
    assert recorded == [0.25]

    clock[0] = 200.0
    limiter.wait()
    assert recorded == [0.25]
    assert limiter.next_at == 200.25


def test_score_skill_returns_the_published_record_shape():
    client = FakeClient({"brew coffee": (2.0, 3.0)})
    limiter = NullLimiter()

    entry = typesafe.score_skill(client, limiter, SKILL, "jev-test")

    assert entry == {
        "uri": "u/a",
        "title": "brew coffee",
        "automation_risk": 5.5,
        "amplification_potential": 7.5,
        "automation_probs": [0.1, 0.2, 0.3, 0.25, 0.15],
        "amplification_probs": [0.05, 0.05, 0.4, 0.3, 0.2],
        "automation_confidence": 0.875,
        "amplification_confidence": 0.5,
        "model": "jev-test",
        "input_tokens": 11,
    }
    assert limiter.waits == 1


def test_score_skill_sends_the_skill_as_trimmed_state():
    client = FakeClient()
    typesafe.score_skill(client, NullLimiter(), SKILL, "jev-test")
    assert client.calls[0].state == {
        "skill": {"title": "brew coffee", "description": "Prepare.",
                  "type": "skill"},
    }
    assert client.calls[0].model == "jev-test"


def test_score_skill_tolerates_a_skill_without_description_or_type():
    client = FakeClient()
    typesafe.score_skill(client, NullLimiter(), {"uri": "u/x", "title": "x"},
                         "m")
    assert client.calls[0].state["skill"] == {"title": "x", "description": "",
                                             "type": ""}


def test_score_skill_asks_knowledge_items_the_knowledge_question():
    client = FakeClient()
    knowledge = {"uri": "u/b", "title": "types of sugars", "type": "knowledge"}

    typesafe.score_skill(client, NullLimiter(), knowledge, "m")

    asked = client.calls[0].questions
    assert asked["automation"] is typesafe.KNOWLEDGE_AUTOMATION
    assert asked["amplification"] is typesafe.QUESTIONS["amplification"]


def test_score_skill_asks_plain_skills_the_standard_question():
    client = FakeClient()
    typesafe.score_skill(client, NullLimiter(), SKILL, "m")
    assert client.calls[0].questions is typesafe.QUESTIONS


def test_save_writes_indented_json_and_removes_its_temporary_file(tmp_path,
                                                                  monkeypatch):
    workspace(tmp_path, monkeypatch)
    typesafe.save({"u/a": {"uri": "u/a", "title": "café"}})

    output = tmp_path / "data" / "skill_scores_typesafe.json"
    assert output.read_text() == (
        "[\n {\n  \"uri\": \"u/a\",\n  \"title\": \"caf\\u00e9\"\n }\n]"
    )
    assert not (tmp_path / "data" / "skill_scores_typesafe.json.tmp").exists()


def test_main_writes_the_scores_file_byte_for_byte(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient({"café management": (2.0, 3.0)}))

    run_main(monkeypatch, "--start", "2", "--end", "3", "--workers", "1")

    assert (tmp_path / "data" / "skill_scores_typesafe.json").read_text() == (
        "[\n"
        " {\n"
        '  "uri": "http://example.org/skill/c",\n'
        '  "title": "caf\\u00e9 management",\n'
        '  "automation_risk": 5.5,\n'
        '  "amplification_potential": 7.5,\n'
        '  "automation_probs": [\n'
        "   0.1,\n"
        "   0.2,\n"
        "   0.3,\n"
        "   0.25,\n"
        "   0.15\n"
        "  ],\n"
        '  "amplification_probs": [\n'
        "   0.05,\n"
        "   0.05,\n"
        "   0.4,\n"
        "   0.3,\n"
        "   0.2\n"
        "  ],\n"
        '  "automation_confidence": 0.875,\n'
        '  "amplification_confidence": 0.5,\n'
        '  "model": "jev-test",\n'
        '  "input_tokens": 11\n'
        " }\n"
        "]"
    )


def test_main_prints_the_run_summary(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "1", "--workers", "1", "--model", "jev-test")

    assert capsys.readouterr().out == (
        "Scoring 1 skills with jev-test\n"
        "Already scored: 0\n"
        "Remaining to score: 1\n"
        "\nDone in 0s. Total scored: 1, errors: 0.\n"
        "Input tokens this run: 11\n"
        "\nSummary across 1 skills:\n"
        "  Average automation risk:         5.50\n"
        "  Average amplification potential: 7.50\n"
    )


def test_main_stops_when_everything_is_already_scored(tmp_path, monkeypatch,
                                                      capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    done = [{"uri": "http://example.org/skill/a", "automation_risk": 1.0,
             "amplification_potential": 1.0, "input_tokens": 1}]
    (tmp_path / "data" / "skill_scores_typesafe.json").write_text(
        json.dumps(done))

    run_main(monkeypatch, "--end", "1")

    assert "Nothing to score. Use --force to re-score all." in (
        capsys.readouterr().out)


def test_main_force_rescores_everything(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    done = [{"uri": "http://example.org/skill/a", "automation_risk": 1.0,
             "amplification_potential": 1.0, "input_tokens": 1}]
    (tmp_path / "data" / "skill_scores_typesafe.json").write_text(
        json.dumps(done))
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "1", "--force", "--workers", "1")

    written = json.loads(
        (tmp_path / "data" / "skill_scores_typesafe.json").read_text())
    assert [entry["automation_risk"] for entry in written] == [5.5]


def test_main_samples_a_fixed_subset_for_a_given_seed(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--sample", "2", "--seed", "7", "--workers", "1")

    written = json.loads(
        (tmp_path / "data" / "skill_scores_typesafe.json").read_text())
    assert sorted(entry["uri"] for entry in written) == [
        "http://example.org/skill/a", "http://example.org/skill/c"]


def test_main_samples_no_more_than_the_slice_holds(tmp_path, monkeypatch,
                                                   capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "2", "--sample", "99", "--workers", "1")

    assert "Scoring 2 skills" in capsys.readouterr().out


def test_main_keeps_going_when_one_skill_is_refused(tmp_path, monkeypatch,
                                                    capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient(failures=["brew coffee"]))

    run_main(monkeypatch, "--end", "2", "--workers", "1")

    out = capsys.readouterr().out
    assert "ERROR 'brew coffee': refused brew coffee" in out
    assert "Total scored: 1, errors: 1." in out
    assert "failed: http://example.org/skill/a" in out
    written = json.loads(
        (tmp_path / "data" / "skill_scores_typesafe.json").read_text())
    assert [entry["uri"] for entry in written] == ["http://example.org/skill/b"]


def test_main_checkpoints_while_the_pool_runs(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    monkeypatch.setattr(typesafe, "CHECKPOINT_EVERY", 1)
    clock = iter(1000.0 + step for step in range(1000))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    install(monkeypatch, FakeClient())

    run_main(monkeypatch, "--end", "2", "--workers", "1")

    out = capsys.readouterr().out
    assert "  1/2 (" in out
    assert "  2/2 (" in out
    assert "22 tokens, errors 0)" in out


def test_main_writes_the_checkpoint_even_when_the_pool_raises(tmp_path,
                                                              monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    kept = [{"uri": "http://example.org/skill/a", "automation_risk": 1.0,
             "amplification_potential": 1.0, "input_tokens": 1}]
    (tmp_path / "data" / "skill_scores_typesafe.json").write_text(
        json.dumps(kept))
    install(monkeypatch, FakeClient(crashes=["types of sugars"]))

    with pytest.raises(RuntimeError, match="crashed on types of sugars"):
        run_main(monkeypatch, "--end", "2", "--workers", "1")

    written = json.loads(
        (tmp_path / "data" / "skill_scores_typesafe.json").read_text())
    assert [entry["uri"] for entry in written] == ["http://example.org/skill/a"]


def test_the_summary_divides_by_zero_when_every_skill_failed(tmp_path,
                                                             monkeypatch):
    # BUG: the closing summary averages over scored.values() without checking
    # that anything was scored, so a run in which every request fails and no
    # checkpoint exists dies with ZeroDivisionError after printing "Done in".
    # Pinned rather than fixed: changing it would change the script's output.
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    install(monkeypatch, FakeClient(failures=["brew coffee"]))

    with pytest.raises(ZeroDivisionError):
        run_main(monkeypatch, "--end", "1", "--workers", "1")


def test_the_script_runs_as_a_program(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["score_skills_typesafe.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("score_skills_typesafe", run_name="__main__")
    assert "usage: score_skills_typesafe.py" in capsys.readouterr().out
