"""Behaviour of the narrative generator: aggregation, prompt, retries, output."""

import hashlib
import json
import runpy
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import generate_narratives

API_URL = "https://openrouter.ai/api/v1/chat/completions"
FIXTURES = Path(__file__).parent / "fixtures" / "scoring"

CONTEXT = {
    "uri": "occ/1",
    "title": "barista",
    "isco_code": "5132",
    "quadrant": "TRANSFORM",
    "auto_avg": 7.0,
    "amp_avg": 6.6,
    "essential": [
        {"title": "brew coffee", "automation_risk": 7.0,
         "amplification_potential": 4.0, "rationale": "Machines pour."},
        {"title": "types of sugars", "automation_risk": 9.0,
         "amplification_potential": 8.0, "rationale": "A lookup."},
    ],
    "optional": [],
}

NARRATIVE = {
    "title": "barista",
    "evolution_story": "Your role changes.",
    "time_savings_pct": 40,
    "automated_tasks": ["a"],
    "amplified_capabilities": ["b"],
    "ai_tools_applicable": ["c"],
    "rebalanced_week": {"before": {"admin": 100},
                        "after": {"admin": 60, "new_ai_augmented": 40}},
    "timeline": "3-5 years",
    "advice": "Learn.",
}


def narrative(title, savings=40):
    """A valid narrative for one occupation."""
    return {**NARRATIVE, "title": title, "time_savings_pct": savings}


def reply(content, status=200):
    """An httpx response whose body carries one assistant message."""
    body = {"choices": [{"message": {"content": content}}]}
    return httpx.Response(status, json=body, request=httpx.Request("POST", API_URL))


class FakePoster:
    """Stands in for httpx.Client.post and replays queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def fake_client(*responses):
    """A client-shaped object whose post replays the given responses."""
    return SimpleNamespace(post=FakePoster(*responses), close=lambda: None)


def workspace(tmp_path, monkeypatch, *names):
    """Copy the named scoring fixtures into tmp_path/data and work there."""
    (tmp_path / "data").mkdir(exist_ok=True)
    for name in names:
        shutil.copy(FIXTURES / name, tmp_path / "data" / name)
    monkeypatch.chdir(tmp_path)


def run_main(monkeypatch, *argv):
    """Run main() with the given command line."""
    monkeypatch.setattr(sys, "argv", ["generate_narratives.py", *argv])
    generate_narratives.main()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No real key, no real sleeping, no real HTTP, no leaked output path."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(generate_narratives, "OUTPUT_FILE",
                        generate_narratives.OUTPUT_FILE)
    monkeypatch.setattr(
        httpx.Client, "post",
        lambda *a, **k: pytest.fail("a test reached the network"),
    )


@pytest.fixture
def sleeps(monkeypatch):
    """Record every backoff instead of waiting for it."""
    recorded = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def test_system_prompt_is_unchanged():
    digest = hashlib.sha256(generate_narratives.SYSTEM_PROMPT.encode()).hexdigest()
    assert digest == (
        "31064957ded0cd8ba2e99dac43c833ffd7fc67b60f2b61d4ae0c69fe57e73d1c"
    )


def test_constants_pin_the_cli_contract():
    assert generate_narratives.DEFAULT_MODEL == "google/gemini-3-flash-preview"
    assert generate_narratives.OCCUPATIONS_FILE == "data/esco_occupations.json"
    assert generate_narratives.SKILL_SCORES_FILE == "data/skill_scores.json"
    assert generate_narratives.OUTPUT_FILE == "data/occupation_narratives.json"


@pytest.mark.parametrize(("auto", "amp", "quadrant"), [
    (6, 6, "TRANSFORM"),
    (6, 5.9, "SHRINK"),
    (5.9, 6, "EVOLVE"),
    (5.9, 5.9, "STABLE"),
])
def test_assign_quadrant_splits_on_the_threshold(auto, amp, quadrant):
    assert generate_narratives.assign_quadrant(auto, amp) == quadrant


def test_aggregate_occupation_scores_weights_essential_skills_double():
    occupation = {
        "essential_skills": [{"uri": "s/a", "title": "brew coffee"}],
        "optional_skills": [{"uri": "s/c", "title": "manage"}],
    }
    scores = {
        "s/a": {"automation_risk": 7.0, "amplification_potential": 4.0,
                "rationale": "r"},
        "s/c": {"automation_risk": 1.0, "amplification_potential": 10.0,
                "rationale": ""},
    }
    auto, amp, essential, optional = generate_narratives.aggregate_occupation_scores(
        occupation, scores)
    assert (auto, amp) == (5.0, 6.0)
    assert essential == [{"title": "brew coffee", "automation_risk": 7.0,
                          "amplification_potential": 4.0, "rationale": "r"}]
    assert optional[0]["title"] == "manage"


def test_aggregate_occupation_scores_returns_none_without_any_scored_skill():
    occupation = {"essential_skills": [{"uri": "s/x", "title": "x"}]}
    assert generate_narratives.aggregate_occupation_scores(occupation, {}) is None


def test_build_batch_prompt_lays_out_one_occupation():
    assert generate_narratives.build_batch_prompt([CONTEXT]) == (
        "Generate an AI evolution narrative for each of the following "
        "occupations:\n"
        "\n--- Occupation 1 ---\n"
        "Title: barista\n"
        "ISCO Code: 5132\n"
        "Quadrant: TRANSFORM\n"
        "Aggregated Scores: automation_risk=7.0, amplification_potential=6.6\n"
        "\nScored Essential Skills (2 total):\n"
        '  - "brew coffee" (auto=7.0, amp=4.0) — Machines pour.\n'
        '  - "types of sugars" (auto=9.0, amp=8.0) — A lookup.\n'
        "\nTop 5 Most Automatable Essential Skills:\n"
        '  - "types of sugars" (auto=9.0)\n'
        '  - "brew coffee" (auto=7.0)\n'
        "\nTop 5 Most AI-Amplifiable Essential Skills:\n"
        '  - "types of sugars" (amp=8.0)\n'
        '  - "brew coffee" (amp=4.0)\n'
        "\nFor each occupation, respond with a JSON array of objects with "
        "fields: title, evolution_story, time_savings_pct, automated_tasks, "
        "amplified_capabilities, ai_tools_applicable, rebalanced_week, "
        "timeline, advice."
    )


def test_build_batch_prompt_keeps_only_the_five_most_automatable():
    essential = [{"title": f"s{i}", "automation_risk": float(i),
                  "amplification_potential": 1.0, "rationale": ""}
                 for i in range(7)]
    prompt = generate_narratives.build_batch_prompt(
        [{**CONTEXT, "essential": essential}])
    automatable = prompt.split("Top 5 Most Automatable Essential Skills:\n")[1]
    assert automatable.split("\nTop 5")[0].count("  - ") == 5


def test_validate_result_accepts_a_complete_narrative():
    assert generate_narratives.validate_result(NARRATIVE) == []


@pytest.mark.parametrize(("change", "message"), [
    ({"title": ""}, "missing or empty 'title'"),
    ({"title": 1}, "missing or empty 'title'"),
    ({"evolution_story": ""}, "missing or empty 'evolution_story'"),
    ({"time_savings_pct": "40"},
     "'time_savings_pct' must be a number 0-100, got '40'"),
    ({"time_savings_pct": -1},
     "'time_savings_pct' must be a number 0-100, got -1"),
    ({"time_savings_pct": 101},
     "'time_savings_pct' must be a number 0-100, got 101"),
    ({"automated_tasks": "a"}, "'automated_tasks' must be an array"),
    ({"amplified_capabilities": None},
     "'amplified_capabilities' must be an array"),
    ({"ai_tools_applicable": {}}, "'ai_tools_applicable' must be an array"),
    ({"rebalanced_week": []}, "'rebalanced_week' must be an object"),
    ({"rebalanced_week": {"after": {}}}, "'rebalanced_week' missing 'before'"),
    ({"rebalanced_week": {"before": {}}}, "'rebalanced_week' missing 'after'"),
])
def test_validate_result_names_each_problem(change, message):
    assert generate_narratives.validate_result({**NARRATIVE, **change}) == [message]


def test_validate_result_reports_every_missing_field_at_once():
    assert len(generate_narratives.validate_result({})) == 7


def test_generate_batch_sends_the_documented_request():
    client = fake_client(reply(json.dumps([narrative("barista")])))
    generate_narratives.generate_batch(client, [CONTEXT], "test-model")

    url, kwargs = client.post.calls[0]
    assert url == API_URL
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["timeout"] == 180
    assert kwargs["json"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": generate_narratives.SYSTEM_PROMPT},
            {"role": "user",
             "content": generate_narratives.build_batch_prompt([CONTEXT])},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }


def test_generate_batch_strips_code_fences_and_trailing_commas():
    body = "```json\n" + json.dumps([narrative("barista")])[:-1] + ",]\n```"
    results = generate_narratives.generate_batch(fake_client(reply(body)),
                                                 [CONTEXT], "m")
    assert results[0]["title"] == "barista"


@pytest.mark.parametrize("status", [402, 429, 500])
def test_generate_batch_backs_off_then_succeeds(status, sleeps, capsys):
    good = reply(json.dumps([narrative("barista")]))
    client = fake_client(reply("", status), good)
    assert generate_narratives.generate_batch(client, [CONTEXT], "m")
    assert sleeps == [2.0]
    assert f"Rate limited/server error ({status})" in capsys.readouterr().out


def test_generate_batch_raises_a_status_it_does_not_retry(sleeps):
    with pytest.raises(httpx.HTTPStatusError):
        generate_narratives.generate_batch(fake_client(reply("", 404)),
                                           [CONTEXT], "m")
    assert sleeps == []


def test_generate_batch_gives_up_after_five_http_attempts(sleeps):
    client = fake_client(*[reply("", 429) for _ in range(5)])
    with pytest.raises(httpx.HTTPStatusError):
        generate_narratives.generate_batch(client, [CONTEXT], "m")
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_generate_batch_retries_unparseable_json_then_gives_up(sleeps, capsys):
    client = fake_client(*[reply("nonsense") for _ in range(5)])
    with pytest.raises(json.JSONDecodeError):
        generate_narratives.generate_batch(client, [CONTEXT], "m")
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert capsys.readouterr().out.count("Parse error:") == 4


def test_generate_batch_rejects_a_reply_that_is_not_an_array(sleeps):
    client = fake_client(*[reply('{"title": "barista"}') for _ in range(5)])
    with pytest.raises(ValueError, match="Expected a JSON array, got dict"):
        generate_narratives.generate_batch(client, [CONTEXT], "m")


def test_generate_batch_rejects_an_invalid_narrative(sleeps):
    broken = json.dumps([{**NARRATIVE, "evolution_story": ""}])
    client = fake_client(*[reply(broken) for _ in range(5)])
    with pytest.raises(ValueError, match="Validation failed for 'barista'"):
        generate_narratives.generate_batch(client, [CONTEXT], "m")


def test_a_missing_api_key_is_reported_as_a_parse_error(monkeypatch, sleeps,
                                                        capsys):
    # BUG: generate_batch reads os.environ["OPENROUTER_API_KEY"] inside the try
    # block, so a missing key raises KeyError where the parse-error handler
    # catches it, printing "Parse error: 'OPENROUTER_API_KEY'" five times.
    # Same defect as score_skills.py; pinned rather than fixed.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = fake_client()
    with pytest.raises(KeyError):
        generate_narratives.generate_batch(client, [CONTEXT], "m")
    assert client.post.calls == []
    assert capsys.readouterr().out.count(
        "Parse error: 'OPENROUTER_API_KEY'") == 4


def test_main_writes_the_narratives_file_byte_for_byte(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    body = json.dumps([narrative("café manager", savings=25)])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--start", "1", "--end", "2")

    assert (tmp_path / "data" / "occupation_narratives.json").read_text() == (
        "[\n"
        "  {\n"
        '    "uri": "http://example.org/occupation/2",\n'
        '    "title": "caf\\u00e9 manager",\n'
        '    "evolution_story": "Your role changes.",\n'
        '    "time_savings_pct": 25,\n'
        '    "automated_tasks": [\n'
        '      "a"\n'
        "    ],\n"
        '    "amplified_capabilities": [\n'
        '      "b"\n'
        "    ],\n"
        '    "ai_tools_applicable": [\n'
        '      "c"\n'
        "    ],\n"
        '    "rebalanced_week": {\n'
        '      "before": {\n'
        '        "admin": 100\n'
        "      },\n"
        '      "after": {\n'
        '        "admin": 60,\n'
        '        "new_ai_augmented": 40\n'
        "      }\n"
        "    },\n"
        '    "timeline": "3-5 years",\n'
        '    "advice": "Learn."\n'
        "  }\n"
        "]"
    )


def test_main_reports_quadrants_and_time_savings(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    body = json.dumps([narrative("barista", 40),
                       narrative("café manager", 25)])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--end", "2", "--model", "m")

    assert capsys.readouterr().out == (
        "Generating narratives for 2 occupations with m\n"
        "Batch size: 5\n"
        "Already narrated: 0\n"
        "Remaining to narrate: 2\n"
        "\n  Batch 1/1 (2 occupations): 'barista' ... 'café manager' "
        "OK (2 narrated, avg time_savings=32.5%)\n"
        "\nDone. Total narrated: 2, errors: 0.\n"
        "\nSummary across 2 occupations:\n"
        "  Average time_savings_pct: 32.5%\n"
        "\nTime savings distribution:\n"
        "    20-29%: █ (1)\n"
        "    40-49%: █ (1)\n"
        "\nQuadrant breakdown:\n"
        "  TRANSFORM   :    1 ( 50.0%)\n"
        "  SHRINK      :    0 (  0.0%)\n"
        "  EVOLVE      :    1 ( 50.0%)\n"
        "  STABLE      :    0 (  0.0%)\n"
        "\nAverage time savings by quadrant:\n"
        "  TRANSFORM   :  40.0% (n=1)\n"
        "  SHRINK      : n/a\n"
        "  EVOLVE      :  25.0% (n=1)\n"
        "  STABLE      : n/a\n"
    )


def test_main_skips_occupations_without_any_scored_skill(tmp_path, monkeypatch,
                                                         capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    body = json.dumps([narrative("barista"), narrative("café manager")])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch)

    assert "Generating narratives for 2 occupations" in capsys.readouterr().out


def test_main_resumes_from_the_checkpoint_and_force_ignores_it(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    done = [{"uri": "http://example.org/occupation/1", "title": "barista",
             "time_savings_pct": 10}]
    (tmp_path / "data" / "occupation_narratives.json").write_text(json.dumps(done))

    run_main(monkeypatch, "--end", "1")
    assert "Nothing to narrate. Use --force to re-generate all." in (
        capsys.readouterr().out)

    body = json.dumps([narrative("barista", 55)])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))
    run_main(monkeypatch, "--end", "1", "--force")

    written = json.loads(
        (tmp_path / "data" / "occupation_narratives.json").read_text())
    assert [n["time_savings_pct"] for n in written] == [55]


def test_main_writes_a_shard_to_the_output_flag(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    body = json.dumps([narrative("café manager")])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--start", "1", "--end", "2",
             "--output", "data/shard_2.json")

    assert not (tmp_path / "data" / "occupation_narratives.json").exists()
    shard = json.loads((tmp_path / "data" / "shard_2.json").read_text())
    assert [n["uri"] for n in shard] == ["http://example.org/occupation/2"]


def test_main_batches_and_delays_between_batches(tmp_path, monkeypatch, sleeps,
                                                 capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    replies = [reply(json.dumps([narrative("barista")])),
               reply(json.dumps([narrative("café manager")]))]
    monkeypatch.setattr(httpx.Client, "post", FakePoster(*replies))

    run_main(monkeypatch, "--batch-size", "1", "--delay", "3.0")

    out = capsys.readouterr().out
    assert "Batch 1/2 (1 occupations)" in out
    assert "Batch 2/2 (1 occupations)" in out
    assert sleeps == [3.0]


def test_main_warns_when_an_occupation_has_no_narrative(tmp_path, monkeypatch,
                                                        capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    body = json.dumps([narrative("barista")])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch)

    out = capsys.readouterr().out
    assert "WARNING: No result for 'café manager'" in out
    assert "Done. Total narrated: 1, errors: 1." in out
    assert "Failed URIs (1):\n  http://example.org/occupation/2" in out


def test_main_records_a_whole_failed_batch_and_keeps_going(tmp_path, monkeypatch,
                                                           capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    good = reply(json.dumps([narrative("café manager")]))
    monkeypatch.setattr(httpx.Client, "post",
                        FakePoster(reply("", 404), good))

    run_main(monkeypatch, "--batch-size", "1")

    out = capsys.readouterr().out
    assert "ERROR: " in out
    assert "Done. Total narrated: 1, errors: 1." in out


def test_main_writes_a_checkpoint_after_every_batch(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    seen = []

    def post(self, url, **kwargs):
        checkpoint = tmp_path / "data" / "occupation_narratives.json"
        seen.append(checkpoint.exists())
        return reply(json.dumps([narrative("barista")]))

    monkeypatch.setattr(httpx.Client, "post", post)
    run_main(monkeypatch, "--batch-size", "1")

    assert seen == [False, True]


def test_main_prints_no_statistics_when_nothing_was_narrated(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply("[]")))

    run_main(monkeypatch, "--end", "1")

    out = capsys.readouterr().out
    assert "Done. Total narrated: 0, errors: 1." in out
    assert "Summary across" not in out


def test_main_ignores_checkpointed_occupations_outside_the_slice(tmp_path,
                                                                 monkeypatch,
                                                                 capsys):
    workspace(tmp_path, monkeypatch, "esco_occupations.json", "skill_scores.json")
    outside = [{"uri": "http://example.org/occupation/1", "title": "barista",
                "time_savings_pct": 90}]
    (tmp_path / "data" / "occupation_narratives.json").write_text(
        json.dumps(outside))
    body = json.dumps([narrative("café manager", 25)])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--start", "1", "--end", "2")

    out = capsys.readouterr().out
    assert "  Average time_savings_pct: 57.5%" in out
    assert "  EVOLVE      :  25.0% (n=1)" in out
    assert "  TRANSFORM   : n/a" in out


def test_the_script_runs_as_a_program(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["generate_narratives.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("generate_narratives", run_name="__main__")
    assert "usage: generate_narratives.py" in capsys.readouterr().out
