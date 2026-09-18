"""Behaviour of the OpenRouter skill scorer: prompt, retries, resume, output."""

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

import score_skills

API_URL = "https://openrouter.ai/api/v1/chat/completions"
FIXTURES = Path(__file__).parent / "fixtures" / "scoring"

SKILLS = [
    {"uri": "u/a", "title": "brew coffee", "description": "Prepare coffee."},
    {"uri": "u/b", "title": "wash dishes", "description": ""},
]


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
    monkeypatch.setattr(sys, "argv", ["score_skills.py", *argv])
    score_skills.main()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No real key, no real sleeping and no real HTTP in any test here."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
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
    digest = hashlib.sha256(score_skills.SYSTEM_PROMPT.encode()).hexdigest()
    assert digest == (
        "2c7e4e19715a3465c84b0ecb28552b0da6a9d5c12f0e870998a0ee77292bca89"
    )


def test_constants_pin_the_cli_contract():
    assert score_skills.DEFAULT_MODEL == "google/gemini-3-flash-preview"
    assert score_skills.INPUT_FILE == "data/esco_skills.json"
    assert score_skills.OUTPUT_FILE == "data/skill_scores.json"


def test_build_batch_prompt_numbers_skills_and_keeps_descriptions():
    assert score_skills.build_batch_prompt(SKILLS) == (
        "Score each of the following skills/knowledge areas:\n\n"
        '1. "brew coffee" - Prepare coffee.\n'
        '2. "wash dishes"\n'
        "\nFor each skill, respond with a JSON array of objects with fields: "
        "title, automation_risk, amplification_potential, rationale."
    )


def test_build_batch_prompt_omits_a_missing_description():
    assert '1. "brew coffee"\n' in score_skills.build_batch_prompt(
        [{"uri": "u/a", "title": "brew coffee"}]
    )


def test_score_batch_sends_the_documented_request():
    client = fake_client(reply('[{"title": "brew coffee", '
                               '"automation_risk": 7, '
                               '"amplification_potential": 4}]'))
    score_skills.score_batch(client, SKILLS[:1], "test-model")

    url, kwargs = client.post.calls[0]
    assert url == API_URL
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["timeout"] == 120
    assert kwargs["json"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": score_skills.SYSTEM_PROMPT},
            {"role": "user",
             "content": score_skills.build_batch_prompt(SKILLS[:1])},
        ],
        "temperature": 0.2,
    }


def test_score_batch_strips_code_fences_and_trailing_commas():
    fenced = (
        "```json\n"
        '[{"title": "brew coffee", "automation_risk": 7, '
        '"amplification_potential": 4, "rationale": "x",},]\n'
        "```"
    )
    results = score_skills.score_batch(fake_client(reply(fenced)), SKILLS[:1], "m")
    assert results == [{"title": "brew coffee", "automation_risk": 7,
                        "amplification_potential": 4, "rationale": "x"}]


def test_score_batch_accepts_float_scores():
    body = '[{"title": "brew coffee", "automation_risk": 7.5, ' \
           '"amplification_potential": 4.5}]'
    results = score_skills.score_batch(fake_client(reply(body)), SKILLS[:1], "m")
    assert results[0]["automation_risk"] == 7.5


@pytest.mark.parametrize("status", [429, 500, 503])
def test_score_batch_backs_off_then_succeeds(status, sleeps, capsys):
    good = '[{"title": "brew coffee", "automation_risk": 7, ' \
           '"amplification_potential": 4}]'
    client = fake_client(reply("", status), reply(good))
    assert score_skills.score_batch(client, SKILLS[:1], "m")[0]["automation_risk"] == 7
    assert sleeps == [2.0]
    assert (f"Rate limited/server error ({status}), retrying in 2s "
            "(attempt 1/5)...") in capsys.readouterr().out


def test_score_batch_raises_a_status_it_does_not_retry(sleeps):
    client = fake_client(reply("", 400))
    with pytest.raises(httpx.HTTPStatusError):
        score_skills.score_batch(client, SKILLS[:1], "m")
    assert sleeps == []


def test_score_batch_gives_up_after_five_http_attempts(sleeps):
    client = fake_client(*[reply("", 429) for _ in range(5)])
    with pytest.raises(httpx.HTTPStatusError):
        score_skills.score_batch(client, SKILLS[:1], "m")
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_score_batch_retries_unparseable_json_then_gives_up(sleeps, capsys):
    client = fake_client(*[reply("not json") for _ in range(5)])
    with pytest.raises(json.JSONDecodeError):
        score_skills.score_batch(client, SKILLS[:1], "m")
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert capsys.readouterr().out.count("Parse error:") == 4


def test_score_batch_rejects_a_reply_that_is_not_an_array(sleeps):
    client = fake_client(*[reply('{"title": "brew coffee"}') for _ in range(5)])
    with pytest.raises(ValueError, match="Expected a JSON array, got dict"):
        score_skills.score_batch(client, SKILLS[:1], "m")


def test_score_batch_rejects_a_non_numeric_automation_risk(sleeps):
    body = '[{"title": "brew coffee", "automation_risk": "high", ' \
           '"amplification_potential": 4}]'
    client = fake_client(*[reply(body) for _ in range(5)])
    with pytest.raises(ValueError,
                       match="Invalid automation_risk for 'brew coffee'"):
        score_skills.score_batch(client, SKILLS[:1], "m")


def test_score_batch_rejects_a_missing_amplification_potential(sleeps):
    body = '[{"title": "brew coffee", "automation_risk": 7}]'
    client = fake_client(*[reply(body) for _ in range(5)])
    with pytest.raises(ValueError,
                       match="Invalid amplification_potential for 'brew coffee'"):
        score_skills.score_batch(client, SKILLS[:1], "m")


def test_score_batch_retries_a_reply_without_choices(sleeps):
    broken = httpx.Response(200, json={"error": "nope"},
                            request=httpx.Request("POST", API_URL))
    client = fake_client(*[broken for _ in range(5)])
    with pytest.raises(KeyError):
        score_skills.score_batch(client, SKILLS[:1], "m")
    assert sleeps == [2.0, 4.0, 8.0, 16.0]


def test_a_missing_api_key_is_reported_as_a_parse_error(monkeypatch, sleeps,
                                                        capsys):
    # BUG: score_batch reads os.environ["OPENROUTER_API_KEY"] inside the try
    # block, so a missing key raises KeyError where the parse-error handler
    # catches it. The run then prints "Parse error: 'OPENROUTER_API_KEY'" and
    # retries five times before failing. README's troubleshooting table
    # documents this exact symptom, so it is pinned here rather than fixed.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = fake_client()
    with pytest.raises(KeyError):
        score_skills.score_batch(client, SKILLS[:1], "m")
    assert client.post.calls == []
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert capsys.readouterr().out.count(
        "Parse error: 'OPENROUTER_API_KEY'") == 4


def test_main_writes_the_scores_file_byte_for_byte(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    body = json.dumps([
        {"title": "café management", "automation_risk": 3,
         "amplification_potential": 9, "rationale": "People work."},
    ])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--start", "2", "--end", "3")

    assert (tmp_path / "data" / "skill_scores.json").read_text() == (
        "[\n"
        "  {\n"
        '    "uri": "http://example.org/skill/c",\n'
        '    "title": "caf\\u00e9 management",\n'
        '    "automation_risk": 3,\n'
        '    "amplification_potential": 9,\n'
        '    "rationale": "People work."\n'
        "  }\n"
        "]"
    )


def test_main_reports_progress_and_the_score_distributions(tmp_path, monkeypatch,
                                                           capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    body = json.dumps([
        {"title": "brew coffee", "automation_risk": 7,
         "amplification_potential": 4, "rationale": "a"},
        {"title": "types of sugars", "automation_risk": 7,
         "amplification_potential": 8, "rationale": "b"},
    ])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--end", "2", "--model", "m")

    assert capsys.readouterr().out == (
        "Scoring 2 skills with m\n"
        "Batch size: 10\n"
        "Already scored: 0\n"
        "Remaining to score: 2\n"
        "\n  Batch 1/1 (2 skills): 'brew coffee' ... 'types of sugars' "
        "OK (2 scored, avg risk=7.0, avg amp=6.0)\n"
        "\nDone. Total scored: 2, errors: 0.\n"
        "\nSummary across 2 skills:\n"
        "  Average automation risk:        7.00\n"
        "  Average amplification potential: 6.00\n"
        "\nAutomation risk distribution:\n"
        "   7: ██ (2)\n"
        "\nAmplification potential distribution:\n"
        "   4: █ (1)\n"
        "   8: █ (1)\n"
    )


def test_main_resumes_from_the_checkpoint_and_force_ignores_it(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    done = [{"uri": "http://example.org/skill/a", "title": "brew coffee",
             "automation_risk": 1, "amplification_potential": 1,
             "rationale": "old"}]
    (tmp_path / "data" / "skill_scores.json").write_text(json.dumps(done))

    run_main(monkeypatch, "--end", "1")
    assert "Nothing to score. Use --force to re-score all." in capsys.readouterr().out

    body = json.dumps([{"title": "brew coffee", "automation_risk": 9,
                        "amplification_potential": 2, "rationale": "new"}])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))
    run_main(monkeypatch, "--end", "1", "--force")

    scores = json.loads((tmp_path / "data" / "skill_scores.json").read_text())
    assert scores == [{"uri": "http://example.org/skill/a",
                       "title": "brew coffee", "automation_risk": 9,
                       "amplification_potential": 2, "rationale": "new"}]


def test_main_keeps_already_scored_skills_and_appends_the_rest(tmp_path,
                                                              monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    done = [{"uri": "http://example.org/skill/a", "title": "brew coffee",
             "automation_risk": 1, "amplification_potential": 1,
             "rationale": "old"}]
    (tmp_path / "data" / "skill_scores.json").write_text(json.dumps(done))
    body = json.dumps([{"title": "types of sugars", "automation_risk": 8,
                        "amplification_potential": 8, "rationale": "new"}])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--end", "2")

    scores = json.loads((tmp_path / "data" / "skill_scores.json").read_text())
    assert [s["uri"] for s in scores] == ["http://example.org/skill/a",
                                          "http://example.org/skill/b"]
    assert scores[0]["rationale"] == "old"


def test_main_splits_into_batches_and_delays_between_them(tmp_path, monkeypatch,
                                                          sleeps, capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    replies = [
        reply(json.dumps([{"title": "brew coffee", "automation_risk": 7,
                           "amplification_potential": 4, "rationale": "a"}])),
        reply(json.dumps([{"title": "types of sugars", "automation_risk": 8,
                           "amplification_potential": 8, "rationale": "b"}])),
    ]
    monkeypatch.setattr(httpx.Client, "post", FakePoster(*replies))

    run_main(monkeypatch, "--end", "2", "--batch-size", "1", "--delay", "0.5")

    out = capsys.readouterr().out
    assert "Batch 1/2 (1 skills)" in out
    assert "Batch 2/2 (1 skills)" in out
    assert sleeps == [0.5]


def test_main_falls_back_to_the_title_when_the_reply_is_short(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    body = json.dumps([{"title": "Types Of Sugars", "automation_risk": 8,
                        "amplification_potential": 8, "rationale": "b"}])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--end", "2")

    out = capsys.readouterr().out
    scores = json.loads((tmp_path / "data" / "skill_scores.json").read_text())
    assert [s["uri"] for s in scores] == ["http://example.org/skill/a",
                                          "http://example.org/skill/b"]
    assert scores[0]["title"] == "brew coffee"
    assert "WARNING" not in out


def test_main_warns_when_a_skill_has_no_result_at_all(tmp_path, monkeypatch,
                                                      capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    body = json.dumps([{"title": "brew coffee", "automation_risk": 7,
                        "amplification_potential": 4, "rationale": "a"}])
    monkeypatch.setattr(httpx.Client, "post", FakePoster(reply(body)))

    run_main(monkeypatch, "--end", "2")

    out = capsys.readouterr().out
    assert "WARNING: No result for 'types of sugars'" in out
    assert "Done. Total scored: 1, errors: 1." in out
    assert "Failed URIs (1):\n  http://example.org/skill/b" in out


def test_main_records_a_whole_failed_batch_and_keeps_going(tmp_path, monkeypatch,
                                                           capsys):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    failing = [reply("", 400)]
    good = reply(json.dumps([{"title": "types of sugars", "automation_risk": 8,
                              "amplification_potential": 8, "rationale": "b"}]))
    monkeypatch.setattr(httpx.Client, "post", FakePoster(*failing, good))

    run_main(monkeypatch, "--end", "2", "--batch-size", "1")

    out = capsys.readouterr().out
    assert "ERROR: " in out
    assert "Done. Total scored: 1, errors: 1." in out
    scores = json.loads((tmp_path / "data" / "skill_scores.json").read_text())
    assert [s["uri"] for s in scores] == ["http://example.org/skill/b"]


def test_main_lists_at_most_ten_failed_uris(tmp_path, monkeypatch, capsys):
    workspace(tmp_path, monkeypatch)
    skills = [{"uri": f"u/{i}", "title": f"s{i}", "description": ""}
              for i in range(11)]
    (tmp_path / "data" / "esco_skills.json").write_text(json.dumps(skills))
    monkeypatch.setattr(httpx.Client, "post",
                        FakePoster(reply("[]"), reply("[]")))

    run_main(monkeypatch)

    out = capsys.readouterr().out
    assert out.count("WARNING: No result") == 11
    assert "  ... and 1 more" in out


def test_main_writes_a_checkpoint_after_every_batch(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, "esco_skills.json")
    written = []

    def post(self, url, **kwargs):
        checkpoint = tmp_path / "data" / "skill_scores.json"
        written.append(json.loads(checkpoint.read_text()) if checkpoint.exists()
                       else None)
        return reply(json.dumps([{"title": "x", "automation_risk": 5,
                                  "amplification_potential": 5,
                                  "rationale": ""}]))

    monkeypatch.setattr(httpx.Client, "post", post)
    run_main(monkeypatch, "--end", "2", "--batch-size", "1")

    assert written[0] is None
    assert len(written[1]) == 1


def test_the_script_runs_as_a_program(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["score_skills.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("score_skills", run_name="__main__")
    assert "usage: score_skills.py" in capsys.readouterr().out
