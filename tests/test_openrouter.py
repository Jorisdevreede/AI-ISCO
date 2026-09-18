"""Behaviour of the shared OpenRouter call: payload, clean-up, retries, loop."""

import time
from types import SimpleNamespace

import httpx
import pytest

from aiisco import openrouter
from aiisco.openrouter import (
    API_URL,
    BatchLoop,
    ChatRequest,
    backoff_for,
    build_payload,
    call_model,
    collect_results,
    fix_json,
    iter_batches,
    match_results,
    parse_array,
    pause_between,
    post_completion,
    print_batch_header,
    print_failures,
    print_histogram,
    run_batches,
    strip_code_fences,
    tally,
    wait_or_raise_parse_error,
    wait_or_raise_status,
    warn_missing,
)

REQUEST = ChatRequest(system_prompt="be terse", model="test-model", timeout=42)
ITEMS = [{"uri": "u/a", "title": "brew coffee"},
         {"uri": "u/b", "title": "wash dishes"}]


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
    return SimpleNamespace(post=FakePoster(*responses))


def status_error(status):
    """An HTTPStatusError carrying the given status."""
    response = httpx.Response(status, request=httpx.Request("POST", API_URL))
    return httpx.HTTPStatusError("boom", request=response.request,
                                 response=response)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """A synthetic key and no real waiting."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


@pytest.fixture
def sleeps(monkeypatch):
    """Record every backoff instead of waiting for it."""
    recorded = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def test_build_payload_leaves_out_an_unset_token_limit():
    assert build_payload(REQUEST, "hello") == {
        "model": "test-model",
        "messages": [{"role": "system", "content": "be terse"},
                     {"role": "user", "content": "hello"}],
        "temperature": 0.2,
    }


def test_build_payload_adds_the_token_limit_when_one_is_set():
    request = ChatRequest(system_prompt="be terse", model="m", timeout=1,
                          max_tokens=4096)
    assert build_payload(request, "hello")["max_tokens"] == 4096


def test_post_completion_returns_the_assistant_message():
    client = fake_client(reply("the answer"))
    assert post_completion(client, REQUEST, "hello") == "the answer"
    url, kwargs = client.post.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["timeout"] == 42


def test_post_completion_raises_on_an_error_status():
    with pytest.raises(httpx.HTTPStatusError):
        post_completion(fake_client(reply("", 503)), REQUEST, "hello")


def test_post_completion_needs_the_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
        post_completion(fake_client(reply("x")), REQUEST, "hello")


@pytest.mark.parametrize(("raw", "expected"), [
    ("[1]", "[1]"),
    ("  [1]  ", "[1]"),
    ("```json\n[1]\n```", "[1]"),
    ("```\n[1]\n```", "[1]"),
    ("```json\n[1]", "[1]"),
])
def test_strip_code_fences_removes_only_a_leading_fence(raw, expected):
    assert strip_code_fences(raw) == expected


def test_a_single_line_fenced_reply_breaks_the_fence_stripper():
    # BUG: strip_code_fences splits on the first newline without checking that
    # there is one, so a reply fenced on a single line raises IndexError.
    # IndexError is not in call_model's retry list, so it escapes to
    # run_batches and fails the whole batch instead of being retried. Pinned
    # rather than fixed: recovering here would change which batches succeed.
    with pytest.raises(IndexError):
        strip_code_fences("``` [1] ```")


@pytest.mark.parametrize(("raw", "expected"), [
    ('[1,]', "[1]"),
    ('{"a": 1,}', '{"a": 1}'),
    ('[{"a": 1,},]', '[{"a": 1}]'),
    ('[1, 2]', "[1, 2]"),
])
def test_fix_json_drops_trailing_commas(raw, expected):
    assert fix_json(raw) == expected


def test_parse_array_accepts_a_fenced_array_with_a_trailing_comma():
    assert parse_array('```json\n[{"a": 1},]\n```') == [{"a": 1}]


def test_parse_array_rejects_anything_that_is_not_an_array():
    with pytest.raises(ValueError, match="Expected a JSON array, got dict"):
        parse_array('{"a": 1}')


def test_backoff_for_doubles_every_attempt():
    assert [backoff_for(n) for n in range(5)] == [2.0, 4.0, 8.0, 16.0, 32.0]


@pytest.mark.parametrize("status", [429, 500, 502])
def test_wait_or_raise_status_backs_off_from_transient_failures(status, sleeps):
    wait_or_raise_status(status_error(status), (429,), 1)
    assert sleeps == [4.0]


def test_wait_or_raise_status_honours_an_extra_retry_status(sleeps):
    wait_or_raise_status(status_error(402), (402, 429), 0)
    assert sleeps == [2.0]


def test_wait_or_raise_status_reraises_a_permanent_failure(sleeps):
    error = status_error(404)
    with pytest.raises(httpx.HTTPStatusError):
        wait_or_raise_status(error, (429,), 0)
    assert sleeps == []


def test_wait_or_raise_parse_error_backs_off_until_the_last_attempt(sleeps):
    wait_or_raise_parse_error(ValueError("bad"), 3)
    assert sleeps == [16.0]
    with pytest.raises(ValueError, match="bad"):
        wait_or_raise_parse_error(ValueError("bad"), 4)


def test_call_model_validates_every_item():
    seen = []
    results = call_model(fake_client(reply('[{"a": 1}, {"a": 2}]')), REQUEST,
                         "hello", seen.append)
    assert results == [{"a": 1}, {"a": 2}]
    assert seen == results


def test_call_model_returns_the_first_usable_reply(sleeps):
    client = fake_client(reply("", 429), reply("nonsense"), reply("[1]"))
    assert call_model(client, REQUEST, "hello", lambda item: None) == [1]
    assert sleeps == [2.0, 4.0]


def test_call_model_raises_the_last_error_when_http_retries_run_out(sleeps):
    client = fake_client(*[reply("", 500) for _ in range(5)])
    with pytest.raises(httpx.HTTPStatusError):
        call_model(client, REQUEST, "hello", lambda item: None)
    assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert len(client.post.calls) == 5


@pytest.mark.parametrize(("size", "expected"), [
    (1, [["a"], ["b"], ["c"]]),
    (2, [["a", "b"], ["c"]]),
    (3, [["a", "b", "c"]]),
    (9, [["a", "b", "c"]]),
])
def test_iter_batches_splits_into_consecutive_runs(size, expected):
    assert iter_batches(["a", "b", "c"], size) == expected


def test_iter_batches_of_nothing_is_nothing():
    assert iter_batches([], 5) == []


def test_match_results_pairs_by_position():
    results = [{"title": "A"}, {"title": "B"}]
    assert match_results(ITEMS, results) == [(ITEMS[0], results[0]),
                                             (ITEMS[1], results[1])]


def test_match_results_falls_back_to_a_case_insensitive_title():
    results = [{"title": "WASH DISHES"}]
    assert match_results(ITEMS, results) == [(ITEMS[0], results[0]),
                                             (ITEMS[1], results[0])]


def test_match_results_reports_an_item_the_model_skipped():
    results = [{"title": "brew coffee"}]
    assert match_results(ITEMS, results)[1] == (ITEMS[1], None)


def test_collect_results_records_the_items_left_out(capsys):
    state = SimpleNamespace(added=[])
    state.add = lambda item, result: state.added.append((item, result))
    errors = []

    collect_results(state, ITEMS, [{"title": "brew coffee"}], errors)

    assert [item["uri"] for item, _ in state.added] == ["u/a"]
    assert errors == ["u/b"]
    assert "WARNING: No result for 'wash dishes'" in capsys.readouterr().out


def test_print_batch_header_names_the_first_and_last_item(capsys):
    print_batch_header(0, 3, ITEMS, "skills")
    assert capsys.readouterr().out == (
        "\n  Batch 1/3 (2 skills): 'brew coffee' ... 'wash dishes' ")


def test_warn_missing_names_the_item(capsys):
    warn_missing(ITEMS[0])
    assert capsys.readouterr().out == "\n    WARNING: No result for 'brew coffee'\n"


def test_pause_between_skips_the_wait_after_the_last_batch(sleeps):
    pause_between(0, 2, 1.5)
    pause_between(1, 2, 1.5)
    assert sleeps == [1.5]


def test_run_batches_checkpoints_after_every_batch(capsys):
    saves = []
    loop = BatchLoop(batches=[[ITEMS[0]], [ITEMS[1]]], noun="skills",
                     handle=lambda batch, errors: None,
                     checkpoint=lambda: saves.append(len(saves)), delay=0.0)
    assert run_batches(loop) == []
    assert saves == [0, 1]
    assert capsys.readouterr().out.count("Batch ") == 2


def test_run_batches_blames_a_failing_batch_for_all_of_its_items(capsys):
    def handle(batch, errors):
        raise RuntimeError("model is down")

    loop = BatchLoop(batches=[ITEMS], noun="skills", handle=handle,
                     checkpoint=lambda: None)
    assert run_batches(loop) == ["u/a", "u/b"]
    assert "ERROR: model is down" in capsys.readouterr().out


def test_run_batches_keeps_the_warnings_a_failing_batch_already_recorded():
    def handle(batch, errors):
        errors.append("u/a")
        raise RuntimeError("halfway")

    loop = BatchLoop(batches=[ITEMS], noun="skills", handle=handle,
                     checkpoint=lambda: None)
    assert run_batches(loop) == ["u/a", "u/a", "u/b"]


def test_print_failures_says_nothing_when_nothing_failed(capsys):
    print_failures([])
    assert capsys.readouterr().out == ""


def test_print_failures_lists_every_uri_up_to_ten(capsys):
    print_failures(["u/a", "u/b"])
    assert capsys.readouterr().out == "Failed URIs (2):\n  u/a\n  u/b\n"


def test_print_failures_truncates_a_long_list(capsys):
    print_failures([f"u/{i}" for i in range(13)])
    out = capsys.readouterr().out
    assert out.startswith("Failed URIs (13):")
    assert out.count("\n  u/") == 10
    assert out.endswith("  ... and 3 more\n")


def test_tally_counts_repeats():
    assert tally(["a", "b", "a"]) == {"a": 2, "b": 1}


def test_print_histogram_draws_one_bar_per_bucket_in_order(capsys):
    print_histogram("Scores:", {3: 2, 1: 1})
    assert capsys.readouterr().out == (
        "\nScores:\n  1: █ (1)\n  3: ██ (2)\n")


def test_print_histogram_takes_a_label_formatter(capsys):
    print_histogram("Scores:", {1: 1}, lambda bucket: f"{bucket:>3}")
    assert capsys.readouterr().out == "\nScores:\n    1: █ (1)\n"


def test_the_retry_budget_is_five_attempts():
    assert openrouter.MAX_RETRIES == 5
    assert openrouter.INITIAL_BACKOFF == 2.0
