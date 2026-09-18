"""The plumbing both TypeSafe scorers share: throttling, selection, the pool.

Nothing here reaches the network or a keychain: the pool is driven by a function
that answers from a script, and the clock is frozen where timing is asserted.
"""

import argparse
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from typesafe_sdk import TypeSafeError

from aiisco import systemone


def items(count, first=0):
    """Pool items shaped the way both scorers feed them in."""
    return [{"uri": f"u{i}", "title": f"t{i}"} for i in range(first, first + count)]


def config(score_one, workers=1, checkpoint_every=1000, saves=None):
    """A pool configuration whose save snapshots what it was handed.

    The pool passes its live dict, so a snapshot is what makes the saves
    distinguishable from one another.
    """
    def save(scored):
        if saves is not None:
            saves.append(dict(scored))

    return systemone.PoolConfig(score_one=score_one, save=save, workers=workers,
                                rps=1000.0, checkpoint_every=checkpoint_every)


def entry(item):
    """The minimum a scored entry needs for the pool to fold it in."""
    return {"uri": item["uri"], "input_tokens": 3}


# --- the rate limiter -------------------------------------------------------

def test_rate_limiter_spaces_starts_and_skips_the_wait_when_it_is_late(monkeypatch):
    clock = [100.0]
    recorded = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", recorded.append)

    limiter = systemone.RateLimiter(4.0)
    limiter.wait()
    limiter.wait()
    assert recorded == [0.25]

    clock[0] = 200.0
    limiter.wait()
    assert recorded == [0.25]
    assert limiter.next_at == 200.25


# --- the 1-10 display scale -------------------------------------------------

@pytest.mark.parametrize(("level", "expected"), [
    (0, 1.5), (1, 3.5), (2, 5.5), (3, 7.5), (4, 9.5), (1.234, 3.97),
])
def test_to_ten_scale_maps_a_band_position_onto_the_rubric_scale(level, expected):
    assert systemone.to_ten_scale(level) == expected


# --- selecting what to score ------------------------------------------------

def args(start=0, end=None, sample=None, seed=7):
    return SimpleNamespace(start=start, end=end, sample=sample, seed=seed)


def test_select_items_takes_the_requested_slice():
    assert [i["uri"] for i in systemone.select_items(items(5), args(start=1, end=3))] == \
        ["u1", "u2"]


def test_select_items_samples_the_same_subset_for_a_given_seed():
    first = systemone.select_items(items(20), args(sample=4, seed=3))
    second = systemone.select_items(items(20), args(sample=4, seed=3))
    assert [i["uri"] for i in first] == [i["uri"] for i in second]
    assert len(first) == 4


def test_select_items_samples_no_more_than_the_slice_holds():
    assert len(systemone.select_items(items(3), args(sample=99))) == 3


# --- the pool ---------------------------------------------------------------

def test_run_pool_scores_every_item_and_counts_the_tokens():
    run = systemone.run_pool(items(5), {}, config(lambda limiter, item: entry(item)))

    assert sorted(run.scored) == ["u0", "u1", "u2", "u3", "u4"]
    assert (run.done, run.tokens, run.errors, run.stopped) == (5, 15, [], "")


def test_run_pool_keeps_what_was_already_scored():
    already = {"u9": {"uri": "u9", "input_tokens": 1}}
    run = systemone.run_pool(items(1), already, config(lambda limiter, item: entry(item)))
    assert sorted(run.scored) == ["u0", "u9"]


def test_run_pool_saves_at_every_checkpoint_and_on_the_way_out(capsys):
    saves = []
    systemone.run_pool(items(4), {}, config(lambda limiter, item: entry(item),
                                            checkpoint_every=2, saves=saves))

    assert [len(scored) for scored in saves] == [2, 4, 4]
    assert "  2/4 (" in capsys.readouterr().out


def test_run_pool_saves_what_it_had_when_a_request_crashes():
    saves = []

    def crash(limiter, item):
        if item["uri"] == "u1":
            raise RuntimeError("boom")
        return entry(item)

    with pytest.raises(RuntimeError, match="boom"):
        systemone.run_pool(items(2), {}, config(crash, saves=saves))

    assert list(saves[-1]) == ["u0"]


def test_a_refused_request_is_recorded_and_the_run_carries_on(capsys):
    def refuse(limiter, item):
        if item["uri"] == "u0":
            raise TypeSafeError("refused")
        return entry(item)

    run = systemone.run_pool(items(3), {}, config(refuse))

    assert run.errors == ["u0"]
    assert sorted(run.scored) == ["u1", "u2"]
    assert "ERROR 't0': refused" in capsys.readouterr().out


def test_the_pool_stops_itself_after_twenty_failures_in_a_row():
    def refuse(limiter, item):
        raise TypeSafeError("refused")

    run = systemone.run_pool(items(60), {}, config(refuse))

    assert run.stopped == "20 requests in a row failed"
    assert len(run.errors) == systemone.MAX_CONSECUTIVE_FAILURES
    assert run.done == systemone.MAX_CONSECUTIVE_FAILURES


def settled(outcome):
    """A finished future holding a result, or the error the request raised."""
    future = Future()
    if isinstance(outcome, Exception):
        future.set_exception(outcome)
    else:
        future.set_result(outcome)
    return future


def test_one_success_resets_the_consecutive_failure_count(capsys):
    """Collected directly: a pool of one still finishes futures in whatever order."""
    run = systemone.PoolRun(scored={}, total=3, started=0.0)
    item = items(1)[0]

    systemone.collect(run, settled(TypeSafeError("refused")), item)
    systemone.collect(run, settled(TypeSafeError("refused")), item)
    assert run.consecutive == 2

    systemone.collect(run, settled(entry(item)), item)
    assert run.consecutive == 0
    assert (run.tokens, len(run.errors)) == (3, 2)
    assert "ERROR 't0': refused" in capsys.readouterr().out


# --- the shared command line ------------------------------------------------

def test_add_run_arguments_gives_both_scorers_the_same_flags():
    parser = argparse.ArgumentParser()
    systemone.add_run_arguments(parser, workers=6, rps=10.0)

    parsed = parser.parse_args([])
    assert (parsed.start, parsed.end, parsed.sample, parsed.seed) == (0, None, None, 7)
    assert (parsed.workers, parsed.rps, parsed.force) == (6, 10.0, False)


def test_add_run_arguments_takes_the_defaults_it_is_given():
    parser = argparse.ArgumentParser()
    systemone.add_run_arguments(parser, workers=2, rps=15.0)
    parsed = parser.parse_args(["--force", "--sample", "5"])
    assert (parsed.workers, parsed.rps, parsed.sample, parsed.force) == \
        (2, 15.0, 5, True)
