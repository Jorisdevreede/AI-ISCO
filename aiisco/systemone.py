"""The plumbing both TypeSafe scorers share: keys, throttling, the worker pool.

`score_skills_typesafe.py` and `score_skills_v2.py` ask different questions and
write different records, but they run the same way: read the key without ever
putting it on a command line, space the request starts so a wide pool cannot burst
through the rate limit, work through the skills on a thread pool, and write what
has been collected often enough that a killed run resumes rather than restarts.

The pool stops itself after MAX_CONSECUTIVE_FAILURES failures in a row. A run that
is failing every request is not going to recover by pushing on, and the useful
thing to do with it is end it while the checkpoint is still worth resuming from.
"""

import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from typesafe_sdk import TypeSafeError

MAX_CONSECUTIVE_FAILURES = 20

KEYCHAIN_COMMAND = ["security", "find-generic-password", "-a", "typesafe",
                    "-s", "typesafe-api-key", "-w"]


def load_api_key():
    """Return the TypeSafe key from the environment or the login keychain."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    result = subprocess.run(KEYCHAIN_COMMAND, capture_output=True, text=True,
                            check=False)
    if result.returncode != 0:
        raise SystemExit(
            "No TypeSafe key: set TYPESAFE_API_KEY or add the keychain item "
            "'typesafe-api-key'."
        )
    return result.stdout.strip()


def to_ten_scale(level_score):
    """Map a 0-4 band position onto the rubric's 1-10 scale (band centres)."""
    return round(1.5 + 2.0 * level_score, 2)


class RateLimiter:
    """Space request starts evenly so the run stays under the per-minute cap."""

    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        if start > now:
            time.sleep(start - now)


@dataclass
class PoolRun:
    """What the worker pool writes to, and what the progress lines report."""

    scored: dict
    total: int
    started: float
    errors: list = field(default_factory=list)
    done: int = 0
    tokens: int = 0
    consecutive: int = 0
    stopped: str = ""


@dataclass
class PoolConfig:
    """How one pooled run is driven: what to ask, where to save, how fast."""

    score_one: object
    save: object
    workers: int
    rps: float
    checkpoint_every: int


def collect(run, future, item):
    """Fold one finished request into the run, recording a refused skill."""
    try:
        entry = future.result()
    except TypeSafeError as e:
        run.errors.append(item["uri"])
        run.consecutive += 1
        print(f"\n  ERROR {item['title']!r}: {e}")
        return
    run.consecutive = 0
    run.scored[entry["uri"]] = entry
    run.tokens += entry["input_tokens"]


def print_progress(run):
    """Print the throughput line written at every checkpoint."""
    rate = run.done / (time.monotonic() - run.started)
    print(
        f"  {run.done}/{run.total} "
        f"({rate:.1f}/s, {run.tokens:,} tokens, "
        f"errors {len(run.errors)})",
        flush=True,
    )


def submit_all(pool, items, config, limiter):
    """Queue every item, keeping the item each future was submitted for."""
    return {pool.submit(config.score_one, limiter, item): item for item in items}


def drain(run, futures, config):
    """Fold finished requests in until they run out or the failure guard fires."""
    for future in as_completed(futures):
        collect(run, future, futures[future])
        run.done += 1
        if run.done % config.checkpoint_every == 0:
            config.save(run.scored)
            print_progress(run)
        if run.consecutive >= MAX_CONSECUTIVE_FAILURES:
            run.stopped = f"{MAX_CONSECUTIVE_FAILURES} requests in a row failed"
            break
    for future in futures:
        future.cancel()


def run_pool(items, scored, config):
    """Run every item through the pool, checkpointing as it goes.

    The checkpoint is written on the way out whatever happens, so a run that is
    killed, crashes or stops itself resumes from what it had rather than from zero.
    """
    limiter = RateLimiter(config.rps)
    run = PoolRun(scored=scored, total=len(items), started=time.monotonic())
    try:
        with ThreadPoolExecutor(config.workers) as pool:
            drain(run, submit_all(pool, items, config, limiter), config)
    finally:
        config.save(run.scored)
    return run


def select_items(items, args):
    """Take the requested slice, then the fixed random sample if asked for."""
    subset = items[args.start:args.end]
    if args.sample:
        random.seed(args.seed)
        subset = random.sample(subset, min(args.sample, len(subset)))
    return subset


def add_run_arguments(parser, workers, rps):
    """Add the slice, sample, pool and force flags both scorers take."""
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None,
                        help="Score a fixed random sample of this size")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for --sample")
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--rps", type=float, default=rps,
                        help="Request starts per second; keep under TypeSafe's "
                             "documented rate limit")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if already cached")
