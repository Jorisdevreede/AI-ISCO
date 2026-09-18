"""Calling OpenRouter for the two scripts that score with a generative model.

Holds the chat-completion request they share, the retry and backoff policy
around it, the clean-up their replies need before they parse as JSON, and the
batch loop and progress reporting both runs are built from. The scripts differ
only in their prompt, their timeout and which HTTP statuses are worth retrying,
so those are fields of `ChatRequest` rather than separate copies of the code.
"""

import json
import os
import re
import time
from dataclasses import dataclass

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Maximum retries for transient / rate-limit errors
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


@dataclass(frozen=True)
class ChatRequest:
    """How one script asks OpenRouter: prompt, payload limits and retry rules."""

    system_prompt: str
    model: str
    timeout: int
    retry_statuses: tuple = (429,)
    max_tokens: int = None


@dataclass(frozen=True)
class BatchLoop:
    """The parts of a batched run that differ between the two scripts.

    `handle` receives one batch and the shared error list so it can record the
    items the model left out; `checkpoint` writes the results collected so far.
    """

    batches: list
    noun: str
    handle: object
    checkpoint: object
    delay: float = 0.0


def build_payload(request, user_prompt):
    """Build the chat-completions body, omitting max_tokens when it is unset."""
    payload = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    return payload


def post_completion(client, request, user_prompt):
    """Post one chat completion and return the assistant's raw reply.

    The key is read here, inside the caller's try block, which is why a missing
    OPENROUTER_API_KEY surfaces as a parse error (see the README's
    troubleshooting table).
    """
    response = client.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        },
        json=build_payload(request, user_prompt),
        timeout=request.timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def strip_code_fences(content):
    """Strip markdown code fences from LLM response."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]  # remove first line
        content = content.removesuffix("```")
        content = content.strip()
    return content


def fix_json(text):
    """Fix common LLM JSON issues: trailing commas, comments."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text


def parse_array(content):
    """Parse a reply into the JSON array both scripts ask the model for."""
    results = json.loads(fix_json(strip_code_fences(content)))
    if not isinstance(results, list):
        # ValueError, not TypeError: call_model only retries on ValueError.
        raise ValueError(  # noqa: TRY004
            f"Expected a JSON array, got {type(results).__name__}"
        )
    return results


def backoff_for(attempt):
    """The exponential wait before the next attempt."""
    return INITIAL_BACKOFF * (2 ** attempt)


def wait_or_raise_status(error, retry_statuses, attempt):
    """Back off from a rate limit or server error, or re-raise anything else."""
    status = error.response.status_code
    if status not in retry_statuses and status < 500:
        raise error
    backoff = backoff_for(attempt)
    print(
        f"\n    Rate limited/server error ({status}), "
        f"retrying in {backoff:.0f}s (attempt {attempt + 1}/"
        f"{MAX_RETRIES})...",
        flush=True,
    )
    time.sleep(backoff)


def wait_or_raise_parse_error(error, attempt):
    """Back off from an unusable reply, or re-raise once the attempts run out."""
    if attempt >= MAX_RETRIES - 1:
        raise error
    backoff = backoff_for(attempt)
    print(
        f"\n    Parse error: {error}, retrying in {backoff:.0f}s "
        f"(attempt {attempt + 1}/{MAX_RETRIES})...",
        flush=True,
    )
    time.sleep(backoff)


def ask_once(client, request, user_prompt, validate_item):
    """Post one request and return its validated results."""
    results = parse_array(post_completion(client, request, user_prompt))
    for item in results:
        validate_item(item)
    return results


def call_model(client, request, user_prompt, validate_item):
    """Ask the model for a JSON array, retrying transient and unusable replies."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return ask_once(client, request, user_prompt, validate_item)
        except httpx.HTTPStatusError as error:
            last_error = error
            wait_or_raise_status(error, request.retry_statuses, attempt)
        except (json.JSONDecodeError, ValueError, KeyError) as error:
            last_error = error
            wait_or_raise_parse_error(error, attempt)
    raise last_error


def iter_batches(items, size):
    """Split items into consecutive batches of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def match_results(batch, results):
    """Pair each item with its result, by position and then by title."""
    result_by_title = {r["title"].lower(): r for r in results}
    pairs = []
    for item in batch:
        index = batch.index(item)
        if index < len(results):
            pairs.append((item, results[index]))
        else:
            pairs.append((item, result_by_title.get(item["title"].lower())))
    return pairs


def collect_results(state, batch, results, errors):
    """Hand every answered item to state.add and record the ones left out."""
    for item, result in match_results(batch, results):
        if result is None:
            warn_missing(item)
            errors.append(item["uri"])
            continue
        state.add(item, result)


def print_batch_header(index, total, batch, noun):
    """Announce the batch that is about to be sent."""
    titles = [item["title"] for item in batch]
    print(
        f"\n  Batch {index + 1}/{total} "
        f"({len(batch)} {noun}): {titles[0]!r} ... {titles[-1]!r}",
        end=" ",
        flush=True,
    )


def warn_missing(item):
    """Report an item the model left out of its reply."""
    print(f"\n    WARNING: No result for '{item['title']}'")


def pause_between(index, total, delay):
    """Wait between batches, but not after the last one."""
    if index < total - 1:
        time.sleep(delay)


def run_batches(loop):
    """Run every batch, checkpointing after each and pausing in between.

    Returns the URIs the run could not produce a result for: a failed batch
    contributes all of its items.
    """
    errors = []
    for index, batch in enumerate(loop.batches):
        print_batch_header(index, len(loop.batches), batch, loop.noun)
        try:
            loop.handle(batch, errors)
        except Exception as error:  # noqa: BLE001 - a bad batch must not stop the run
            print(f"ERROR: {error}")
            errors.extend(item["uri"] for item in batch)
        loop.checkpoint()
        pause_between(index, len(loop.batches), loop.delay)
    return errors


def print_failures(errors):
    """List the first ten URIs the run failed on."""
    if not errors:
        return
    print(f"Failed URIs ({len(errors)}):")
    for uri in errors[:10]:
        print(f"  {uri}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")


def tally(values):
    """Count how often each value occurs."""
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def print_histogram(title, counts, label=str):
    """Print one bar per bucket, in ascending bucket order."""
    print(f"\n{title}")
    for key in sorted(counts):
        print(f"  {label(key)}: {'█' * counts[key]} ({counts[key]})")
