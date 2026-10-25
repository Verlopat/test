#!/usr/bin/env python3
"""Minimal interactive, provider-first LLM fallback router.

No health probes, persistent cache, cooldown files, or saved prompts.
The order is read from model_priority.json. Each question is sent unchanged
when switching models/providers. Uses model_router.py for API adapters and
safe model catalog validation.

Usage:
    python simple_router.py
    python simple_router.py 'What is 12 * 13?'
    python simple_router.py --expect 156 'What is 12 * 13?'
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from answer_validation import matches_expected
from model_router import KEY_NAMES, Router, load_config, redacted  # loads local .env
from router_errors import CandidateError

# Match a stand-alone provider error, NOT a legitimate explanation about errors.
_BUSY_ONLY = re.compile(
    r"""(?ix) ^\s*
    (?:(?:error\s*[:\-]\s*)|(?:sorry[,!.]?\s*)|(?:i(?:'m|\s+am)\s+sorry[,!.]?\s*))*
    (?:
        (?:the\s+)?(?:server|service|model)\s+(?:is\s+)?(?:temporarily\s+)?
            (?:busy|overloaded|unavailable)
      | (?:rate\s+limit(?:\s+exceeded)?|too\s+many\s+requests|quota\s+exceeded)
      | (?:please\s+)?try\s+again(?:\s+later)?
    )
    (?:[,;:.!\s]*(?:please\s+)?try\s+again(?:\s+later)?)?
    [\s.!]* $"""
)


def is_real_answer(text: str) -> bool:
    """Basic structural validation only, not a factual correctness judge."""
    answer = text.strip() if isinstance(text, str) else ""
    return bool(answer) and not bool(_BUSY_ONLY.fullmatch(answer))


def quota_group(candidate) -> str:
    # Without explicit knowledge, assume all models at a provider share quota.
    return getattr(candidate, "quota_group", None) or "__shared__"


def group_is_blocked(candidate, limited: dict[str, set[str]]) -> bool:
    groups = limited.get(candidate.provider, set())
    if not groups:
        return False
    if "__shared__" in groups:
        return True
    group = getattr(candidate, "quota_group", None)
    return group is None or group in groups


def ask(
    router: Router,
    prompt: str,
    candidates,
    *,
    retries_per_model: int = 1,
    retry_delay: float = 2.0,
    max_wait: float = 5.0,
    expected: str | None = None,
    dead_providers: set[str] | None = None,
    log=print,
):
    """Try ranked candidates in order. Return (text, actual model) or (None, None).

    A bad API key disables that provider for the interactive session. Rate-limit
    blocking is scoped to this QUESTION, not saved between questions.
    Other failures immediately downgrade after at most one short retry.
    """
    if retries_per_model < 0 or retry_delay < 0 or max_wait < 0:
        raise ValueError("retries_per_model, retry_delay and max_wait cannot be negative")
    if dead_providers is None:
        dead_providers = set()

    seen: set[tuple[str, str]] = set()
    limited: dict[str, set[str]] = {}

    for entry in candidates:
        provider = entry.provider
        if provider in dead_providers:
            continue
        key_name = KEY_NAMES.get(provider)
        if key_name is None:
            log(f"SKIP  {redacted(provider)}: unsupported provider")
            continue
        if not os.getenv(key_name, "").strip():
            log(f"SKIP  {provider}: API key missing")
            dead_providers.add(provider)
            continue
        if group_is_blocked(entry, limited):
            log(f"SKIP  {provider}/{entry.model}: quota group rate-limited")
            continue

        try:
            # Reuse the parent's catalog rules (OpenRouter :free, LLM7 Turbo,
            # Ollama listed models). Pass the SAME seen set every time.
            batch = router.expand(entry, seen)
        except CandidateError as exc:
            log(f"SKIP  {provider}/{entry.model}: {redacted(exc)}")
            if exc.kind == "auth":
                dead_providers.add(provider)
            elif exc.kind == "rate_limit":
                limited.setdefault(provider, set()).add(quota_group(entry))
            continue
        except (KeyError, TypeError, ValueError):
            log(f"SKIP  {provider}/{entry.model}: invalid model catalog")
            continue

        for model in batch:
            identity = (model.provider, model.model)
            if identity in seen:
                continue
            seen.add(identity)
            if model.provider in dead_providers:
                break
            if group_is_blocked(model, limited):
                log(f"SKIP  {model.provider}/{model.model}: quota group rate-limited")
                continue

            for attempt in range(retries_per_model + 1):
                started = time.monotonic()
                try:
                    # EXACT same prompt on each try, even after fallback.
                    answer = router.generate(model, prompt, probe=False)
                    if not is_real_answer(answer):
                        raise CandidateError("empty or error-only reply", kind="invalid_response")
                    if expected is not None and not matches_expected(answer, expected):
                        raise CandidateError("answer did not match --expect", kind="invalid_response")
                except CandidateError as exc:
                    log(f"FAIL  {model.provider}/{model.model}: {redacted(exc)} "
                        f"({time.monotonic() - started:.1f}s)")
                    if exc.kind == "auth":
                        dead_providers.add(model.provider)
                        break
                    if exc.kind == "rate_limit":
                        # Respect Retry-After. Do not cap the wait and then retry
                        # prematurely. Move on if it exceeds our wait budget.
                        wait = exc.retry_after
                        if wait is None:
                            wait = retry_delay * (attempt + 1)
                        if attempt < retries_per_model and wait <= max_wait:
                            log(f"WAIT  {wait:.1f}s; retrying {model.model} once")
                            time.sleep(wait)
                            continue
                        limited.setdefault(model.provider, set()).add(quota_group(model))
                        break
                    if exc.kind in {"server", "timeout"} and attempt < retries_per_model:
                        if retry_delay <= max_wait:
                            log(f"WAIT  {retry_delay:.1f}s; retrying {model.model} once")
                            time.sleep(retry_delay)
                            continue
                    break  # unavailable, context too long, invalid reply, etc.
                except (KeyError, TypeError, ValueError):
                    log(f"FAIL  {model.provider}/{model.model}: malformed response")
                    break
                else:
                    log(f"PASS  {model.provider}/{model.model} "
                        f"({time.monotonic() - started:.1f}s)")
                    return answer, model
            if model.provider in dead_providers:
                break
    return None, None


def choose_candidates(candidates, *, only_provider=None, start_provider=None,
                      skip_providers=()):
    """Filter the configured order without changing the order within providers."""
    if only_provider and start_provider:
        raise ValueError("--only-provider and --start-provider cannot be combined")
    known = {item.provider for item in candidates}
    selected = only_provider or start_provider
    if selected and selected not in known:
        raise ValueError(
            f"Unknown provider {selected!r}. Choices: {', '.join(sorted(known))}"
        )
    if only_provider:
        filtered = [item for item in candidates if item.provider == only_provider]
    elif start_provider:
        first = next(i for i, item in enumerate(candidates)
                     if item.provider == start_provider)
        filtered = candidates[first:]
    else:
        filtered = list(candidates)
    return [item for item in filtered if item.provider not in skip_providers]


def provider_batches(candidates):
    """Each provider gets its own fallback list; retain configured order."""
    grouped = {}
    for item in candidates:
        grouped.setdefault(item.provider, []).append(item)
    return list(grouped.items())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="ask once and exit; omit for interactive mode")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("model_priority.json"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--max-wait", type=float, default=5.0)
    parser.add_argument("--expect", help="optional expected answer for known-answer tests")
    parser.add_argument("--only-provider", help="try only this provider and its backups")
    parser.add_argument("--start-provider", help="start at this provider, then try subsequent providers")
    parser.add_argument("--skip-provider", action="append", default=[],
                        help="exclude a provider from this run (repeat to exclude more)")
    parser.add_argument("--check-providers", action="store_true",
                        help="ask every configured provider independently, using backups within each")
    args = parser.parse_args(argv)
    if not 2 <= args.timeout <= 120:
        parser.error("--timeout must be between 2 and 120 seconds")
    if args.retries < 0 or args.retries > 3 or args.retry_delay < 0 or args.max_wait < 0:
        parser.error("--retries must be 0–3; delays cannot be negative")

    try:
        candidates = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot load ranking: {redacted(exc)}")
    try:
        candidates = choose_candidates(
            candidates, only_provider=args.only_provider,
            start_provider=args.start_provider, skip_providers=args.skip_provider
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not candidates:
        parser.error("No models remain after provider filtering")
    router = Router(timeout=args.timeout)
    bad_keys_this_session: set[str] = set()

    def run(question: str) -> bool:
        answer, model = ask(
            router, question, candidates, retries_per_model=args.retries,
            retry_delay=args.retry_delay, max_wait=args.max_wait,
            expected=args.expect, dead_providers=bad_keys_this_session,
        )
        if model is None:
            print("\nNo model returned an accepted answer.\n")
            return False
        print(f"\nSelected: {model.provider} / {model.model}\n\n{answer}\n")
        return True

    if args.check_providers:
        if not args.question:
            parser.error("--check-providers requires a question to ask each provider")
        question = " ".join(args.question)
        passed = failed = skipped = 0
        print(f"Checking {len(provider_batches(candidates))} providers independently.\n")
        for provider, entries in provider_batches(candidates):
            key_name = KEY_NAMES.get(provider)
            if not key_name or not os.getenv(key_name, "").strip():
                print(f"SKIP PROVIDER {provider}: API key not configured\n")
                skipped += 1
                continue
            print(f"CHECK PROVIDER {provider}", flush=True)
            answer, model = ask(
                router, question, entries, retries_per_model=args.retries,
                retry_delay=args.retry_delay, max_wait=args.max_wait,
                expected=args.expect, dead_providers=bad_keys_this_session,
            )
            if model is None:
                failed += 1
                print(f"PROVIDER FAILED {provider}\n")
            else:
                passed += 1
                print(f"PROVIDER PASSED {provider} using {model.model}: {answer[:160]!r}\n")
        print(f"Provider checks complete: {passed} passed, {failed} failed, {skipped} skipped.")
        return 1 if failed or not passed else 0

    if args.question:
        return 0 if run(" ".join(args.question)) else 1

    print("Ask a question (type 'exit' to quit; Ctrl+D also quits).")
    while True:
        try:
            question = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if question.strip().lower() in {"exit", "quit"}:
            return 0
        if question.strip():
            run(question)  # Preserve the actual question (including whitespace).


if __name__ == "__main__":
    raise SystemExit(main())
