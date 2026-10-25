"""Cached-health model selection, failure-aware fallback and council role attribution.

The router retains the *exact same* user prompt on every actual query attempt.
Probes use a separate brief prompt and only occur for stale/unknown models.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from answer_validation import matches_expected
from router_errors import CandidateError
from router_health import HealthCache


TRANSIENT_KINDS = frozenset(("timeout", "server", "connection"))


def estimated_tokens(prompt: str) -> int:
    # Rough protective estimate only: actual tokenization depends on model.
    return math.ceil(len(prompt) / 3)


def call_with_short_retry(router, candidate, prompt: str, *, probe: bool, log):
    """One bounded retry on transport/server failures, never on quota/auth."""
    try:
        return router.generate(candidate, prompt, probe=probe)
    except CandidateError as error:
        if error.kind not in TRANSIENT_KINDS:
            raise
        stage = "PROBE" if probe else "QUERY"
        shorter = min(8.0, router.probe_timeout if probe else router.timeout)
        log(f"RETRY {stage} {candidate.provider}/{candidate.model}: {error}")
        # Do not mutate or append anything to the real prompt.
        time.sleep(0.25)
        return router.generate(candidate, prompt, probe=probe,
                               request_timeout=shorter)


def _model_capacity_problem(candidate, prompt: str) -> bool:
    limit = getattr(candidate, "context_window", None)
    if limit is None:
        return False
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return False
    if limit <= 0:
        return False
    # Leave room for the generated answer and chat-message overhead.
    reserved = min(512, max(32, limit // 8))
    return estimated_tokens(prompt) + reserved > limit


def ask_cached(router, prompt: str, ordered, *, cache: HealthCache,
               expected: str | None = None, max_attempts: int = 20,
               log=None, probe: bool = True, show_rejected: bool = False):
    """Return (answer, actual_candidate, attempts).

    max_attempts counts models (not lightweight probes or retries). A cooldown
    is stored on disk and checked again by the next CLI process.
    """
    from model_router import KEY_NAMES  # same provider registry as the CLI

    if log is None:
        log = print
    used = set()
    attempted = []
    for entry in ordered:
        provider = entry.provider
        model = entry.model
        quota_group = getattr(entry, "quota_group", None)
        if provider not in KEY_NAMES:
            log(f"SKIP {provider}: unsupported provider")
            continue
        if not os.getenv(KEY_NAMES[provider], "").strip():
            log(f"SKIP {provider}: key not configured")
            continue
        blocked = cache.blocked(provider, model, quota_group)
        if blocked:
            log(f"COOLDOWN {provider}/{model}: {blocked[0]} ({blocked[1]}s remaining)")
            continue
        try:
            expanded = router.expand(entry, used)
        except CandidateError as error:
            cache.failure(provider, model, error.kind, quota_group, error.retry_after)
            log(f"SKIP {provider}/{model}: catalog check {error}")
            continue
        for candidate in expanded:
            ident = (candidate.provider, candidate.model)
            if ident in used:
                continue
            if len(attempted) >= max_attempts:
                raise CandidateError(f"No usable answer within {max_attempts} models")
            used.add(ident)
            quota = getattr(candidate, "quota_group", None)
            blocked = cache.blocked(candidate.provider, candidate.model, quota)
            if blocked:
                log(f"COOLDOWN {candidate.provider}/{candidate.model}: {blocked[0]} ({blocked[1]}s remaining)")
                continue
            if _model_capacity_problem(candidate, prompt):
                log(f"CONTEXT SKIP {candidate.provider}/{candidate.model}: configured context capacity too small")
                # Not a model-health failure; do not write a cooldown.
                continue

            if probe:
                if cache.healthy(candidate.provider, candidate.model, quota):
                    log(f"PROBE CACHED {candidate.provider}/{candidate.model}")
                else:
                    try:
                        call_with_short_retry(router, candidate, "Hi. Reply with OK.",
                                              probe=True, log=log)
                    except CandidateError as error:
                        cache.failure(candidate.provider, candidate.model, error.kind,
                                      quota, error.retry_after)
                        attempted.append((candidate.provider, candidate.model,
                                          f"probe failed: {error.kind}"))
                        log(f"PROBE FAIL {candidate.provider}/{candidate.model}: {error}")
                        continue
                    except (KeyError, ValueError, TypeError):
                        cache.failure(candidate.provider, candidate.model, "invalid_response", quota)
                        attempted.append((candidate.provider, candidate.model, "probe malformed"))
                        log(f"PROBE FAIL {candidate.provider}/{candidate.model}: malformed response")
                        continue
                    cache.success(candidate.provider, candidate.model)
                    log(f"PROBE OK {candidate.provider}/{candidate.model} (cached {cache.ttl}s)")

            try:
                # Exact prompt, with no hidden suffix, system-message change or
                # loss of earlier conversational context.
                answer = call_with_short_retry(router, candidate, prompt, probe=False, log=log)
                if expected is not None and not matches_expected(answer, expected):
                    detail = "answer did not match --expect"
                    if show_rejected:
                        from model_router import redacted
                        detail += "; preview: " + redacted(repr(answer[:120]))
                    raise CandidateError(detail, kind="invalid_response")
            except CandidateError as error:
                cache.failure(candidate.provider, candidate.model, error.kind,
                              quota, error.retry_after)
                attempted.append((candidate.provider, candidate.model, error.kind))
                log(f"QUERY FAIL {candidate.provider}/{candidate.model}: {error}")
                continue
            except (KeyError, ValueError, TypeError):
                cache.failure(candidate.provider, candidate.model, "invalid_response", quota)
                attempted.append((candidate.provider, candidate.model, "invalid_response"))
                log(f"QUERY FAIL {candidate.provider}/{candidate.model}: malformed response")
                continue
            cache.success(candidate.provider, candidate.model)
            attempted.append((candidate.provider, candidate.model, "OK"))
            log(f"QUERY PASS {candidate.provider}/{candidate.model}")
            return answer, candidate, attempted
    raise CandidateError("All configured candidates failed or were skipped")


def _private_json_save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False
        ) as out:
            temporary = Path(out.name)
            os.chmod(temporary, 0o600)
            json.dump(data, out, ensure_ascii=False, indent=2)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def underlying_model_identity(candidate) -> str:
    """Compare base models even when accessed through different providers.

    Explicit model_family metadata wins. In its absence, use the last
    component of the API model ID, stripping only the :free access suffix.
    This is a heuristic; custom fine-tunes should provide model_family.
    """
    family = getattr(candidate, "model_family", None)
    if isinstance(family, str) and family.strip():
        return family.strip().lower()
    suffix = candidate.model.rsplit("/", 1)[-1]
    if suffix.endswith(":free"):
        suffix = suffix[:-5]
    if suffix.lower() in ("latest", "default", "auto-free", "auto-turbo"):
        return (candidate.provider + "/" + candidate.model).lower()
    return suffix.lower()


def record_council_role(path: Path, run_id: str, role: str, candidate,
                        *, fallback_used: bool = False) -> list[str]:
    """Record who *actually answered* a role. No prompts, answers or keys saved.

    Return warning strings for duplicated underlying models among roles.
    """
    path = Path(path)
    data: dict[str, Any] = {"version": 1, "runs": {}}
    if path.exists():
        try:
            read = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(read, dict) and isinstance(read.get("runs"), dict):
                data = read
        except (ValueError, OSError, UnicodeError):
            pass
    runs = data["runs"]
    selected = runs.setdefault(run_id[:128], {"roles": {}, "updated_at": 0})
    selected["roles"][role[:128]] = {
        "provider": candidate.provider,
        "model_id": candidate.model,
        "base_model": underlying_model_identity(candidate),
        "display_name": getattr(candidate, "display_name", None) or candidate.model,
        "fallback_used": bool(fallback_used),
        "answered_at": time.time(),
    }
    selected["updated_at"] = time.time()
    # Keep the 50 most recently updated runs to limit local metadata growth.
    if len(runs) > 50:
        sorted_keys = sorted(runs, key=lambda key: runs[key].get("updated_at", 0))
        for key in sorted_keys[:-50]:
            del runs[key]
    _private_json_save(path, data)

    role_groups = {}
    for role_name, result in selected["roles"].items():
        identifier = result.get("base_model") or (
            result["provider"] + "/" + result["model_id"])
        role_groups.setdefault(identifier, []).append(role_name)
    alerts = []
    for identifier, same_roles in role_groups.items():
        if len(same_roles) >= 2:
            count = len(same_roles)
            prefix = "COUNCIL INDEPENDENCE WARNING" if count >= 3 else "COUNCIL MODEL REUSE"
            alerts.append(
                f"{prefix}: {count} roles ({', '.join(same_roles)}) used "
                f"the same underlying model ({identifier}) across their selected APIs; "
                "do not count these as independent model perspectives."
            )
    return alerts
