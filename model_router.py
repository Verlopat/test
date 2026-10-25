#!/usr/bin/env python3
"""Ranked, sequential fallback across configured free-tier inference models.

The first *usable* response wins. A successful HTTP response does not prove
semantic correctness; use --expect for a prompt with a known exact answer.

No API keys or raw server error responses are written to disk.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv
from answer_validation import matches_expected
from router_errors import CandidateError, from_http

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=True)

SECRETS = tuple(os.getenv(name, '').strip() for name in (
    'GEMINI_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY', 'CLOUDFLARE_API_TOKEN',
    'CLOUDFLARE_ACCOUNT_ID', 'ZAI_API_KEY', 'LLM7_API_KEY', 'OLLAMA_API_KEY',
    'NVIDIA_API_KEY') if os.getenv(name, '').strip())

KEY_NAMES = {
    'Gemini': 'GEMINI_API_KEY',
    'Groq': 'GROQ_API_KEY',
    'OpenRouter': 'OPENROUTER_API_KEY',
    'Cloudflare': 'CLOUDFLARE_API_TOKEN',
    'Z.ai': 'ZAI_API_KEY',
    'LLM7.io': 'LLM7_API_KEY',
    'Ollama Cloud': 'OLLAMA_API_KEY',
    'NVIDIA NIM': 'NVIDIA_API_KEY',
}


def redacted(s: Any) -> str:
    message = str(s)
    for secret in SECRETS:
        if len(secret) >= 5:
            message = message.replace(secret, '[REDACTED]')
    return ' '.join(message.split())[:220]


def api_call(method: str, url: str, *, key: str | None = None,
             headers: dict | None = None, payload: dict | None = None,
             timeout: float = 22) -> dict:
    hs = {'Content-Type': 'application/json', **(headers or {})}
    if key:
        hs['Authorization'] = 'Bearer ' + key
    try:
        resp = requests.request(method, url, json=payload, headers=hs,
                                timeout=(5, timeout))
    except requests.exceptions.Timeout:
        raise CandidateError('Timed out', kind='timeout') from None
    except requests.RequestException:
        # Never include request URLs, exception text or authorization headers.
        raise CandidateError('Connection failed', kind='connection') from None
    if not resp.ok:
        raise from_http(resp.status_code, resp.text,
                        resp.headers.get('Retry-After'))
    try:
        parsed = resp.json()
    except ValueError:
        raise CandidateError('Non-JSON response', kind='invalid_response') from None
    if not isinstance(parsed, dict):
        raise CandidateError('Unexpected API response structure',
                             kind='invalid_response')
    if parsed.get('success') is False or parsed.get('error'):
        raise CandidateError('Provider reported an error', kind='invalid_response')
    return parsed


def extract_openai(obj: dict) -> tuple[str, str]:
    choices = obj.get('choices') or []
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise CandidateError('No choices in response')
    choice = choices[0]
    reason = choice.get('finish_reason')
    if reason in ('length', 'content_filter'):
        raise CandidateError(f'Generation stopped: {reason}')
    msg = choice.get('message') or {}
    if not isinstance(msg, dict):
        raise CandidateError('Missing message in response')
    content = msg.get('content')
    if isinstance(content, list):
        content = '\n'.join(str(x.get('text', '')) for x in content if isinstance(x, dict))
    return (content if isinstance(content, str) else ''), str(reason or '')


def clean_answer(raw: str) -> str:
    # Remove only *complete* exposed reasoning blocks. Incomplete blocks are not
    # silently transformed into successful answers.
    output = re.sub(r'<think\b[^>]*>.*?</think\s*>', '\n', raw,
                    flags=re.IGNORECASE | re.DOTALL).strip()
    if re.search(r'</?think\b', output, flags=re.IGNORECASE):
        raise CandidateError('Malformed or incomplete reasoning markup')
    if not output:
        raise CandidateError('Empty final answer')
    # Some models repeat the same answer on both sides of a thinking block.
    # The newline separator prevents accidental concatenation to 156156.
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) > 1 and all(line == lines[0] for line in lines):
        return lines[0]
    return output


def valid_zero(price: Any) -> bool:
    try:
        return Decimal(str(price)) == 0
    except (InvalidOperation, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str  # exact API model ID; never use a display name in a request
    display_name: str | None = None
    quota_group: str | None = None  # same group => assume shared rate quota
    context_window: int | None = None  # optional user-confirmed token capacity
    model_family: str | None = None  # optional cross-provider base-model identity


class Router:
    def __init__(self, timeout: float = 22.0, probe_timeout: float = 8.0):
        self.timeout = timeout
        self.probe_timeout = probe_timeout
        self.catalog: dict[str, list[dict]] = {}
        self._gemini_checked: set[str] = set()
        self._last_llm7_request: float | None = None

    def _pace_llm7(self):
        """Respect LLM7's published limit of roughly 1 request/second."""
        if self._last_llm7_request is not None:
            elapsed = time.monotonic() - self._last_llm7_request
            if elapsed < 1.1:
                time.sleep(1.1 - elapsed)
        self._last_llm7_request = time.monotonic()

    def model_catalog(self, provider: str) -> list[dict]:
        if provider in self.catalog:
            return self.catalog[provider]
        endpoint = {
            'OpenRouter': 'https://openrouter.ai/api/v1/models',
            'LLM7.io': 'https://api.llm7.io/v1/models',
            'Ollama Cloud': 'https://ollama.com/api/tags',
        }.get(provider)
        if not endpoint:
            return []
        key = os.getenv(KEY_NAMES[provider], '')
        if provider == 'LLM7.io':
            self._pace_llm7()
        response = api_call('GET', endpoint, key=key if provider != 'Ollama Cloud' else None,
                            timeout=min(self.timeout, 15))
        items = response.get('models' if provider == 'Ollama Cloud' else 'data')
        if not isinstance(items, list):
            raise CandidateError('Model catalog unavailable')
        self.catalog[provider] = [item for item in items if isinstance(item, dict)]
        return self.catalog[provider]

    def expand(self, candidate: Candidate, used: set[tuple[str, str]], limit: int = 3) -> list[Candidate]:
        # Auto catalog discovery is restricted to explicitly zero-cost OpenRouter
        # models and the provider-marked LLM7 turbo allowance.
        if candidate.provider == 'OpenRouter':
            items = self.model_catalog('OpenRouter')
            eligible = [item['id'] for item in items if isinstance(item.get('id'), str)
                        and item['id'].endswith(':free')
                        and isinstance(item.get('pricing'), dict)
                        and valid_zero(item['pricing'].get('prompt'))
                        and valid_zero(item['pricing'].get('completion'))]
            if candidate.model != 'auto-free':
                if candidate.model not in eligible:
                    raise CandidateError('Model not confirmed zero-priced in catalog')
                return [candidate]
            # Not a benchmark-derived quality order. Users may explicitly rank IDs.
            eligible.sort(key=lambda m: ('nemotron-3-ultra' not in m, m))
        elif candidate.provider == 'LLM7.io':
            items = self.model_catalog('LLM7.io')
            eligible = [item['id'] for item in items if isinstance(item.get('id'), str)
                        and item.get('tier') == 'turbo'
                        and not item.get('usage_based_only', False)
                        and item.get('model_type', 'chat') == 'chat']
            if candidate.model != 'auto-turbo':
                if candidate.model not in eligible:
                    raise CandidateError('Model not confirmed eligible for Turbo allowance')
                return [candidate]
            eligible.sort()
        elif candidate.provider == 'Ollama Cloud':
            names = {item.get('name') or item.get('model') for item in self.model_catalog('Ollama Cloud')}
            if candidate.model not in names:
                raise CandidateError('Model not in Ollama catalog')
            return [candidate]
        else:
            return [candidate]
        return [Candidate(candidate.provider, model) for model in eligible
                if (candidate.provider, model) not in used][:limit]

    def generate(self, candidate: Candidate, prompt: str, *, probe: bool = False,
                 request_timeout: float | None = None) -> str:
        """A probe is an actual short inference, not just a model lookup."""
        provider, model = candidate.provider, candidate.model
        key = os.getenv(KEY_NAMES[provider], '').strip()
        if not key:
            raise CandidateError('API key not configured')
        timeout = request_timeout if request_timeout is not None else (
            self.probe_timeout if probe else self.timeout)
        token_limit = 192 if probe else 750
        if probe:
            prompt = 'Hi. Reply with OK.'
        if provider == 'Gemini':
            headers = {'x-goog-api-key': key}
            # Validate the candidate model is visible to this key first.
            base = 'https://generativelanguage.googleapis.com/v1beta/models/' + model
            if model not in self._gemini_checked:
                api_call('GET', base, headers=headers, timeout=min(timeout, 12))
                self._gemini_checked.add(model)
            generation = {'maxOutputTokens': 192 if probe else 700}
            if model == 'gemini-3.8-flash':
                generation['thinkingConfig'] = {'thinkingLevel': 'LOW'}
            result = api_call('POST', base + ':generateContent', headers=headers,
                              payload={'contents': [{'parts': [{'text': prompt}]}],
                                       'generationConfig': generation}, timeout=timeout)
            choices = result.get('candidates') or []
            if not choices or not isinstance(choices[0], dict):
                raise CandidateError('No Gemini candidates')
            entry = choices[0]
            if entry.get('finishReason') in ('MAX_TOKENS', 'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT'):
                raise CandidateError('Generation stopped before completion')
            parts = (entry.get('content') or {}).get('parts', [])
            raw = '\n'.join(p['text'] for p in parts
                            if isinstance(p, dict) and isinstance(p.get('text'), str))
        elif provider == 'Cloudflare':
            account = os.getenv('CLOUDFLARE_ACCOUNT_ID', '').strip()
            if not account:
                raise CandidateError('Cloudflare account ID not configured')
            if not model.startswith('@cf/'):
                raise CandidateError('Cloudflare model must use a @cf/ ID')
            result = api_call('POST', f'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}',
                              key=key, payload={'messages': [{'role': 'user', 'content': prompt}]},
                              timeout=timeout)
            data = result.get('result') or {}
            if isinstance(data, dict):
                raw = data.get('response') or data.get('text') or ''
                if not raw and data.get('choices'):
                    raw, _ = extract_openai(data)
            else:
                raw = data if isinstance(data, str) else ''
        elif provider == 'Ollama Cloud':
            result = api_call('POST', 'https://ollama.com/api/chat', key=key,
                              payload={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                                       'stream': False}, timeout=timeout)
            if result.get('done_reason') in ('length', 'unload'):
                raise CandidateError('Ollama generation incomplete')
            raw = (result.get('message') or {}).get('content', '')
        else:
            endpoint = {
                'Groq': 'https://api.groq.com/openai/v1/chat/completions',
                'OpenRouter': 'https://openrouter.ai/api/v1/chat/completions',
                'Z.ai': 'https://api.z.ai/api/paas/v4/chat/completions',
                'LLM7.io': 'https://api.llm7.io/v1/chat/completions',
                'NVIDIA NIM': 'https://integrate.api.nvidia.com/v1/chat/completions',
            }[provider]
            if provider == 'LLM7.io':
                self._pace_llm7()
            result = api_call('POST', endpoint, key=key,
                              payload={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                                       'max_tokens': token_limit, 'stream': False}, timeout=timeout)
            raw, _ = extract_openai(result)
        if not isinstance(raw, str):
            raise CandidateError('Invalid final answer format')
        return clean_answer(raw)

    def ask(self, prompt: str, ordered: list[Candidate], *, expected: str | None = None,
            max_attempts: int = 20, log=None, show_rejected: bool = False,
            probe: bool = False) -> tuple[str, Candidate, list[tuple[str, str, str]]]:
        used: set[tuple[str, str]] = set()
        attempts: list[tuple[str, str, str]] = []
        log = log if log is not None else print
        for entry in ordered:
            if entry.provider not in KEY_NAMES:
                log(f'SKIP {redacted(entry.provider)}: unsupported provider')
                continue
            if not os.getenv(KEY_NAMES[entry.provider], '').strip():
                log(f'SKIP {entry.provider}: API key not configured')
                continue
            try:
                batch = self.expand(entry, used)
            except CandidateError as e:
                log(f'SKIP {entry.provider}/{redacted(entry.model)}: {redacted(e)}')
                continue
            for candidate in batch:
                identity = (candidate.provider, candidate.model)
                if identity in used:
                    continue
                if len(attempts) >= max_attempts:
                    raise CandidateError(f'No usable answer within {max_attempts} attempts')
                used.add(identity)
                start = time.monotonic()
                if probe:
                    try:
                        self.generate(candidate, 'Hi. Reply with OK.', probe=True)
                    except CandidateError as e:
                        reason = 'probe failed: ' + redacted(e)
                        attempts.append((candidate.provider, candidate.model, reason))
                        log(f'PROBE FAIL {candidate.provider}/{candidate.model}: {reason} ({time.monotonic() - start:.1f}s)')
                        continue
                    except (KeyError, TypeError, ValueError):
                        reason = 'probe failed: malformed provider response'
                        attempts.append((candidate.provider, candidate.model, reason))
                        log(f'PROBE FAIL {candidate.provider}/{candidate.model}: {reason}')
                        continue
                    log(f'PROBE OK {candidate.provider}/{candidate.model} ({time.monotonic() - start:.1f}s)')
                start = time.monotonic()
                try:
                    response = self.generate(candidate, prompt)
                    if expected is not None and not matches_expected(response, expected):
                        detail = 'Answer did not match --expect'
                        if show_rejected:
                            detail += '; preview: ' + redacted(repr(response[:120]))
                        raise CandidateError(detail)
                except CandidateError as e:
                    reason = redacted(e)
                    attempts.append((candidate.provider, candidate.model, reason))
                    log(f'QUERY FAIL {candidate.provider}/{candidate.model}: {reason} ({time.monotonic() - start:.1f}s)')
                    continue
                except (KeyError, TypeError, ValueError):
                    # Catch malformed provider replies without hiding programming
                    # errors such as an undefined variable.
                    reason = 'Malformed provider response'
                    attempts.append((candidate.provider, candidate.model, reason))
                    log(f'QUERY FAIL {candidate.provider}/{candidate.model}: {reason}')
                    continue
                attempts.append((candidate.provider, candidate.model, 'OK'))
                log(f'QUERY PASS {candidate.provider}/{candidate.model} ({time.monotonic() - start:.1f}s)')
                return response, candidate, attempts
        raise CandidateError('All configured/eligible candidates failed or were skipped')


def load_config(path: Path) -> list[Candidate]:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('priority'), list):
        raise ValueError('Config must have a priority list')
    result = []
    for row in doc['priority']:
        if not isinstance(row, dict) or not isinstance(row.get('provider'), str):
            raise ValueError('Each priority entry must provide provider and model_id/model')
        if not row['provider'].strip():
            raise ValueError('Empty provider name')
        model_id = row.get('model_id', row.get('model'))
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('Each priority entry requires a nonempty model_id/model')
        window = row.get('context_window')
        if window is not None and (not isinstance(window, int) or window <= 0):
            raise ValueError('context_window must be a positive integer or omitted')
        if row.get('enabled', True):
            result.append(Candidate(
                provider=row['provider'],
                model=model_id,
                display_name=row.get('display_name') or model_id,
                quota_group=row.get('quota_group'),
                context_window=window,
                model_family=row.get('model_family'),
            ))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prompt', nargs='?', help='Question to ask; if omitted, use --prompt or stdin')
    p.add_argument('--prompt', dest='named_prompt')
    p.add_argument('--expect', help='Known-answer check, allowing common numeric formatting')
    p.add_argument('--show-rejected', action='store_true',
                   help='Show a short preview of rejected answers (only for non-sensitive prompts)')
    p.add_argument('--config', type=Path, default=ROOT / 'model_priority.json')
    p.add_argument('--health-file', type=Path, default=ROOT / '.router_health.json')
    p.add_argument('--health-ttl', type=int, default=600,
                   help='Seconds a successfully checked model stays healthy (default 600)')
    p.add_argument('--reset-health', action='store_true',
                   help='Clear saved health/cooldowns, e.g. after changing API keys')
    p.add_argument('--prompt-file', type=Path,
                   help='Read the full unmodified question and conversation context from file')
    p.add_argument('--role', help='Council role, e.g. prover or critic')
    p.add_argument('--council-run', help='ID grouping the roles of a single council run')
    p.add_argument('--council-file', type=Path, default=ROOT / '.council_roles.json')
    p.add_argument('--timeout', type=float, default=22.0, help='Read timeout per API call (seconds)')
    p.add_argument('--max-attempts', type=int, default=20)
    p.add_argument('--no-probe', action='store_true',
                   help='Skip lightweight Hi inference before each real question')
    p.add_argument('--probe-timeout', type=float, default=8.0,
                   help='Read timeout for each lightweight probe (seconds)')
    p.add_argument('--list', action='store_true', help='Show configured ranking without sending requests')
    a = p.parse_args(argv)
    if not (2 <= a.timeout <= 120) or not (1 <= a.probe_timeout <= 120) or not (1 <= a.max_attempts <= 100):
        p.error('--timeout must be 2-120, --probe-timeout 1-120 seconds, and --max-attempts 1-100')
    if not (1 <= a.health_ttl <= 86400):
        p.error('--health-ttl must be between 1 and 86400 seconds')
    if bool(a.role) != bool(a.council_run):
        p.error('--role and --council-run must be used together')
    from router_health import HealthCache
    from router_policy import ask_cached, record_council_role
    if a.reset_health:
        a.health_file.unlink(missing_ok=True)
        print('Cleared cached health/cooldown state.')
        if not a.prompt and not a.named_prompt and not a.prompt_file:
            return 0
    try:
        candidates = load_config(a.config)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f'Configuration error: {redacted(e)}', file=sys.stderr)
        return 2
    if a.list:
        for i, c in enumerate(candidates, 1):
            status = 'configured' if os.getenv(KEY_NAMES.get(c.provider, ''), '') else 'no key'
            label = (c.display_name or c.model)
            print(f'{i:>2}. {c.provider:<14} {label:<28} [{c.model}] {status}')
        return 0
    if a.prompt_file and (a.named_prompt is not None or a.prompt is not None):
        p.error('Use either --prompt-file or a prompt argument, not both')
    if a.prompt_file:
        try:
            prompt = a.prompt_file.read_text(encoding='utf-8')
        except OSError as error:
            p.error(f'Cannot read prompt file: {type(error).__name__}')
    else:
        prompt = a.named_prompt if a.named_prompt is not None else a.prompt
        if prompt is None and not sys.stdin.isatty():
            prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        p.error('Give a question, use --prompt-file, or pipe a prompt via stdin')
    try:
        response, winning, attempts = ask_cached(
            Router(a.timeout, a.probe_timeout), prompt, candidates,
            cache=HealthCache(a.health_file, ttl_seconds=a.health_ttl),
            expected=a.expect, max_attempts=a.max_attempts,
            show_rejected=a.show_rejected, probe=not a.no_probe)
    except CandidateError as e:
        print(f'FAILED: {redacted(e)}', file=sys.stderr)
        return 1
    print(f'\nSelected: {winning.provider} / {winning.model}\n\n{response}')
    if a.role and a.council_run:
        # Count failed attempts to flag when this role needed a fallback.
        role_alerts = record_council_role(
            a.council_file, a.council_run, a.role, winning,
            fallback_used=len(attempts) > 1
        )
        print(f'\nCouncil attribution: {a.role} -> {winning.provider}/{winning.model}')
        for alert in role_alerts:
            print(alert, file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
