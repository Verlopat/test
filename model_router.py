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


class CandidateError(Exception):
    """Safe descriptive error that triggers fallback."""


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
        raise CandidateError('Timed out') from None
    except requests.RequestException:
        # Do not include exception text, URLs or credential-bearing headers.
        raise CandidateError('Connection failed') from None
    if not resp.ok:
        detail = {401: 'authentication failed', 403: 'access denied',
                  404: 'model/endpoint unavailable', 408: 'request timeout',
                  429: 'rate limit or quota exceeded',
                  500: 'provider internal error', 502: 'provider gateway error',
                  503: 'provider overloaded', 504: 'provider timeout'}
        raise CandidateError(f'HTTP {resp.status_code}: {detail.get(resp.status_code, "request rejected")}')
    try:
        parsed = resp.json()
    except ValueError:
        raise CandidateError('Non-JSON response') from None
    if not isinstance(parsed, dict):
        raise CandidateError('Unexpected API response structure')
    if parsed.get('success') is False or parsed.get('error'):
        raise CandidateError('Provider reported an error')
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
    model: str


class Router:
    def __init__(self, timeout: float = 22.0):
        self.timeout = timeout
        self.catalog: dict[str, list[dict]] = {}

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

    def generate(self, candidate: Candidate, prompt: str) -> str:
        provider, model = candidate.provider, candidate.model
        key = os.getenv(KEY_NAMES[provider], '').strip()
        if not key:
            raise CandidateError('API key not configured')
        timeout = self.timeout
        if provider == 'Gemini':
            headers = {'x-goog-api-key': key}
            # Validate the candidate model is visible to this key first.
            base = 'https://generativelanguage.googleapis.com/v1beta/models/' + model
            api_call('GET', base, headers=headers, timeout=min(timeout, 12))
            generation = {'maxOutputTokens': 700}
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
            result = api_call('POST', endpoint, key=key,
                              payload={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                                       'max_tokens': 750, 'stream': False}, timeout=timeout)
            raw, _ = extract_openai(result)
        if not isinstance(raw, str):
            raise CandidateError('Invalid final answer format')
        return clean_answer(raw)

    def ask(self, prompt: str, ordered: list[Candidate], *, expected: str | None = None,
            max_attempts: int = 12, log=None, show_rejected: bool = False) -> tuple[str, Candidate, list[tuple[str, str, str]]]:
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
                    log(f'FAIL {candidate.provider}/{candidate.model}: {reason} ({time.monotonic() - start:.1f}s)')
                    continue
                except (KeyError, TypeError, ValueError):
                    # Catch malformed provider replies without hiding programming
                    # errors such as an undefined variable.
                    reason = 'Malformed provider response'
                    attempts.append((candidate.provider, candidate.model, reason))
                    log(f'FAIL {candidate.provider}/{candidate.model}: {reason}')
                    continue
                attempts.append((candidate.provider, candidate.model, 'OK'))
                log(f'PASS {candidate.provider}/{candidate.model} ({time.monotonic() - start:.1f}s)')
                return response, candidate, attempts
        raise CandidateError('All configured/eligible candidates failed or were skipped')


def load_config(path: Path) -> list[Candidate]:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('priority'), list):
        raise ValueError('Config must have a priority list')
    result = []
    for row in doc['priority']:
        if not isinstance(row, dict) or not isinstance(row.get('provider'), str) or not isinstance(row.get('model'), str):
            raise ValueError('Each priority entry must provide provider and model strings')
        if not row['provider'].strip() or not row['model'].strip():
            raise ValueError('Empty provider or model name')
        if row.get('enabled', True):
            result.append(Candidate(row['provider'], row['model']))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prompt', nargs='?', help='Question to ask; if omitted, use --prompt or stdin')
    p.add_argument('--prompt', dest='named_prompt')
    p.add_argument('--expect', help='Known-answer check, allowing common numeric formatting')
    p.add_argument('--show-rejected', action='store_true',
                   help='Show a short preview of rejected answers (only for non-sensitive prompts)')
    p.add_argument('--config', type=Path, default=ROOT / 'model_priority.json')
    p.add_argument('--timeout', type=float, default=22.0, help='Read timeout per API call (seconds)')
    p.add_argument('--max-attempts', type=int, default=12)
    p.add_argument('--list', action='store_true', help='Show configured ranking without sending requests')
    a = p.parse_args(argv)
    if not (2 <= a.timeout <= 120) or not (1 <= a.max_attempts <= 100):
        p.error('--timeout must be 2-120 seconds and --max-attempts must be 1-100')
    try:
        candidates = load_config(a.config)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f'Configuration error: {redacted(e)}', file=sys.stderr)
        return 2
    if a.list:
        for i, c in enumerate(candidates, 1):
            status = 'configured' if os.getenv(KEY_NAMES.get(c.provider, ''), '') else 'no key'
            print(f'{i:>2}. {c.provider:<14} {c.model:<43} {status}')
        return 0
    prompt = a.named_prompt or a.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        p.error('Give a question, use --prompt, or pipe a prompt via stdin')
    try:
        response, winning, attempts = Router(a.timeout).ask(
            prompt.strip() + ('\n\nFor this test, output only the final answer; no explanation.' if a.expect is not None else ''),
            candidates, expected=a.expect, max_attempts=a.max_attempts,
            show_rejected=a.show_rejected)
    except CandidateError as e:
        print(f'FAILED: {redacted(e)}', file=sys.stderr)
        return 1
    print(f'\nSelected: {winning.provider} / {winning.model}\n\n{response}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
