#!/usr/bin/env python3
"""Local API smoke checks. Never prints API keys; does not perform paid-mode validation.

Usage:
  python verify_free_apis_v2.py
  python verify_free_apis_v2.py --provider Gemini --provider Z.ai
  python verify_free_apis_v2.py --test-allowance

For Mistral/Ollama, --test-allowance sends one inference request only if you have
personally confirmed the relevant account is on its free, non-billed plan.
"""
import argparse
import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
# File entries take precedence over any stale export from an older terminal.
load_dotenv(HERE / '.env', override=True)
PROMPT = 'What is 12 multiplied by 13? Answer with only the number.'
KEYS = ['GEMINI_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY',
        'MISTRAL_API_KEY', 'CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID',
        'ZAI_API_KEY', 'LLM7_API_KEY', 'OLLAMA_API_KEY']
SECRETS = [os.getenv(key, '').strip() for key in KEYS]


def redact(value):
    s = str(value)
    for secret in SECRETS:
        if len(secret) >= 6:
            s = s.replace(secret, '[REDACTED]')
    return ' '.join(s.split())[:240]


def fetch(method, url, *, key=None, payload=None, headers=None, timeout=(8, 40)):
    hs = dict(headers or {})
    if key:
        hs['Authorization'] = f'Bearer {key}'
    try:
        r = requests.request(method, url, json=payload, headers=hs, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(f'Network/model timeout after {timeout[1]}s; key may still be valid') from None
    except requests.RequestException as e:
        raise RuntimeError(f'Connection problem: {redact(e)}') from None
    if not r.ok:
        raise RuntimeError(f'HTTP {r.status_code}: {redact(r.text)}')
    try:
        return r.json()
    except ValueError:
        raise RuntimeError('API returned non-JSON data') from None


def extract(value):
    """Support OpenAI-compatible, Cloudflare nested response, and Ollama outputs."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [extract(v) for v in value]
        return ' '.join(x for x in parts if x)
    if isinstance(value, dict):
        if isinstance(value.get('message'), dict):
            v = extract(value['message'].get('content'))
            if v:
                return v
        for field in ('choices', 'response', 'output', 'text', 'content', 'result'):
            if field in value:
                v = extract(value[field])
                if v:
                    return v
    return ''


def chat(url, key, model, *, timeout=(8, 45), tokens=300, extra=None):
    payload = {'model': model, 'messages': [{'role': 'user', 'content': PROMPT}],
               'max_tokens': tokens, 'stream': False}
    if extra:
        payload.update(extra)
    data = fetch('POST', url, key=key, payload=payload, timeout=timeout)
    answer = extract(data).strip()
    return answer[:160] if answer else '(empty answer)'


def evaluate(model, answer, extra_note=''):
    status = 'INFERENCE OK' if answer.strip() == '156' else 'RESPONSE WARNING'
    detail = answer + (f' | {extra_note}' if extra_note else '')
    return status, model, detail


def gemini(args):
    key = os.environ['GEMINI_API_KEY']
    model = 'gemini-3.8-flash'
    base = 'https://generativelanguage.googleapis.com/v1beta'
    headers = {'x-goog-api-key': key, 'Content-Type': 'application/json'}
    # An authenticated GET separates valid credentials from busy inference servers.
    info = fetch('GET', f'{base}/models/{model}', headers=headers, timeout=(8, 18))
    if not info.get('name'):
        return 'CATALOG WARNING', model, 'GET returned no model name'
    try:
        data = fetch('POST', f'{base}/models/{model}:generateContent',
                     headers=headers,
                     payload={'contents': [{'parts': [{'text': PROMPT}]}],
                              'generationConfig': {'maxOutputTokens': 256,
                                                   'thinkingConfig': {'thinkingLevel': 'LOW'}}},
                     timeout=(8, 85))
    except RuntimeError as exc:
        return 'AUTHENTICATED; INFERENCE ERROR', model, str(exc)
    candidates = data.get('candidates', [])
    answer = ' '.join(p.get('text', '') for c in candidates
                      for p in (c.get('content') or {}).get('parts', []) if p.get('text')).strip()
    return evaluate(model, answer or '(no text in response)')


def groq(args):
    model = 'openai/gpt-oss-120b'
    return evaluate(model, chat('https://api.groq.com/openai/v1/chat/completions',
                                os.environ['GROQ_API_KEY'], model))


def openrouter(args):
    key = os.environ['OPENROUTER_API_KEY']
    data = fetch('GET', 'https://openrouter.ai/api/v1/models', key=key)
    from decimal import Decimal, InvalidOperation
    def is_zero(x):
        try:
            return Decimal(str(x)) == 0
        except (InvalidOperation, ValueError, TypeError):
            return False
    free = [m['id'] for m in data.get('data', []) if m.get('id', '').endswith(':free')
            and is_zero(m.get('pricing', {}).get('prompt'))
            and is_zero(m.get('pricing', {}).get('completion'))]
    if not free:
        return 'CATALOG ONLY', '-', 'No verified zero-priced :free models found'
    free.sort(key=lambda x: ('nemotron-3-ultra' not in x, x))
    model = free[0]
    return evaluate(model, chat('https://openrouter.ai/api/v1/chat/completions', key, model))


def mistral(args):
    key = os.environ['MISTRAL_API_KEY']
    data = fetch('GET', 'https://api.mistral.ai/v1/models', key=key)
    names = [x.get('id', '') for x in data.get('data', []) if x.get('id')]
    if not args.test_allowance:
        return 'AUTHENTICATED; GENERATION NOT TESTED', '-', f'{len(names)} models visible; add --test-allowance only if your plan is Free mode'
    model = 'mistral-small-latest' if 'mistral-small-latest' in names else next((x for x in names if 'mistral-small' in x), None)
    if not model:
        return 'CATALOG ONLY', '-', 'No Mistral Small model found; generation safely skipped'
    return evaluate(model, chat('https://api.mistral.ai/v1/chat/completions', key, model))


def cloudflare(args):
    key = os.environ['CLOUDFLARE_API_TOKEN']
    acct = os.environ['CLOUDFLARE_ACCOUNT_ID']
    model = '@cf/zai-org/glm-4.7-flash'
    url = f'https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}'
    data = fetch('POST', url, key=key, payload={'messages': [{'role': 'user', 'content': PROMPT}]},
                 timeout=(8, 60))
    if data.get('success') is False:
        return 'API ERROR', model, redact(data.get('errors'))
    return evaluate(model, extract(data.get('result', {})).strip() or '(empty answer)')


def zai(args):
    model = 'glm-4.7-flash'
    return evaluate(model, chat('https://api.z.ai/api/paas/v4/chat/completions',
                                os.environ['ZAI_API_KEY'], model, timeout=(8, 60)))


def llm7(args):
    key = os.environ['LLM7_API_KEY']
    data = fetch('GET', 'https://api.llm7.io/v1/models', key=key)
    free = [m['id'] for m in data.get('data', []) if m.get('tier') == 'turbo'
            and not m.get('usage_based_only', False)
            and m.get('model_type', 'chat') == 'chat' and m.get('id')]
    if not free:
        return 'CATALOG ONLY', '-', 'No eligible turbo models found'
    free.sort()
    model = 'minimax-m2.7' if 'minimax-m2.7' in free else free[0]
    return evaluate(model, chat('https://api.llm7.io/v1/chat/completions', key, model))


def ollama(args):
    key = os.environ['OLLAMA_API_KEY']
    # This GET is public. It does not authenticate the API key.
    data = fetch('GET', 'https://ollama.com/api/tags')
    models = [m.get('name') or m.get('model') for m in data.get('models', [])]
    if not args.test_allowance:
        return 'PUBLIC CATALOG ONLY', '-', f'{len(models)} models listed; add --test-allowance after checking Free plan'
    model = 'gemma4:31b' if 'gemma4:31b' in models else next((m for m in models if m and m.startswith('gemma4:')), None)
    if not model:
        return 'CATALOG ONLY', '-', 'No Gemma 4 cloud model found; inference skipped'
    data = fetch('POST', 'https://ollama.com/api/chat', key=key,
                 payload={'model': model,
                          'messages': [{'role': 'user', 'content': PROMPT}],
                          'stream': False}, timeout=(8, 80))
    return evaluate(model, extract(data).strip() or '(empty answer)')


PROVIDERS = [
    ('Gemini', ['GEMINI_API_KEY'], gemini),
    ('Groq', ['GROQ_API_KEY'], groq),
    ('OpenRouter', ['OPENROUTER_API_KEY'], openrouter),
    ('Mistral', ['MISTRAL_API_KEY'], mistral),
    ('Cloudflare', ['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID'], cloudflare),
    ('Z.ai', ['ZAI_API_KEY'], zai),
    ('LLM7.io', ['LLM7_API_KEY'], llm7),
    ('Ollama Cloud', ['OLLAMA_API_KEY'], ollama),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider', action='append', choices=[row[0] for row in PROVIDERS],
                   help='Repeat this option to test selected providers; default: all')
    p.add_argument('--test-allowance', action='store_true',
                   help='Also send one prompt through Mistral/Ollama; use only after confirming Free mode for both')
    args = p.parse_args()
    chosen = set(args.provider or [p[0] for p in PROVIDERS])
    print('API verification v2 — credentials are never printed.\n')
    rows = []
    for name, needed, action in PROVIDERS:
        if name not in chosen:
            continue
        missing = [x for x in needed if not os.environ.get(x, '').strip()]
        if missing:
            status, model, note = 'MISSING CONFIG', '-', ', '.join(missing)
            elapsed = 0
        else:
            started = time.monotonic()
            try:
                status, model, note = action(args)
            except (RuntimeError, requests.RequestException, KeyError, ValueError, TypeError) as e:
                status, model, note = 'ERROR', '-', redact(e)
            elapsed = time.monotonic() - started
        note = redact(note)
        print(f'{name:<15} {status:<34} {elapsed:>6.2f}s  {model}', flush=True)
        print('  ' + note, flush=True)
        rows.append([name, status, model, note, f'{elapsed:.2f}'])
    dest = HERE / 'api_test_results_v2.csv'
    with dest.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Provider', 'Status', 'Model', 'Response_or_note', 'Seconds'])
        w.writerows(rows)
    print(f'\nSaved: {dest.name}\nDo not share .env or any API keys.')


if __name__ == '__main__':
    main()
