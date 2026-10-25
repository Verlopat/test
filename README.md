# Free-tier API checks + automatic model fallback

Local Python tools for **seven previously configured providers** (Gemini, Groq, OpenRouter, Cloudflare, Z.ai, LLM7.io and Ollama Cloud) plus **optional NVIDIA NIM**. Mistral has been excluded because this account's API inference requires an upgrade.

No LLM Council or website has been built. **Mistral is not used in either the default verifier or the new router.**

## Install

From your existing Git checkout:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On first setup, copy `.env.example` to .env, enter your keys **locally**, then `chmod 600 .env`. Never commit or paste your keys.

## Provider-first handshake and fallback

This is now a **two-step inference process per candidate**, enabled by default:

1. Send a short `Hi. Reply with OK.` prompt to Gemini 3.8 Flash.
2. If the probe fails, try Gemini 3.6 Flash, then Gemini 3.5 Flash-Lite.
3. If a probe succeeds, send the **real prompt** to that same model.
4. If the real prompt fails, still move to the next model.
5. After Gemini's candidates, try LLM7's MiniMax-M2.7, Nemotron Nano 30B, GPT-OSS 20B, and Codestral.
6. Only after those move to Groq, OpenRouter, Ollama Cloud, Cloudflare, Z.ai and optional NVIDIA NIM.

The order is in `model_priority.json`; these are preferences, not a proven quality ranking. Each LLM7 candidate must still appear as an eligible **Turbo** model in the live catalog. Mistral's **own provider API** remains excluded; Codestral via the independent LLM7 service is a different endpoint.

**A successful Hi response does not guarantee the real prompt will succeed.** The probe is an actual paid-or-free-allowance inference request, **not** a permanent connection or authorization reservation. Every model attempted may consume two inference requests, and the router sends the real question only after its probe succeeds. Free quotas and limits are enforced by the providers; the router does not guarantee free billing.

The LLM7 connector spaces requests to respect the documented roughly one-request-per-second free-tier rate limit. If probes are consuming too much quota or slowing down the workflow, use `--no-probe`. The router still performs fallback for failed real queries without probes.

## Run the ranked fallback router

```bash
# Show the ranking and which providers have keys without making API calls
python model_router.py --list

# Default: send a short Hi probe, then the real question to a responsive model
python model_router.py "Explain what a zero-knowledge proof is"

# Skip the extra Hi requests if you need fewer API calls
python model_router.py --no-probe "Explain a zero-knowledge proof"

# Test accuracy for a prompt with a known answer
python model_router.py --expect 156 "What is 12 multiplied by 13?"

# Avoid waiting long for an overloaded model
python model_router.py --probe-timeout 7 --timeout 12 --max-attempts 8 "Explain Merkle trees"
```

The **first eligible model with a usable response to the real question wins**, and later models are not called. Messages identify each `PROBE OK`, `PROBE FAIL`, `QUERY PASS` and `QUERY FAIL`. Failures including 401/403, 429, 5xx, timeouts, missing keys, nonexistent models, empty output, malformed thinking tags and truncated responses fall through to the next candidate. The program prints FAIL/SKIP/PASS events, the winning provider and its answer. If all configured options fail, it exits with an error rather than inventing an answer.

**Critical distinction:** Usable does not mean factually correct. The router cannot automatically prove answer quality for open-ended prompts. For exact-answer tests, `--expect` verifies the output and falls back if wrong.

### Ranking and selecting fallback models

Edit `model_priority.json`. Entries are attempted **top-to-bottom**:

```json
{"priority": [
  {"provider": "Gemini", "model": "gemini-3.8-flash"},
  {"provider": "Gemini", "model": "gemini-3.5-flash-lite"},
  {"provider": "Groq", "model": "openai/gpt-oss-120b"}
]}
```

If Gemini 3.8 Flash fails, try Gemini 3.5 Flash-Lite. If that fails, try Groq. Change the order whenever you want; there is **no scientifically verified universal model ranking**.

The included ranking deliberately begins with Gemini 3.8/3.6/3.5 Flash-Lite, then LLM7 MiniMax and its backups, as requested. Consequently, the default run can be slower when Gemini is overloaded. `--probe-timeout` controls the time spent on Hi probes. `auto-free` under OpenRouter checks the live model catalog and tries up to three additional models with **both input and output listed at zero price**. `auto-turbo` for LLM7.io is available if added to the configuration, and checks its Turbo/non-usage-based catalog markers. These automated catalog orders are **not quality rankings**. Candidate IDs are attempted only once.

Some services offer a free quota but also paid plans. This code cannot inspect billing settings or guarantee charges cannot occur. **Confirm your own free/non-billed plan and quota** for every configured provider before making requests. A free API key is not automatically a free API inference plan.

The router makes no requests to providers that have no configured key. It does not share prompts with every provider; only those attempted until a valid answer is obtained. Avoid sending sensitive or private information to third-party APIs.

## Adding NVIDIA NIM instead of Mistral

NVIDIA's Developer Program offers free hosted **Free Endpoint** API access for prototyping (subject to eligibility and limits). Create a key via <https://build.nvidia.com/> and add this line to your existing local .env:

```dotenv
NVIDIA_API_KEY=your_private_key_here
```

The default configuration includes NVIDIA's `nvidia/nemotron-3-super-120b-a12b` hosted candidate. Its inference will be **skipped** until the key exists. Check in the NVIDIA model catalog that this candidate still has a Free Endpoint before testing. Other NVIDIA endpoints may be paid or unavailable.

Test it alone:

```bash
python verify_free_apis_v2.py --provider "NVIDIA NIM"
```

For the other providers, run the original smoke test:

```bash
python verify_free_apis_v2.py

# Ollama inference is opt-in, after confirming Free plan
python verify_free_apis_v2.py --provider "Ollama Cloud" --test-allowance
```

Mistral is no longer tested. Reports from the verifier go to `api_test_results_v2.csv` and the router does not persist prompts or responses.

## Offline regression tests

```bash
python -m unittest discover -s tests -v
```

Tests use mocked HTTP responses and do not call external providers. Live results still depend on account access, uptime and current catalog/prices.

## Existing folder

If already linked to this GitHub repo: `git pull --ff-only`. If the folder has no Git metadata, download the code archive directly into it:

```bash
curl -fL https://github.com/Verlopat/test/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1
```

Do not upload your .env or secrets. .gitignore prevents them from being tracked accidentally.
