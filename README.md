# Free API model router — cached health + provider-first fallback

Local Python router that tries **Gemini main + backups → LLM7 main + backups → Groq → OpenRouter → Ollama Cloud → Cloudflare → Z.ai → optional NVIDIA NIM**. It is not a full multi-model research council or website.

The default configuration has **exact API model IDs** in `model_priority.json` alongside human-readable `display_name` fields. A display name is **never** sent to the API. Mistral's own provider API is excluded (it requires a plan upgrade in this account). LLM7 may host a Codestral model independently.

## Install / update (Arch Linux or other Linux)

From your existing repository directory:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On first setup only, copy `.env.example` to .env, fill in your API keys locally, and run `chmod 600 .env`. Do not share or commit your real keys. Existing local .env is not overwritten by `git pull`.

## How the new selection works

1. **First contact:** call the top model with a small `Hi. Reply with OK.` inference probe.
2. **If it responds:** send the **original, unchanged** question to the same model.
3. **If it succeeds:** mark that exact provider/model healthy in a private local file for **10 minutes** (`600` seconds).
4. **Further questions:** send the real question **directly to the last successful model** (the temporary preferred winner) while its 10-minute health cache is valid; no repeat Hi calls.
5. **If either the probe or real request fails:** classify the error, apply a cooldown when appropriate, and try the next eligible model **within the same provider**, then move to the next provider.
6. **After 10 minutes:** the temporary preference expires, and the router starts from the original Gemini → LLM7 → remaining-provider ranking again. A cooled-down model is not retried until its cooldown expires.

A successful Hi check is **not a guarantee** the real question will work. Even cached models always retain fallback handling for the actual request.

The last successful model is preferred only during its health TTL. Use `--no-sticky` to always start with your original provider ranking (while still reusing per-model healthy status).

The cache survives separate terminal executions as `.router_health.json` (private `0600` permissions). It stores **only provider/model IDs, health timestamps, cooldown timestamps and failure categories**—not API keys, requests or model outputs. .gitignore excludes it.

### Failure behavior

| Condition | Action |
| --- | --- |
| Invalid credentials (401/403) | Skip remaining models of this provider; provider cooldown |
| Model missing/unavailable (404, matching 400) | Cool down only that model; try backup |
| 429 rate limit | Honor provider `Retry-After` if provided; cool down the quota group and try another eligible independent group or provider |
| Timeout / 5xx / connection failure | One short retry, then model cooldown and fallback |
| Context too long (400/413/422 matching context error) | Move to a model with enough known context; **do not** mark the first one unhealthy |
| Malformed, empty, incomplete or known-wrong answer | Model cooldown and fallback |
| Missing provider key | Skip that provider without making requests |

**Quota groups:** model entries can specify `quota_group`. Without an explicit group the router assumes the provider's quota is shared, to avoid hammering multiple models on one rate-limited account. Only assign a different group when you have verified the two models really have separate quotas. Long `Retry-After` delays are saved as cooldowns rather than making the router wait indefinitely.

**Context size:** add optional `context_window` (tokens) only when verified for that exact model. The local pre-check is an approximate token estimate and cannot replace the provider's tokenizer. The LLM7 model windows in the default config come from its public catalog. Other context sizes remain unspecified; their API errors are handled if they arise.

### Basic usage

```bash
# Show exact API IDs, display names and which keys are configured (no API calls)
python model_router.py --list

# On first use: probe, then send this exact question
python model_router.py "Explain lattice-based VRFs"

# Known-answer smoke test
python model_router.py --expect 156 "What is 12 multiplied by 13?"

# Full conversation context stored in one text file, unmodified across attempts
python model_router.py --prompt-file research_question.txt

# Make health checks last 10 minutes (default), with explicit timeouts
python model_router.py --health-ttl 600 --probe-timeout 8 --timeout 22 "Explain ZK folding schemes"

# Disable probes entirely (real questions still populate the health cache)
python model_router.py --no-probe "What is a lattice trapdoor?"

# Always follow the original Gemini-first order instead of last-winner preference
python model_router.py --no-sticky "Explain a commitment scheme"

# Clear saved health and cooldowns, especially after replacing a bad API key
python model_router.py --reset-health

# For harmless diagnostics only; do not use with private prompts
python model_router.py --expect 156 --show-rejected "What is 12 multiplied by 13?"
```

**Known-answer validation:** `--expect` can recognize simple numerical answer formatting, but the router does not know whether a long research explanation is factually correct. It returns the first successfully completed response that passes structural checks, not necessarily the best answer.

**Privacy:** each attempted provider receives the same real user question, including any conversation history supplied as text. Only models up to and including the winner receive it. Do not supply sensitive material to third-party providers unless you are comfortable with their policies.

### Ranking and model IDs

Edit `model_priority.json` to change the exact order:

```json
{
  "priority": [
    {
      "provider": "Gemini",
      "model_id": "gemini-3.8-flash",
      "display_name": "Gemini 3.8 Flash"
    },
    {
      "provider": "Gemini",
      "model_id": "gemini-3.7-flash",
      "display_name": "Gemini 3.7 Flash"
    },
    {
      "provider": "LLM7.io",
      "model_id": "minimax-m2.7",
      "display_name": "MiniMax M2.7",
      "context_window": 180000
    }
  ]
}
```

`model_id` is the exact text sent to the provider. `display_name` is only for people. For explicit independent quota pools, set a supported `quota_group`; for cross-provider council comparison you may set a `model_family` indicating the common underlying model. The router also accepts older config files using `model` instead of `model_id`.

Model ranking is **user-preference order**, not a benchmark-based proof of superiority. Live model IDs and availability can change.

### Council role attribution (optional)

This does **not** create an LLM Council. It only records which model actually answered each named role. Keep one `--council-run` ID across that council session:

```bash
python model_router.py --council-run trial-001 --role proposer "Propose a theorem"
python model_router.py --council-run trial-001 --role critic "Critique the theorem"
python model_router.py --council-run trial-001 --role verifier "Evaluate the proof"
```

The private `.council_roles.json` records **role → actual provider, exact model ID, underlying base model, whether fallback was needed, timestamp**. It stores no prompt text or answers. If multiple roles resolve to the same underlying base model (including through different API providers), the program prints a model-reuse warning, and a stronger independence warning when three or more roles share it. Thus a council never silently counts one base model as three independent model perspectives.

## Costs and free-tier safeguards

The router cannot verify billing configuration or guarantee an unlimited free allowance. Before running queries, confirm each provider account is on a free/non-billed plan and check current usage limits. An API key by itself does not mean inference is free.

OpenRouter only uses explicit :free model IDs that its live catalog reports with zero input **and** output prices. LLM7 only uses eligible Turbo models with the provider's non-usage-based catalog flag. NVIDIA NIM is optional and skipped when `NVIDIA_API_KEY` is not configured; verify that a particular endpoint is available under the developer free quota before adding its key. Model IDs in the config do not authorize paid access by themselves.

## Test

```bash
python -m unittest discover -s tests -v
python -m py_compile model_router.py router_policy.py router_health.py router_errors.py answer_validation.py
```

Offline tests use mocked HTTP and do not use any real API keys or inference credits. GitHub Actions runs them on pushes to `main`. A passing offline suite does **not** replace live provider testing.

See also `verify_free_apis_v2.py` for a basic per-provider smoke check.
