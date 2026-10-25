# Free AI API verification

A **local Python-only** smoke test for eight hosted model providers. This repository does **not** contain a website or an LLM Council implementation yet.

## Setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
nano .env
```

Put the real API keys only in your **local** `.env`. Never upload that file to GitHub.

## Run

```bash
python verify_free_apis_v2.py
python verify_free_apis_v2.py --provider Gemini --provider Z.ai
```

Output is saved to `api_test_results_v2.csv`. It contains no API keys, but may include provider response details.

Mistral and Ollama skip paid-risk inference by default. **Only after checking that your accounts are on Free/non-billed plans**:

```bash
python verify_free_apis_v2.py --provider Mistral --provider "Ollama Cloud" --test-allowance
```

The script is a *smoke test*, not a scientific model-quality benchmark or a guarantee that every request is free. Provider prices, free-tier access, model IDs and limits can change. OpenRouter is restricted to models whose catalog entries say both input and output cost zero. Check other providers' dashboards before running tests.

Known observations: Gemini may return HTTP 503 (model overload); Z.ai may return HTTP 429 (temporary overload); LLM7 may include unwanted `<think>` text. An HTTP 401 generally indicates invalid credentials; an accessible public model catalog (Ollama) does not establish authentication.

## Getting files into an existing local folder

If the directory is **already a Git checkout** of this repository:

```bash
git pull --ff-only origin main
```

If you previously created a plain folder with your own `.env` and `verify_free_apis_v2.py`, **do not run `git pull` in it**. Instead, from within that folder download the script directly:

```bash
curl -fL https://raw.githubusercontent.com/Verlopat/test/main/verify_free_apis_v2.py -o verify_free_apis_v2.py
curl -fL https://raw.githubusercontent.com/Verlopat/test/main/requirements.txt -o requirements.txt
python -m pip install -r requirements.txt
```

This leaves the local `.env` untouched. Alternatively clone into a fresh directory with `git clone https://github.com/Verlopat/test.git`, then copy your private `.env` into the clone.

## Secret safety

`.gitignore` excludes `.env`, virtual environments and test output. Do not put credentials in source code, screenshots, chat messages or Git commits.
