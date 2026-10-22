# Regula — Regulatory Intelligence Agent

Regula is an AI application for exploring Indonesian regulations. It combines retrieval-augmented generation (RAG), agent tool-calling, document comparison, evidence-grounded impact analysis, read-only structured-data queries, and controlled discovery of official regulatory sources.

## Contents

- [Overview](#overview)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Repository layout](#repository-layout)
- [Environment setup](#environment-setup)
- [Configuration](#configuration)
- [Run the application](#run-the-application)
- [Run tests](#run-tests)
- [Monitoring workflow](#monitoring-workflow)
- [Troubleshooting](#troubleshooting)
- [Known limitations and safety boundaries](#known-limitations-and-safety-boundaries)
- [Next steps](#next-steps)

## Overview

Regula is designed to help users find and inspect regulatory evidence without treating an LLM response as a substitute for the underlying legal source. Its workflow distinguishes internal document evidence, external web evidence, and structured SQL results.

The project includes a Streamlit interface and a FastAPI backend. The application also has a monitoring workflow that can be orchestrated with self-hosted n8n.

## Capabilities

- **Regulatory RAG:** retrieves relevant passages from an indexed collection of Indonesian regulations.
- **Metadata-aware retrieval:** uses available regulation identifiers and structural metadata such as chapter, article (*Pasal*), and clause (*Ayat*), with semantic retrieval as a fallback.
- **Agentic tool calling:** routes suitable questions to tools for retrieval, exact regulation lookup, document comparison, read-only SQL, external source discovery, and impact analysis.
- **Document comparison:** aligns provisions and reports matched, missing, or textually different provisions as evidence for further review.
- **Evidence-grounded impact analysis:** uses comparison evidence and can return an unknown/uncertain result when evidence is insufficient.
- **Controlled external research:** discovers sources from configured official-source allowlists and distinguishes a search result from text actually fetched from a source.
- **Regulatory monitoring:** checks a configured JDIH feed, detects changes using fingerprints, persists state, and can send alerts through Telegram.
- **Redis response caching:** caches repeat requests when Redis is configured and available.
- **API and automated tests:** includes FastAPI endpoints and tests for retrieval, parsing, monitoring, caching, comparison, impact analysis, and evidence guardrails.

Capabilities depend on the current configuration, available data, API credentials, and the specific runtime environment.

## Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
FastAPI API
  |
  v
Agent orchestrator
  |-- Regulatory retrieval / RAG
  |-- Exact regulation lookup
  |-- Document comparison
  |-- Read-only SQL
  |-- Allowlisted external research
  `-- Evidence-grounded impact analysis

Monitoring path:
JDIH feed -> n8n workflow -> FastAPI/Python monitoring logic
           -> change detection + persisted state
           -> analysis -> Telegram notification (when configured)
```

The Streamlit UI and FastAPI backend may be deployed separately. Confirm the configured API base URL and hosting arrangement for the deployment you are working on; local development paths are not automatically valid production URLs.

## Technology stack

- Python
- Streamlit
- FastAPI
- LangChain integrations
- ChromaDB
- Redis (optional cache)
- SQLite (structured persistence / project data)
- n8n (monitoring orchestration)
- Pytest
- Docker (for self-hosted services such as n8n and Redis, if used)

Exact versions are controlled by `requirements.txt` and the deployment environment. Do not upgrade core dependencies casually: validate imports and tests after dependency changes.

## Repository layout

```text
RegulaAI/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project/tool configuration
├── pytest.ini              # Pytest configuration
├── TESTING.md              # Testing notes
├── src/
│   ├── api.py              # FastAPI application
│   ├── agent_orchestrator.py
│   ├── llm_chain.py
│   ├── rag_chain.py
│   ├── embeddings.py
│   ├── document_retrieval.py
│   ├── document_comparison.py
│   ├── comparison_engine.py
│   ├── impact_analysis.py
│   ├── external_research.py
│   ├── monitoring*.py
│   ├── cache.py
│   └── redis_client.py
├── tests/                  # Unit, integration, and diagnostic tests
├── n8n/                    # Exported monitoring workflows
└── vectorstore/             # Local Chroma data (environment-specific)
```

Some scripts under `tests/` are diagnostic, migration, audit, or live-integration utilities rather than ordinary unit tests. Review a script before running it, especially if its name includes `apply`, `migrate`, `cleanup`, `reconcile`, or `sync`.

## Environment setup

### 1. Use the intended project folder

There have been multiple local copies of this project. Confirm which one you are working in before installing packages or editing code.

```powershell
# Example: deployment copy
cd C:\Project\Streamlit\RegulaAI

Get-Location
```

The path shown in a traceback may be a different copy, such as `C:\Project\RegulaAI`. Changes made in one folder do not automatically update another folder or the GitHub repository.

### 2. Check installed Python versions

```powershell
py -0p
```

Python 3.11 is a reasonable starting point for this project, subject to compatibility with the pinned dependencies. The active interpreter must be the one inside the project's virtual environment.

### 3. Create or activate a virtual environment

If `.venv` already exists and you intend to reuse it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the active interpreter:

```powershell
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

The executable should point to the intended project's `.venv\Scripts\python.exe`, not a global Python or an Anaconda installation.

If you need to create a new environment, first preserve the old one if you may need it:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Do not create a new `.venv` over an active environment. If replacing an existing environment, deactivate it and rename or remove the old `.venv` only after confirming that you no longer need it.

### 4. Install dependencies

With the intended `.venv` activated:

```powershell
python -m pip install -r requirements.txt
```

Do not proceed to application testing if this command fails. Resolve the first dependency-resolution or installation error, then rerun the command and confirm it completes.

After successful dependency installation, install pytest only if it is not already included:

```powershell
python -m pip install pytest
python -m pytest --version
```

**Important:** `python -m pip install pytest` installs the test runner, not the application's dependencies. It will not install missing packages such as LangChain, FastAPI, or Redis unless they are separately specified and installed.

## Configuration

Create or update a local `.env` file only if the application requires it. Use the variable names referenced in the source code and hosting configuration.

Typical configuration categories include:

- LLM provider API key and model settings
- Embedding provider configuration
- FastAPI base URL used by the Streamlit frontend
- Redis connection URL and cache settings
- Monitoring feed/source configuration
- Telegram bot token and destination/chat configuration

The exact variable names and defaults must be confirmed in the current source code. Never commit `.env`, API keys, bot tokens, or other credentials. If a credential has been exposed, rotate it.

## Run the application

The entry points present in the repository include `app.py` for Streamlit and `src/api.py` for FastAPI. Confirm the application's configured launch command and required environment variables before running it.

Example local commands, from the project root:

```powershell
# Streamlit UI
python -m streamlit run app.py
```

For FastAPI, first inspect `src/api.py` and confirm the exported ASGI application variable. If it is named `app`, a common local command is:

```powershell
python -m uvicorn src.api:app --reload
```

These are example commands; use the module path and app variable actually defined in the checked-out code. If the UI and API are separate, start them in separate terminals and configure the UI to point at the correct API URL.

## Run tests

Always run tests through the active interpreter:

```powershell
python -m pytest
```

Run a focused test first when debugging a specific feature, for example:

```powershell
python -m pytest tests/test_v81_document_retrieval.py -q
python -m pytest tests/test_v82_document_comparison.py -q
python -m pytest tests/test_v83_impact_analysis.py -q
python -m pytest tests/test_v84_agent_integration.py -q
```

Use only test paths that exist in the current checkout. Some tests may require local data, environment variables, a running Redis service, network access, or API credentials. Read `TESTING.md` and the individual test before running live or state-changing checks.

A previous development run reported **126 passed, 10 skipped, 0 failed**, but that result applies only to the code and environment at the time of that run. It is not a claim that the current checkout passes. Record the current date, command, pass/fail/skip counts, and any required external services when validating a release.

## Monitoring workflow

The repository includes exported n8n workflow files under `n8n/`. Import a workflow into the intended n8n instance only after reviewing its nodes, endpoints, credentials, and environment-specific URLs.

The monitoring path is intended to:
1. Fetch a configured regulatory feed.
2. Normalize entries and calculate deterministic fingerprints.
3. Compare the current feed with persisted state to identify new, updated, or removed feed entries.
4. Persist the new state.
5. Run analysis when configured.
6. Send notifications for relevant events when credentials and notification settings are available.

A feed entry being removed is not, by itself, proof that a regulation has been legally revoked. Monitoring output is a signal for investigation, not a substitute for official legal verification.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain'` (or `fastapi`, `redis`, etc.)

This usually means dependencies were not installed in the interpreter running the command, or the wrong Python environment is active.

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip show langchain fastapi redis
```

Both the interpreter and pip should point into the same project `.venv`. Install `requirements.txt` using that interpreter after resolving any installation errors.

### `python -m pytest` returns `No module named pytest`

Install pytest in the active environment:

```powershell
python -m pip install pytest
python -m pytest --version
```

Then install the project dependencies from `requirements.txt` if that has not completed successfully.

### `pip install -r requirements.txt` fails on `redis==8.1.0`

Do not change the pin blindly. First collect diagnostic information:

```powershell
python --version
python -m pip --version
python -m pip config list
python -m pip index versions redis
```

Python 3.11 satisfies the Python-version requirement reported for Redis 8.x, so if pip only lists older releases, check which package index or mirror pip is using and whether that index has the required release. Do not expose private index credentials when sharing the output. Only adjust `requirements.txt` after confirming package availability and compatibility with the project's Redis client usage.

### `ModuleNotFoundError: No module named 'llm_chain'`

This is different from a missing third-party dependency. It is a local import-path issue. The repository contains `src/llm_chain.py`; use import paths consistently with the project's package layout and how the application is launched. Before changing imports, compare the exact file and command in the folder that produced the traceback. There may be multiple local copies with different code.

### Tests report many missing modules during collection

A large number of collection errors can share one root cause: pytest is running with an interpreter that lacks the project's dependencies. Check `sys.executable`, install dependencies into that environment, and rerun `python -m pytest` before treating the collection errors as application bugs.

### Redis connection errors

Installing the Python `redis` package does not start a Redis server. If caching tests or the API require a running server, verify the configured host/port and service availability separately. Follow the project's current Redis configuration; do not assume that a successful package install means Redis is reachable.

### Line-ending warning: LF will be replaced by CRLF

This is commonly a Git/Windows line-ending notice, not necessarily a code error. Do not mass-convert source files merely to remove the warning. Review `git diff --check` and the actual code changes separately.

## Known limitations and safety boundaries

- External search snippets are discovery evidence, not verified legal text.
- A citation should only be treated as support for a claim when the fetched source contains relevant evidence.
- Missing or textually different provisions do not independently prove that a regulation has been revoked, newly effective, or substantively amended.
- Impact analysis should be treated as uncertain when the available evidence does not support a reliable conclusion.
- SQL output and legal source evidence are distinct; structured or synthetic datasets must not be presented as authoritative legal text.
- The application is an informational research aid, not legal advice. Users should verify material conclusions against the official regulation and consult a qualified professional when appropriate.

## Next steps

Use this sequence when resuming development on a new machine or project copy:

1. **Confirm the folder:** run `Get-Location` and verify the repository you intend to work on.
2. **Confirm the environment:** verify `python --version`, `sys.executable`, and `python -m pip --version`.
3. **Finish dependency installation:** run `python -m pip install -r requirements.txt` and address the first error without changing pins prematurely.
4. **Install and verify pytest:** run `python -m pip install pytest` if needed, then `python -m pytest --version`.
5. **Run a focused test:** start with a test related to the feature being changed.
6. **Run the full suite:** use `python -m pytest` and record the actual results.
7. **Smoke-test the app:** verify the API and Streamlit UI with the correct environment variables and URLs.
8. **Review Git changes:** use `git status --short`, `git diff --stat`, and focused `git diff -- <file>` before staging or committing. Do not use `git add .` blindly when the working tree contains unrelated or unreviewed files.
9. **Keep deployment copies in sync deliberately:** commit and push reviewed changes to GitHub, then confirm the intended deployment service is using the updated commit.

---

For development notes, current test commands, and test prerequisites, also see [`TESTING.md`](TESTING.md).
