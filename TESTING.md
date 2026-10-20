# Testing Regula

## 1. Routine automated tests

Run from the project root:

```bash
python -m pytest
```

`pytest.ini` adds the project root to Python's import path and excludes four
legacy evaluation/audit scripts that depend on the local production Chroma
collection or execute as standalone programs rather than normal pytest tests.
They are not deleted; run them explicitly when the local vectorstore and API
credentials are configured.

## 2. Production vectorstore audits/evaluations

Run only when you intentionally want to inspect the local vectorstore:

```bash
python tests/test_metadata_audit.py
python tests/test_legal_tools.py
python tests/test_retrieval.py
python tests/test_rag.py
```

These commands may load the local Chroma collection and, for the RAG evaluation,
call the configured LLM provider. They should not be part of every routine
unit-test run.

## 3. Interpreting vector counts

Do not treat a fixed historical count (such as 776) as proof of correctness.
The collection may legitimately grow as the corpus is updated. A non-empty
count is only a sanity check; the metadata audit and retrieval checks are what
help assess data quality. If duplicate ingestion is suspected, audit stable
document IDs, duplicate content/metadata, and source-level coverage before
deleting or rebuilding vectors.

## 4. Import conventions

Prefer package imports such as `from src.llm_chain import ...` when running
from the repository root. `src/rag_chain.py` and `src/legal_indexer.py` retain
fallback imports for the older `src/`-on-`PYTHONPATH` execution style.
