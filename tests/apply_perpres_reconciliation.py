"""Apply ONLY the 4 verified missing Perpres 4/2025 chunks.

Safety:
- Requires perpres_reconciliation_plan.json.
- Requires exactly 4 ADD_ONLY_CANDIDATE items.
- Verifies exact normalized content is absent immediately before mutation.
- Adds only those 4 documents.
- Does not delete or update any existing Chroma record.
- Verifies the collection count increases by exactly 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.legal_ingestion import build_legal_documents
from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
)

PLAN_PATH = Path("perpres_reconciliation_plan.json")


def content_sha(text: str) -> str:
    return hashlib.sha256(
        " ".join(str(text or "").split()).encode("utf-8")
    ).hexdigest()


def meta(doc):
    m = doc.metadata
    return {
        "source": m.get("source"),
        "document_type": m.get("document_type"),
        "regulation_number": m.get("regulation_number"),
        "year": m.get("year"),
        "pasal": m.get("pasal"),
        "ayat": m.get("ayat"),
        "page": m.get("page"),
        "page_end": m.get("page_end"),
        "target_pasal": m.get("target_pasal"),
        "block_type": m.get("block_type"),
    }


def make_doc_key(doc):
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("page_end"),
        doc.metadata.get("pasal"),
        doc.metadata.get("ayat"),
        doc.metadata.get("target_pasal"),
        doc.metadata.get("block_type"),
        content_sha(doc.page_content),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            "perpres_reconciliation_plan.json tidak ditemukan."
        )

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    items = plan.get("items", [])

    candidates = [
        item for item in items
        if item.get("action") == "ADD_ONLY_CANDIDATE"
    ]

    print("=" * 96)
    print("PERPRES 4/2025 — APPLY 4 VERIFIED MISSING CHUNKS")
    print("=" * 96)

    if len(candidates) != 4:
        raise RuntimeError(
            f"Safety check gagal: expected tepat 4 ADD_ONLY_CANDIDATE, "
            f"ditemukan {len(candidates)}."
        )

    expected_docs, _ = build_legal_documents()

    # Index current parser output by normalized content SHA.
    expected_by_sha = {}
    for doc in expected_docs:
        if doc.metadata.get("source") != "2025perpres004.pdf":
            continue
        expected_by_sha.setdefault(content_sha(
            doc.page_content), []).append(doc)

    docs_to_add = []
    for item in candidates:
        e = item["expected"]
        sha = e["content_sha"]
        matches = expected_by_sha.get(sha, [])

        if len(matches) != 1:
            raise RuntimeError(
                f"Safety check gagal untuk SHA {sha}: "
                f"ditemukan {len(matches)} parser document(s), expected 1."
            )

        doc = matches[0]

        # Verify metadata agrees with the migration plan.
        for field in (
            "source", "document_type", "regulation_number", "year",
            "pasal", "ayat", "page", "page_end",
            "target_pasal", "block_type",
        ):
            expected_value = e.get(field)
            actual_value = doc.metadata.get(field)
            if expected_value != actual_value:
                raise RuntimeError(
                    f"Metadata mismatch pada {sha}, field={field}: "
                    f"plan={expected_value!r}, parser={actual_value!r}"
                )

        docs_to_add.append(doc)

    model = get_embedding_model()
    vs = _open_vectorstore(
        model,
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )
    collection = vs._collection

    before = collection.count()
    existing = collection.get(include=["documents", "metadatas"])
    existing_shas = {
        content_sha(d or "")
        for d in existing.get("documents", [])
    }

    print(f"Collection before : {before}")
    print(f"Verified additions: {len(docs_to_add)}")
    print()

    # Critical duplicate guard.
    duplicate_shas = [
        content_sha(doc.page_content)
        for doc in docs_to_add
        if content_sha(doc.page_content) in existing_shas
    ]
    if duplicate_shas:
        raise RuntimeError(
            "Safety check gagal: satu atau lebih target content SHA sudah "
            f"ada di Chroma: {duplicate_shas}"
        )

    # Also ensure the four additions are distinct.
    add_shas = [content_sha(doc.page_content) for doc in docs_to_add]
    if len(set(add_shas)) != 4:
        raise RuntimeError("Safety check gagal: 4 target tidak unik.")

    print("TARGETS")
    for doc in docs_to_add:
        print(
            f"  + {meta(doc)} | chars={len(doc.page_content)} "
            f"| sha={content_sha(doc.page_content)[:16]}..."
        )

    if not args.apply:
        print()
        print("DRY RUN — no Chroma mutation performed.")
        print("Run with --apply only after reviewing the four targets above.")
        return

    # Only mutation in this script: add exactly four documents.
    vs.add_documents(docs_to_add)

    after = collection.count()
    delta = after - before

    print()
    print("ADD APPLIED")
    print(f"Collection after  : {after}")
    print(f"Added             : {delta}")

    if delta != 4:
        raise RuntimeError(
            f"Post-check gagal: expected count increase 4, actual {delta}."
        )

    # Verify all four content SHAs are now present.
    final = collection.get(include=["documents"])
    final_shas = {
        content_sha(d or "")
        for d in final.get("documents", [])
    }
    missing = [s for s in add_shas if s not in final_shas]

    if missing:
        raise RuntimeError(
            f"Post-check gagal: {len(missing)} added SHA tidak ditemukan."
        )

    print("Verified added SHAs : 4/4")
    print("No DELETE operation performed.")
    print("No UPDATE operation performed.")
    print("STATUS: PASS — exactly 4 verified missing chunks added.")


if __name__ == "__main__":
    main()
