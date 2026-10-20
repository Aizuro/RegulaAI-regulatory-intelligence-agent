"""Read-only reconciliation of the 4 known Perpres outliers.

Uses the exact expected content SHA values from migration_plan.json, avoiding
ambiguous (source, page, pasal, ayat) matching for amendment-instruction chunks.
No Chroma mutation is performed.
"""
from __future__ import annotations

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

PLAN_PATH = Path("migration_plan.json")
OUT_PATH = Path("perpres_reconciliation_plan.json")


def normalize_sha(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def meta(m):
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


def main():
    if not PLAN_PATH.exists():
        raise FileNotFoundError("migration_plan.json tidak ditemukan.")

    migration = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    candidates = migration.get("replace_candidates", [])

    # The previous migration plan is authoritative for which four records
    # were classified as structural outliers.
    targets = []
    for item in candidates:
        e = item.get("expected")
        if not e:
            continue
        targets.append(e)

    # De-duplicate by exact expected content SHA.
    unique = {}
    for e in targets:
        unique[e["content_sha"]] = e
    targets = list(unique.values())

    expected_docs, _ = build_legal_documents()

    model = get_embedding_model()
    vs = _open_vectorstore(
        model,
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )
    collection = vs._collection
    existing = collection.get(include=["documents", "metadatas"])

    chroma = [
        {"id": i, "content": d or "", "metadata": m or {}}
        for i, d, m in zip(
            existing.get("ids", []),
            existing.get("documents", []),
            existing.get("metadatas", []),
        )
    ]

    by_sha = {}
    for row in chroma:
        by_sha.setdefault(normalize_sha(row["content"]), []).append(row)

    # Locate exact expected records from current parser too, to verify that
    # migration_plan.json still agrees with the current parser output.
    current_expected_by_sha = {
        normalize_sha(doc.page_content): doc
        for doc in expected_docs
        if doc.metadata.get("source") == "2025perpres004.pdf"
    }

    print("=" * 96)
    print("PERPRES 4/2025 OUTLIER RECONCILIATION — CORRECTED, READ ONLY")
    print("=" * 96)
    print(f"Chroma vectors     : {len(chroma)}")
    print(f"Expected documents : {len(expected_docs)}")
    print(f"Outlier targets    : {len(targets)}")
    print()

    result_items = []

    for idx, e in enumerate(targets, 1):
        sha_value = e["content_sha"]
        exact = by_sha.get(sha_value, [])
        current_doc = current_expected_by_sha.get(sha_value)

        # Page records are useful context, but are never treated as replacement
        # candidates unless their content SHA is identical.
        page_records = [
            r for r in chroma
            if r["metadata"].get("source") == e.get("source")
            and r["metadata"].get("page") == e.get("page")
        ]

        if exact:
            action = "ALREADY_PRESENT"
            reason = "Exact normalized content already exists in Chroma."
        else:
            action = "ADD_ONLY_CANDIDATE"
            reason = "No exact normalized-content match exists; page-level candidates are different chunks and must be kept."

        item = {
            "action": action,
            "reason": reason,
            "expected": e,
            "current_parser_match": bool(current_doc),
            "exact_matches": [
                {
                    "id": r["id"],
                    **meta(r["metadata"]),
                    "content_sha": normalize_sha(r["content"]),
                    "content_chars": len(r["content"]),
                }
                for r in exact
            ],
            "same_page_records": [
                {
                    "id": r["id"],
                    **meta(r["metadata"]),
                    "content_sha": normalize_sha(r["content"]),
                    "content_chars": len(r["content"]),
                }
                for r in page_records
            ],
        }
        result_items.append(item)

        print(f"[{idx}] {action}")
        print(
            f"    page={e.get('page')} pasal={e.get('pasal')} "
            f"ayat={e.get('ayat')} target={e.get('target_pasal')} "
            f"block={e.get('block_type')} chars={e.get('content_chars')}"
        )
        print(f"    exact matches: {len(exact)}")
        print(f"    same-page records: {len(page_records)}")
        for r in page_records:
            print(
                f"      - {r['id']} | pasal={r['metadata'].get('pasal')} "
                f"ayat={r['metadata'].get('ayat')} "
                f"target={r['metadata'].get('target_pasal')} "
                f"block={r['metadata'].get('block_type')} "
                f"chars={len(r['content'])}"
            )
        print()

    add_only = sum(x["action"] == "ADD_ONLY_CANDIDATE" for x in result_items)
    already = sum(x["action"] == "ALREADY_PRESENT" for x in result_items)

    output = {
        "status": "REVIEW_REQUIRED",
        "mutation_performed": False,
        "chroma_vectors": len(chroma),
        "expected_documents": len(expected_docs),
        "outlier_targets": len(result_items),
        "add_only_candidates": add_only,
        "already_present": already,
        "items": result_items,
    }

    OUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 96)
    print(f"ADD_ONLY candidates : {add_only}")
    print(f"Already present     : {already}")
    print(f"Wrote               : {OUT_PATH.resolve()}")
    print("STATUS              : REVIEW_REQUIRED — NO CHROMA MUTATION")
    print("=" * 96)


if __name__ == "__main__":
    main()
