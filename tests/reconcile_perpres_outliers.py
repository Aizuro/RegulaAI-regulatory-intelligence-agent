"""Read-only reconciliation for the 4 Perpres 4/2025 structural outliers.

This script does NOT mutate Chroma.
It:
1. Rebuilds the current 776 expected legal documents.
2. Loads the current Chroma collection.
3. Locates the four known Perpres outlier expectations.
4. Searches Chroma by source/page and reports all records on those pages.
5. Produces a conservative reconciliation plan:
   - ADD_ONLY when no existing record has the same content.
   - REVIEW when an existing record is a different chunk/content.
No DELETE/UPDATE is proposed automatically.
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

TARGETS = [
    ("2025perpres004.pdf", 14, "20", "1"),
    ("2025perpres004.pdf", 14, None, None),
    ("2025perpres004.pdf", 16, "30", "1"),
    ("2025perpres004.pdf", 17, None, None),
]


def sha(text: str) -> str:
    return hashlib.sha256(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()


def meta_identity(m):
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
    expected_docs, reports = build_legal_documents()

    model = get_embedding_model()
    vs = _open_vectorstore(
        model,
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )
    collection = vs._collection
    existing = collection.get(include=["documents", "metadatas"])

    ids = existing.get("ids", [])
    docs = existing.get("documents", [])
    metas = existing.get("metadatas", [])
    chroma = [
        {"id": i, "content": d or "", "metadata": m or {}}
        for i, d, m in zip(ids, docs, metas)
    ]

    expected = []
    for doc in expected_docs:
        m = doc.metadata
        source = m.get("source")
        page = m.get("page")
        pasal = m.get("pasal")
        ayat = m.get("ayat")
        if (source, page, pasal, ayat) in TARGETS:
            expected.append({
                "metadata": dict(m),
                "content": doc.page_content,
                "content_sha": sha(doc.page_content),
            })

    print("=" * 96)
    print("PERPRES 4/2025 OUTLIER RECONCILIATION — READ ONLY")
    print("=" * 96)
    print(f"Chroma vectors     : {len(ids)}")
    print(f"Expected documents : {len(expected_docs)}")
    print(f"Target outliers    : {len(expected)}")
    print()

    # Content-exact matches across the entire collection.
    sha_to_existing = {}
    for row in chroma:
        sha_to_existing.setdefault(sha(row["content"]), []).append(row)

    plan = []
    for e in expected:
        em = e["metadata"]
        exact = sha_to_existing.get(e["content_sha"], [])

        same_page = [
            row for row in chroma
            if row["metadata"].get("source") == em.get("source")
            and row["metadata"].get("page") == em.get("page")
        ]

        if exact:
            action = "ALREADY_PRESENT"
            reason = "Exact normalized content already exists in Chroma; inspect metadata before changing anything."
        else:
            action = "ADD_ONLY_CANDIDATE"
            reason = "No exact normalized-content match exists; existing page records are different chunks, so do not delete them automatically."

        item = {
            "action": action,
            "reason": reason,
            "expected": {
                **meta_identity(em),
                "content_sha": e["content_sha"],
                "content_chars": len(e["content"]),
                "content": e["content"],
            },
            "exact_matches": [
                {
                    "id": r["id"],
                    **meta_identity(r["metadata"]),
                    "content_sha": sha(r["content"]),
                    "content_chars": len(r["content"]),
                }
                for r in exact
            ],
            "same_page_records": [
                {
                    "id": r["id"],
                    **meta_identity(r["metadata"]),
                    "content_sha": sha(r["content"]),
                    "content_chars": len(r["content"]),
                    "content": r["content"],
                }
                for r in same_page
            ],
        }
        plan.append(item)

    print("RESULTS")
    print("-" * 96)
    for i, item in enumerate(plan, 1):
        e = item["expected"]
        print(f"[{i}] {item['action']}")
        print(
            f"    source={e['source']} page={e['page']} "
            f"pasal={e['pasal']} ayat={e['ayat']} chars={e['content_chars']}"
        )
        print(f"    exact matches : {len(item['exact_matches'])}")
        print(f"    same-page records : {len(item['same_page_records'])}")
        for r in item["same_page_records"]:
            print(
                f"      - {r['id']} | pasal={r['pasal']} ayat={r['ayat']} "
                f"target={r['target_pasal']} block={r['block_type']} "
                f"chars={r['content_chars']}"
            )
        print()

    add_only = sum(x["action"] == "ADD_ONLY_CANDIDATE" for x in plan)
    already = sum(x["action"] == "ALREADY_PRESENT" for x in plan)

    result = {
        "status": "REVIEW_REQUIRED",
        "mutation_performed": False,
        "chroma_vectors": len(ids),
        "expected_documents": len(expected_docs),
        "targets_found": len(expected),
        "add_only_candidates": add_only,
        "already_present": already,
        "plan": plan,
    }

    OUT_PATH.write_text(json.dumps(
        result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 96)
    print(f"ADD_ONLY candidates : {add_only}")
    print(f"Already present    : {already}")
    print(f"Wrote              : {OUT_PATH.resolve()}")
    print("STATUS             : REVIEW_REQUIRED — NO CHROMA MUTATION")
    print("=" * 96)


if __name__ == "__main__":
    main()
