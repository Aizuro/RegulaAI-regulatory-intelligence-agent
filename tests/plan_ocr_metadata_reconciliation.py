"""Read-only targeted reconciliation for stale OCR Pasal metadata.

Finds the remaining Chroma records containing known OCR Pasal values
(168, 2O, 2l, 3O, 3OA), compares them with current parser output by
normalized content, and proposes only metadata corrections when an exact
content match exists in the current parser.

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

OUT = Path("ocr_metadata_reconciliation_plan.json")
BAD = {"168", "2O", "2l", "3O", "3OA"}


def sha(text: str) -> str:
    return hashlib.sha256(
        " ".join(str(text or "").split()).encode("utf-8")
    ).hexdigest()


def identity(m):
    keys = (
        "source", "document_type", "regulation_number", "year",
        "pasal", "ayat", "page", "page_end",
        "target_pasal", "block_type",
    )
    return {k: m.get(k) for k in keys}


def main():
    expected_docs, _ = build_legal_documents()

    model = get_embedding_model()
    vs = _open_vectorstore(
        model,
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )
    collection = vs._collection
    data = collection.get(include=["documents", "metadatas"])

    rows = [
        {"id": i, "content": c or "", "metadata": m or {}}
        for i, c, m in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        )
    ]

    expected_by_sha = {}
    for doc in expected_docs:
        expected_by_sha.setdefault(sha(doc.page_content), []).append(doc)

    stale = []
    for row in rows:
        m = row["metadata"]
        if str(m.get("pasal")) in BAD or str(m.get("target_pasal")) in BAD:
            stale.append(row)

    print("=" * 100)
    print("OCR PASAL METADATA RECONCILIATION — READ ONLY")
    print("=" * 100)
    print(f"Chroma vectors       : {len(rows)}")
    print(f"Expected parser docs : {len(expected_docs)}")
    print(f"Stale OCR records    : {len(stale)}")
    print()

    plan = []

    for i, row in enumerate(stale, 1):
        m = row["metadata"]
        matches = expected_by_sha.get(sha(row["content"]), [])

        # Only exact content matches are eligible for an UPDATE proposal.
        # If there are multiple parser matches, do not guess.
        if len(matches) == 1:
            expected = matches[0]
            em = expected.metadata

            # Compare all metadata fields; identify precisely what changes.
            fields = (
                "source", "document_type", "regulation_number", "year",
                "pasal", "ayat", "page", "page_end",
                "target_pasal", "block_type",
            )
            changes = {
                field: {
                    "from": m.get(field),
                    "to": em.get(field),
                }
                for field in fields
                if m.get(field) != em.get(field)
            }

            if changes:
                action = "UPDATE_METADATA_CANDIDATE"
                reason = "Exact normalized content match to current parser; metadata differs."
            else:
                action = "NO_CHANGE_NEEDED"
                reason = "Exact parser match and metadata already agrees."
        elif len(matches) == 0:
            action = "REVIEW_REQUIRED"
            reason = "No exact normalized-content match in current parser; do not guess metadata."
            expected = None
            changes = {}
        else:
            action = "REVIEW_REQUIRED"
            reason = f"{len(matches)} exact parser matches; ambiguous."
            expected = None
            changes = {}

        item = {
            "action": action,
            "reason": reason,
            "chroma_id": row["id"],
            "current": identity(m),
            "content_sha": sha(row["content"]),
            "content_chars": len(row["content"]),
            "content": row["content"],
            "changes": changes,
            "parser_match_count": len(matches),
            "parser_match": identity(expected.metadata) if expected else None,
        }
        plan.append(item)

        print(f"[{i}] {action}")
        print(f"    id      : {row['id']}")
        print(
            f"    current : pasal={m.get('pasal')} "
            f"ayat={m.get('ayat')} target={m.get('target_pasal')} "
            f"page={m.get('page')} source={m.get('source')}"
        )
        print(f"    parser exact matches: {len(matches)}")
        if changes:
            print("    proposed metadata changes:")
            for field, change in changes.items():
                print(f"      {field}: {change['from']!r} -> {change['to']!r}")
        elif expected:
            print("    metadata: already agrees with parser")
        print()

    update_candidates = [
        x for x in plan if x["action"] == "UPDATE_METADATA_CANDIDATE"
    ]
    review = [x for x in plan if x["action"] == "REVIEW_REQUIRED"]

    output = {
        "status": "REVIEW_REQUIRED",
        "mutation_performed": False,
        "chroma_vectors": len(rows),
        "expected_documents": len(expected_docs),
        "stale_ocr_records": len(stale),
        "update_metadata_candidates": len(update_candidates),
        "review_required": len(review),
        "items": plan,
    }

    OUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"UPDATE_METADATA candidates : {len(update_candidates)}")
    print(f"REVIEW_REQUIRED            : {len(review)}")
    print(f"Wrote                      : {OUT.resolve()}")
    print("STATUS                     : REVIEW_REQUIRED — NO CHROMA MUTATION")
    print("=" * 100)


if __name__ == "__main__":
    main()
