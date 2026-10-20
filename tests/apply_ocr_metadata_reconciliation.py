"""
Apply ONLY the 8 verified OCR metadata corrections.

Safety:
- Requires ocr_metadata_reconciliation_plan.json
- Requires exactly 8 UPDATE_METADATA_CANDIDATE items
- Verifies every target ID exists in Chroma
- Verifies content SHA before mutation
- Updates metadata ONLY
- Does NOT delete vectors
- Does NOT add vectors
- Does NOT modify document content / embeddings
- Verifies all 8 records after mutation
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


PLAN_PATH = Path("ocr_metadata_reconciliation_plan.json")


def content_sha(text: str) -> str:
    """
    Generate the same normalized content SHA used by
    the reconciliation diagnostic.
    """
    normalized = " ".join(str(text or "").split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply verified OCR metadata corrections to Chroma."
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update Chroma metadata.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("APPLY OCR PASAL METADATA RECONCILIATION")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Load reconciliation plan
    # ------------------------------------------------------------------

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"{PLAN_PATH} tidak ditemukan."
        )

    plan = json.loads(
        PLAN_PATH.read_text(
            encoding="utf-8"
        )
    )

    items = plan.get("items", [])

    candidates = [
        item
        for item in items
        if item.get("action") == "UPDATE_METADATA_CANDIDATE"
    ]

    print(f"Plan items             : {len(items)}")
    print(f"Update candidates      : {len(candidates)}")

    # Critical safety check.
    if len(candidates) != 8:
        raise RuntimeError(
            "SAFETY CHECK FAILED: "
            f"expected exactly 8 update candidates, "
            f"found {len(candidates)}."
        )

    # ------------------------------------------------------------------
    # 2. Rebuild current parser output
    # ------------------------------------------------------------------

    print()
    print("Memuat current parser output...")

    expected_docs, _ = build_legal_documents()

    # Map normalized content SHA -> parser document.
    expected_by_sha = {}

    for doc in expected_docs:

        source = doc.metadata.get("source")

        # These corrections are specifically for Perpres 4/2025.
        if source != "2025perpres004.pdf":
            continue

        sha = content_sha(
            doc.page_content
        )

        expected_by_sha.setdefault(
            sha,
            []
        ).append(doc)

    # ------------------------------------------------------------------
    # 3. Connect to Chroma
    # ------------------------------------------------------------------

    print("Membuka Chroma...")

    model = get_embedding_model()

    vectorstore = _open_vectorstore(
        model,
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )

    collection = vectorstore._collection

    before_count = collection.count()

    print()
    print(f"Collection before : {before_count}")

    # ------------------------------------------------------------------
    # 4. Read current records
    # ------------------------------------------------------------------

    current = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    current_by_id = {
        record_id: {
            "content": content or "",
            "metadata": metadata or {},
        }
        for record_id, content, metadata
        in zip(
            current.get("ids", []),
            current.get("documents", []),
            current.get("metadatas", []),
        )
    }

    # ------------------------------------------------------------------
    # 5. Validate all eight candidates BEFORE mutation
    # ------------------------------------------------------------------

    updates = []

    print()
    print("=" * 100)
    print("PRE-MUTATION VALIDATION")
    print("=" * 100)

    for index, item in enumerate(
        candidates,
        start=1,
    ):

        chroma_id = item.get(
            "chroma_id"
        )

        expected = item.get(
            "parser_match"
        )

        expected_sha = item.get(
            "content_sha"
        )

        changes = item.get(
            "changes",
            {}
        )

        # --------------------------------------------------------------
        # ID must exist
        # --------------------------------------------------------------

        if chroma_id not in current_by_id:
            raise RuntimeError(
                f"SAFETY CHECK FAILED: "
                f"Chroma ID tidak ditemukan: {chroma_id}"
            )

        current_record = current_by_id[
            chroma_id
        ]

        current_content = current_record[
            "content"
        ]

        current_metadata = current_record[
            "metadata"
        ]

        # --------------------------------------------------------------
        # Content must remain exactly the expected content
        # --------------------------------------------------------------

        actual_sha = content_sha(
            current_content
        )

        if actual_sha != expected_sha:
            raise RuntimeError(
                "SAFETY CHECK FAILED: "
                f"content SHA mismatch untuk {chroma_id}\n"
                f"Expected: {expected_sha}\n"
                f"Actual  : {actual_sha}"
            )

        # --------------------------------------------------------------
        # Current parser must have exactly one matching document
        # --------------------------------------------------------------

        parser_matches = expected_by_sha.get(
            expected_sha,
            []
        )

        if len(parser_matches) != 1:
            raise RuntimeError(
                "SAFETY CHECK FAILED: "
                f"expected exactly 1 parser match untuk "
                f"{chroma_id}, ditemukan {len(parser_matches)}."
            )

        parser_doc = parser_matches[0]

        parser_metadata = dict(
            parser_doc.metadata
        )

        # --------------------------------------------------------------
        # Verify plan's proposed values against parser
        # --------------------------------------------------------------

        for field, change in changes.items():

            planned_to = change.get(
                "to"
            )

            parser_value = parser_metadata.get(
                field
            )

            if planned_to != parser_value:
                raise RuntimeError(
                    "SAFETY CHECK FAILED: "
                    f"plan/parser mismatch untuk {chroma_id}, "
                    f"field={field}: "
                    f"plan={planned_to!r}, "
                    f"parser={parser_value!r}"
                )

        updates.append(
            {
                "id": chroma_id,
                "content_sha": actual_sha,
                "old_metadata": dict(
                    current_metadata
                ),
                "new_metadata": parser_metadata,
                "changes": changes,
            }
        )

        print(
            f"[{index}/8] PASS | {chroma_id}"
        )

        for field, change in changes.items():

            print(
                f"      {field}: "
                f"{change.get('from')!r} "
                f"-> "
                f"{change.get('to')!r}"
            )

    print()
    print(
        "Pre-mutation validation: "
        "8/8 PASS"
    )

    # ------------------------------------------------------------------
    # 6. Dry run
    # ------------------------------------------------------------------

    if not args.apply:

        print()
        print("=" * 100)
        print("DRY RUN")
        print("=" * 100)

        print(
            "Tidak ada perubahan pada Chroma."
        )

        print()
        print(
            "Jika delapan perubahan di atas sudah sesuai, "
            "jalankan:"
        )

        print()
        print(
            "    py .\\apply_ocr_metadata_reconciliation.py --apply"
        )

        return

    # ------------------------------------------------------------------
    # 7. Apply metadata updates ONLY
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("APPLYING METADATA UPDATES")
    print("=" * 100)

    ids = []
    metadatas = []

    for update in updates:

        ids.append(
            update["id"]
        )

        metadatas.append(
            update["new_metadata"]
        )

    # IMPORTANT:
    # Chroma update() changes metadata only.
    # Existing document content and embeddings remain untouched.
    collection.update(
        ids=ids,
        metadatas=metadatas,
    )

    # ------------------------------------------------------------------
    # 8. Post-mutation count check
    # ------------------------------------------------------------------

    after_count = collection.count()

    print()
    print(f"Collection before : {before_count}")
    print(f"Collection after  : {after_count}")
    print(
        f"Count difference  : "
        f"{after_count - before_count}"
    )

    if after_count != before_count:
        raise RuntimeError(
            "POST-CHECK FAILED: "
            "jumlah vector berubah. "
            "Metadata reconciliation seharusnya tidak mengubah count."
        )

    # ------------------------------------------------------------------
    # 9. Re-read the eight records
    # ------------------------------------------------------------------

    final = collection.get(
        ids=ids,
        include=[
            "documents",
            "metadatas",
        ],
    )

    final_by_id = {
        record_id: {
            "content": content or "",
            "metadata": metadata or {},
        }
        for record_id, content, metadata
        in zip(
            final.get("ids", []),
            final.get("documents", []),
            final.get("metadatas", []),
        )
    }

    print()
    print("=" * 100)
    print("POST-MUTATION VERIFICATION")
    print("=" * 100)

    for update in updates:

        chroma_id = update["id"]

        if chroma_id not in final_by_id:
            raise RuntimeError(
                f"POST-CHECK FAILED: "
                f"{chroma_id} tidak ditemukan setelah update."
            )

        final_record = final_by_id[
            chroma_id
        ]

        # Verify content did not change.
        final_sha = content_sha(
            final_record["content"]
        )

        if final_sha != update["content_sha"]:
            raise RuntimeError(
                "POST-CHECK FAILED: "
                f"content berubah untuk {chroma_id}."
            )

        # Verify metadata now equals parser metadata.
        final_metadata = final_record[
            "metadata"
        ]

        expected_metadata = update[
            "new_metadata"
        ]

        for field, expected_value in expected_metadata.items():

            actual_value = final_metadata.get(
                field
            )

            if actual_value != expected_value:
                raise RuntimeError(
                    "POST-CHECK FAILED: "
                    f"metadata mismatch pada {chroma_id}, "
                    f"field={field}: "
                    f"expected={expected_value!r}, "
                    f"actual={actual_value!r}"
                )

        print(
            f"PASS | {chroma_id}"
        )

    # ------------------------------------------------------------------
    # 10. Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("STATUS: PASS")
    print("=" * 100)

    print(
        "8 metadata records successfully corrected."
    )

    print(
        "Vector count unchanged."
    )

    print(
        "Document content unchanged."
    )

    print(
        "Embeddings unchanged."
    )

    print(
        "No DELETE operation performed."
    )

    print(
        "No ADD operation performed."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
