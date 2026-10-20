"""Apply ONLY the 19 safe duplicate deletions from migration_plan.json.

Default is DRY RUN. Use --apply to mutate Chroma.
This script deliberately does NOT touch the 4 Perpres replacement candidates
or the other review IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    _open_vectorstore,
    get_embedding_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="migration_plan.json")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"Migration plan not found: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    ids = list(plan.get("delete_ids", []))
    if len(ids) != 19:
        raise ValueError(
            f"Expected exactly 19 safe-delete IDs, got {len(ids)}")

    model = get_embedding_model()
    vs = _open_vectorstore(
        model, DEFAULT_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME)
    collection = vs._collection

    before = collection.count()
    existing = collection.get(ids=ids, include=["metadatas"])
    found_ids = set(existing.get("ids", []))
    missing = [x for x in ids if x not in found_ids]

    print("=" * 88)
    print("SAFE CHROMA CLEANUP — 19 DUPLICATE NONE-METADATA RECORDS ONLY")
    print("=" * 88)
    print(f"Collection before : {before}")
    print(f"Plan IDs          : {len(ids)}")
    print(f"Found in Chroma   : {len(found_ids)}")
    print(f"Already missing   : {len(missing)}")

    if missing:
        print("Missing IDs:")
        for x in missing:
            print("  ", x)
        raise RuntimeError(
            "Refusing mutation because the migration plan does not match Chroma exactly.")

    bad_metadata = []
    for rid, meta in zip(existing["ids"], existing["metadatas"]):
        # Safe-delete records must be the incomplete/None-metadata copies.
        if any(meta.get(k) not in (None, "") for k in ("document_type", "regulation_number", "year")):
            bad_metadata.append((rid, meta))

    if bad_metadata:
        print("Refusing mutation: a supposed safe-delete record has complete identity metadata:")
        for rid, meta in bad_metadata:
            print("  ", rid, {k: meta.get(k) for k in (
                "document_type", "regulation_number", "year")})
        raise RuntimeError("Safe-delete validation failed.")

    if not args.apply:
        print("\nDRY RUN — no Chroma mutation performed.")
        print("Run with --apply only after reviewing migration_plan.json.")
        return

    collection.delete(ids=ids)
    after = collection.count()
    print("\nDELETE APPLIED")
    print(f"Collection after  : {after}")
    print(f"Deleted            : {before - after}")
    if after != before - 19:
        raise RuntimeError(
            f"Unexpected final count: expected {before - 19}, got {after}")
    print("STATUS: PASS — exactly 19 safe duplicate records removed.")


if __name__ == "__main__":
    main()
