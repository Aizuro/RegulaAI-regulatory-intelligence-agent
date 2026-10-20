"""Post-migration Chroma integrity audit (READ ONLY).

Checks the current 807-vector Chroma collection against the current parser/ingestion
output without mutating Chroma.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.legal_ingestion import build_legal_documents
from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
)

BAD_PASAL = {"168", "2O", "2l", "3O", "3OA"}


def sha(text: str) -> str:
    return hashlib.sha256(
        " ".join(str(text or "").split()).encode("utf-8")
    ).hexdigest()


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

    ids = data.get("ids", [])
    contents = data.get("documents", [])
    metas = data.get("metadatas", [])

    rows = [
        {"id": i, "content": c or "", "metadata": m or {}}
        for i, c, m in zip(ids, contents, metas)
    ]

    print("=" * 100)
    print("POST-MIGRATION CHROMA INTEGRITY AUDIT — READ ONLY")
    print("=" * 100)
    print(f"Chroma vectors       : {len(rows)}")
    print(f"Expected parser docs : {len(expected_docs)}")
    print(f"Difference           : {len(rows) - len(expected_docs)}")
    print()

    # 1. Bad OCR Pasal metadata.
    bad_pasal = [
        r for r in rows
        if str(r["metadata"].get("pasal")) in BAD_PASAL
        or str(r["metadata"].get("target_pasal")) in BAD_PASAL
    ]
    print(f"[1] Bad OCR Pasal metadata : {len(bad_pasal)}")
    for r in bad_pasal[:20]:
        m = r["metadata"]
        print(
            f"    {r['id']} | source={m.get('source')} page={m.get('page')} "
            f"pasal={m.get('pasal')} target={m.get('target_pasal')}"
        )

    # 2. Empty identity metadata, excluding UUD's intentionally absent regulation number.
    empty_identity = []
    for r in rows:
        m = r["metadata"]
        if not m.get("source"):
            empty_identity.append(r)
            continue
        if m.get("document_type") is None or m.get("year") is None:
            empty_identity.append(r)
        elif m.get("document_type") != "UUD" and m.get("regulation_number") is None:
            empty_identity.append(r)

    print(f"\n[2] Missing core identity metadata : {len(empty_identity)}")
    for r in empty_identity[:30]:
        m = r["metadata"]
        print(
            f"    {r['id']} | source={m.get('source')} "
            f"type={m.get('document_type')} reg={m.get('regulation_number')} "
            f"year={m.get('year')} page={m.get('page')}"
        )

    # 3. Exact duplicate content hashes.
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[sha(r["content"])].append(r)

    duplicate_groups = [group for group in by_sha.values() if len(group) > 1]
    duplicate_rows = sum(len(g) - 1 for g in duplicate_groups)
    print(
        f"\n[3] Duplicate content groups : {len(duplicate_groups)} "
        f"(extra rows={duplicate_rows})"
    )
    for group in duplicate_groups[:30]:
        print("    GROUP")
        for r in group:
            m = r["metadata"]
            print(
                f"      {r['id']} | source={m.get('source')} page={m.get('page')} "
                f"pasal={m.get('pasal')} ayat={m.get('ayat')} "
                f"type={m.get('document_type')} reg={m.get('regulation_number')}"
            )

    # 4. Exact expected content coverage.
    expected_sha = Counter(sha(d.page_content) for d in expected_docs)
    chroma_sha = Counter(sha(r["content"]) for r in rows)

    expected_missing = []
    expected_present = 0
    for s, count in expected_sha.items():
        if chroma_sha.get(s, 0) >= count:
            expected_present += count
        else:
            expected_missing.append((s, count - chroma_sha.get(s, 0)))

    print(
        f"\n[4] Expected content coverage : {expected_present}/{len(expected_docs)}")
    print(
        f"    Missing expected copies   : {sum(n for _, n in expected_missing)}")

    # 5. Newly repaired Perpres targets.
    target_specs = [
        ("2025perpres004.pdf", 14, "20", "1", "20", "ayat"),
        ("2025perpres004.pdf", 14, None, None, "21", "amendment_instruction"),
        ("2025perpres004.pdf", 16, "30", "1", "30", "ayat"),
        ("2025perpres004.pdf", 17, None, None, "30", "amendment_instruction"),
    ]
    repaired_found = []
    for source, page, pasal, ayat, target, block in target_specs:
        matches = [
            r for r in rows
            if r["metadata"].get("source") == source
            and r["metadata"].get("page") == page
            and r["metadata"].get("pasal") == pasal
            and r["metadata"].get("ayat") == ayat
            and r["metadata"].get("target_pasal") == target
            and r["metadata"].get("block_type") == block
        ]
        repaired_found.append(len(matches) == 1)
        print(
            f"[5] Repaired target page={page} pasal={pasal} ayat={ayat} "
            f"target={target} block={block}: {'PASS' if len(matches) == 1 else 'FAIL'}"
        )

    # 6. Per-source counts.
    counts = Counter(r["metadata"].get("source") for r in rows)
    print("\n[6] Chroma counts by source")
    for source, count in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"    {source}: {count}")

    # 7. Explicitly report the 31-row difference without classifying it as an error.
    print("\n[7] Difference analysis")
    print(
        "    The 31-row surplus is NOT automatically treated as obsolete. "
        "It requires structural/content review before any deletion."
    )

    hard_fail = (
        len(bad_pasal) > 0
        or len(empty_identity) > 0
        or sum(n for _, n in expected_missing) > 0
        or not all(repaired_found)
    )

    print("\n" + "=" * 100)
    if hard_fail:
        print("STATUS: REVIEW_REQUIRED")
        print("Chroma was NOT mutated by this audit.")
    else:
        print("STATUS: PASS — integrity checks passed; 31-row surplus remains REVIEW ONLY.")
        print("Chroma was NOT mutated by this audit.")
    print("=" * 100)


if __name__ == "__main__":
    main()
