"""Read-only Chroma migration plan generator.

Generates exact DELETE/REPLACE/ADD candidates from current ingestion output vs
existing Chroma. Never mutates the collection.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from src.legal_ingestion import build_legal_documents
from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
)


def norm(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def meta_sig(m: dict) -> tuple:
    return (
        m.get("source"), m.get("page"), m.get("page_end"),
        m.get("pasal"), m.get("ayat"), sha(m.get("_content", "")),
    )


def identity_complete(m: dict) -> bool:
    dt = m.get("document_type")
    if dt == "UUD":
        return True
    return bool(dt and m.get("regulation_number") and m.get("year"))


def structural_key(m: dict) -> tuple:
    return (m.get("source"), m.get("page"), m.get("page_end"), sha(m.get("_content", "")))


def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False,
                    indent=2, default=str), encoding="utf-8")


def main() -> None:
    expected_docs, reports = build_legal_documents()
    model = get_embedding_model()
    vs = _open_vectorstore(
        model, DEFAULT_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME)
    collection = vs._collection
    got = collection.get(include=["documents", "metadatas"])

    ids = got.get("ids", [])
    docs = got.get("documents", [])
    metas = got.get("metadatas", [])
    chroma = []
    for i, content, meta in zip(ids, docs, metas):
        m = dict(meta or {})
        m["_content"] = content or ""
        m["_id"] = i
        chroma.append(m)

    # Exact structural/content groups.
    by_struct = defaultdict(list)
    for c in chroma:
        by_struct[structural_key(c)].append(c)

    # For each expected doc, find exact structural candidates.
    exact = []
    unmatched = []
    ambiguous = []
    for d in expected_docs:
        m = dict(d.metadata)
        m["_content"] = d.page_content
        candidates = by_struct.get(structural_key(m), [])
        if len(candidates) == 1:
            exact.append((m, candidates[0]))
        elif len(candidates) > 1:
            ambiguous.append((m, candidates))
        else:
            unmatched.append(m)

    # 19 known duplicate groups: same exact content/structural location, where
    # one copy has complete metadata and another is missing identity metadata.
    duplicate_groups = []
    delete_ids = []
    for m, candidates in ambiguous:
        complete = [c for c in candidates if identity_complete(c)]
        incomplete = [c for c in candidates if not identity_complete(c)]
        if len(complete) >= 1 and len(incomplete) >= 1:
            duplicate_groups.append({
                "source": m.get("source"), "page": m.get("page"),
                "page_end": m.get("page_end"), "pasal": m.get("pasal"),
                "ayat": m.get("ayat"), "sha": sha(m.get("_content", "")),
                "keep_ids": [c["_id"] for c in complete],
                "delete_ids": [c["_id"] for c in incomplete],
                "reason": "exact duplicate content/location; retain metadata-complete copy",
            })
            delete_ids.extend(c["_id"] for c in incomplete)

    # Genuine structural outliers identified from the diagnostic: expected
    # corrected Perpres records have no exact content match in Chroma.
    structural_outliers = []
    for m in unmatched:
        if m.get("source") == "2025perpres004.pdf" and m.get("page") in (14, 16, 17):
            structural_outliers.append(m)

    # Find best old candidates for those outliers by source/page and lexical
    # similarity, without relying on embeddings. This is only a plan.
    def token_score(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", norm(a).lower()))
        tb = set(re.findall(r"\w+", norm(b).lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    replace_plan = []
    for m in structural_outliers:
        candidates = [c for c in chroma if c.get("source") == m.get(
            "source") and c.get("page") == m.get("page")]
        ranked = sorted(((token_score(m["_content"], c["_content"]), c)
                        for c in candidates), reverse=True, key=lambda x: x[0])
        replace_plan.append({
            "action": "REPLACE_CANDIDATE",
            "expected": {k: m.get(k) for k in ["source", "document_type", "regulation_number", "year", "pasal", "ayat", "page", "page_end", "target_pasal", "block_type"]} | {"content_sha": sha(m["_content"]), "content_chars": len(m["_content"])},
            "candidate_chroma": [
                {"score": round(s, 4), "id": c["_id"], "document_type": c.get("document_type"), "regulation_number": c.get("regulation_number"), "year": c.get("year"), "pasal": c.get("pasal"), "ayat": c.get(
                    "ayat"), "page": c.get("page"), "page_end": c.get("page_end"), "target_pasal": c.get("target_pasal"), "block_type": c.get("block_type"), "content_sha": sha(c["_content"]), "content_chars": len(c["_content"])}
                for s, c in ranked[:5]
            ],
            "reason": "parser/OCR correction; no exact current-content match exists",
        })

    matched_expected_ids = {c["_id"] for _, c in exact}
    kept_ambiguous_ids = {c["_id"] for g in duplicate_groups for c in []}
    # IDs not explained by exact matches or planned deletes are left untouched.
    all_delete = set(delete_ids)
    all_known_keep = matched_expected_ids | {
        i for g in duplicate_groups for i in g["keep_ids"]}
    extras = [c for c in chroma if c["_id"]
              not in all_known_keep and c["_id"] not in all_delete]

    plan = {
        "status": "REVIEW_REQUIRED",
        "read_only": True,
        "chroma_vectors": len(chroma),
        "expected_documents": len(expected_docs),
        "exact_structural_matches": len(exact),
        "ambiguous_groups": len(ambiguous),
        "duplicate_groups_safe_delete": len(duplicate_groups),
        "safe_delete_ids_count": len(delete_ids),
        "structural_outliers": len(structural_outliers),
        "replace_candidates": len(replace_plan),
        "untouched_extra_or_review_ids": len(extras),
        "delete_ids": delete_ids,
        "duplicate_groups": duplicate_groups,
        "replace_candidates": replace_plan,
        "review_note": "Do not delete/reindex anything from this file alone. Review replace candidates and verify ID behavior with the actual indexer before mutation.",
        "reports": reports,
    }

    out = Path("migration_plan.json")
    dump_json(out, plan)
    print("=" * 96)
    print("CHROMA MIGRATION PLAN — READ ONLY")
    print("=" * 96)
    for k in ["chroma_vectors", "expected_documents", "exact_structural_matches", "ambiguous_groups", "duplicate_groups_safe_delete", "safe_delete_ids_count", "structural_outliers", "replace_candidates", "untouched_extra_or_review_ids"]:
        print(f"{k:32}: {plan[k]}")
    print("\nSAFE DELETE IDS (duplicate None-metadata copies):")
    for x in delete_ids:
        print("  ", x)
    print("\nREPLACE CANDIDATES:")
    for r in replace_plan:
        e = r["expected"]
        print(
            f"  {e['source']} page={e['page']} pasal={e['pasal']} ayat={e['ayat']} sha={e['content_sha'][:12]}...")
        for c in r["candidate_chroma"][:3]:
            print(
                f"    candidate id={c['id']} score={c['score']} pasal={c['pasal']} ayat={c['ayat']} chars={c['content_chars']} sha={c['content_sha'][:12]}...")
    print(f"\nWrote: {out.resolve()}")
    print("STATUS: REVIEW_REQUIRED — NO CHROMA MUTATION")


if __name__ == "__main__":
    main()
