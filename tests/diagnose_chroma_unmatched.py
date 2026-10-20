"""Read-only diagnostic for the 64 unmatched Chroma records.

Compares current ingestion output against Chroma using progressively looser
content normalization and structural keys. No vectors are modified.
"""
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import re

from src.legal_ingestion import build_legal_documents
from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
)


def norm_basic(s):
    return " ".join(str(s or "").split())


def norm_legal(s):
    s = str(s or "")
    # Remove common extraction/cleaning-only differences.
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<=\w)-\s+(?=\w)", "", s)  # line-break hyphenation
    s = re.sub(r"\s*([,.;:()])\s*", r"\1", s)
    return s.strip().lower()


def sha(s):
    return hashlib.sha256(norm_basic(s).encode("utf-8")).hexdigest()


def structure_key(m):
    return (
        m.get("source"),
        m.get("page"),
        m.get("page_end"),
        m.get("pasal"),
        m.get("ayat"),
    )


def identity_key(m):
    return (
        m.get("source"),
        m.get("page"),
        m.get("page_end"),
        m.get("pasal"),
        m.get("ayat"),
        sha(m.get("content", "")),
    )


def similarity(a, b):
    return SequenceMatcher(None, norm_legal(a), norm_legal(b)).ratio()


def main():
    model = get_embedding_model()
    vs = _open_vectorstore(
        model, DEFAULT_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME)
    col = vs._collection
    existing = col.get(include=["documents", "metadatas"])

    expected_docs, _ = build_legal_documents()
    expected = []
    for d in expected_docs:
        m = dict(d.metadata)
        m["content"] = d.page_content
        expected.append(m)

    # Existing Chroma records indexed by progressively looser structural keys.
    by_structure = defaultdict(list)
    by_source_page = defaultdict(list)
    for i, (doc, meta) in enumerate(zip(existing.get("documents", []), existing.get("metadatas", []))):
        m = dict(meta or {})
        m["content"] = doc or ""
        rec = (existing["ids"][i], m)
        by_structure[structure_key(m)].append(rec)
        by_source_page[(m.get("source"), m.get("page"),
                        m.get("page_end"))].append(rec)

    # Identify exact matches first, same logic as previous diagnostic.
    exact_keys = defaultdict(list)
    for i, (doc, meta) in enumerate(zip(existing.get("documents", []), existing.get("metadatas", []))):
        m = dict(meta or {})
        m["content"] = doc or ""
        exact_keys[identity_key(m)].append(existing["ids"][i])

    unmatched = []
    ambiguous_exact = []
    for m in expected:
        ids = exact_keys.get(identity_key(m), [])
        if len(ids) == 0:
            unmatched.append(m)
        elif len(ids) > 1:
            ambiguous_exact.append((m, ids))

    categories = defaultdict(int)
    details = []

    for m in unmatched:
        candidates = []
        # Strongest candidate set: same source + page range + legal structure.
        for rid, em in by_structure.get(structure_key(m), []):
            score = similarity(m["content"], em["content"])
            candidates.append((score, "same_structure", rid, em))

        # If none, use same source + page range.
        if not candidates:
            for rid, em in by_source_page.get((m.get("source"), m.get("page"), m.get("page_end")), []):
                score = similarity(m["content"], em["content"])
                candidates.append((score, "same_source_page", rid, em))

        # Last resort: same source, limited to a manageable set and rank by text similarity.
        if not candidates:
            source_records = [
                (rid, em) for (rid, em) in
                sum(([x for x in vals] for key, vals in by_source_page.items()
                    if key[0] == m.get("source")), [])
            ]
            for rid, em in source_records:
                score = similarity(m["content"], em["content"])
                if score >= 0.70:
                    candidates.append((score, "same_source_fuzzy", rid, em))

        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            category = "NO_CANDIDATE"
        else:
            top = candidates[0][0]
            second = candidates[1][0] if len(candidates) > 1 else 0.0
            if top >= 0.995 and top - second >= 0.01:
                category = "VERY_LIKELY_SAME"
            elif top >= 0.95 and top - second >= 0.01:
                category = "LIKELY_SAME"
            elif top >= 0.85:
                category = "POSSIBLE_SAME"
            else:
                category = "LOW_SIMILARITY"

        categories[category] += 1
        details.append((category, m, candidates[:3]))

    print("=" * 80)
    print("CHROMA UNMATCHED CONTENT DIAGNOSTIC (READ ONLY)")
    print("=" * 80)
    print(f"Chroma vectors       : {len(existing.get('ids', []))}")
    print(f"Expected documents   : {len(expected)}")
    print(f"Exact unmatched     : {len(unmatched)}")
    print(f"Exact ambiguous     : {len(ambiguous_exact)}")
    print()
    print("UNMATCHED CLASSIFICATION:")
    for k in ["VERY_LIKELY_SAME", "LIKELY_SAME", "POSSIBLE_SAME", "LOW_SIMILARITY", "NO_CANDIDATE"]:
        print(f"  {k:20s}: {categories[k]}")

    print("\nDETAILS:")
    for category, m, candidates in details:
        print("-" * 80)
        print(
            f"{category} | {m.get('source')} | page={m.get('page')}-{m.get('page_end')} "
            f"| pasal={m.get('pasal')} | ayat={m.get('ayat')}"
        )
        print(f"  expected chars={len(m.get('content',''))}")
        for score, reason, rid, em in candidates:
            print(
                f"  candidate score={score:.4f} reason={reason} id={rid} "
                f"page={em.get('page')}-{em.get('page_end')} pasal={em.get('pasal')} ayat={em.get('ayat')} "
                f"type={em.get('document_type')} reg={em.get('regulation_number')} year={em.get('year')} "
                f"chars={len(em.get('content',''))}"
            )

    print("\nAMBIGUOUS EXACT MATCHES:")
    for m, ids in ambiguous_exact[:30]:
        print(
            f"  {m.get('source')} | page={m.get('page')}-{m.get('page_end')} "
            f"| pasal={m.get('pasal')} | ayat={m.get('ayat')} | ids={len(ids)}"
        )

    print("\nSTATUS: DIAGNOSTIC ONLY")
    print("No vectors were deleted, updated, or added.")


if __name__ == "__main__":
    main()
