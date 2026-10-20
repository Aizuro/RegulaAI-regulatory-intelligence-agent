"""Read-only diagnostic: map current expected Documents to existing Chroma records.
No vectors are modified or deleted.
"""
from collections import defaultdict
import hashlib
from pathlib import Path

from src.legal_ingestion import build_legal_documents
from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
)


def norm(s):
    return " ".join(str(s or "").split())


def content_key(text):
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def identity_key(m):
    return (m.get("source"), m.get("page"), m.get("page_end"), m.get("pasal"), m.get("ayat"), content_key(m.get("content", "")))


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

    existing_by_key = defaultdict(list)
    for i, (doc, meta) in enumerate(zip(existing.get("documents", []), existing.get("metadatas", []))):
        m = dict(meta or {})
        m["content"] = doc
        existing_by_key[identity_key(m)].append(existing["ids"][i])

    exact = 0
    unmatched = []
    ambiguous = []
    for m in expected:
        ids = existing_by_key.get(identity_key(m), [])
        if len(ids) == 1:
            exact += 1
        elif len(ids) > 1:
            ambiguous.append((m, ids))
        else:
            unmatched.append(m)

    print("=" * 80)
    print("CHROMA CONTENT/ID MAPPING DIAGNOSTIC (READ ONLY)")
    print("=" * 80)
    print(f"Chroma vectors       : {len(existing.get('ids', []))}")
    print(f"Expected documents   : {len(expected)}")
    print(f"Exact content matches: {exact}")
    print(f"Ambiguous matches    : {len(ambiguous)}")
    print(f"Unmatched expected   : {len(unmatched)}")
    print()

    print("UNMATCHED BY SOURCE:")
    counts = defaultdict(int)
    for m in unmatched:
        counts[m.get("source")] += 1
    for source, n in sorted(counts.items()):
        print(f"  {source}: {n}")

    print("\nSAMPLE UNMATCHED:")
    for m in unmatched[:30]:
        print(f"  {m.get('source')} | page={m.get('page')}-{m.get('page_end')} | pasal={m.get('pasal')} | ayat={m.get('ayat')} | type={m.get('document_type')} | reg={m.get('regulation_number')} | year={m.get('year')}")

    print("\nSTATUS: DIAGNOSTIC ONLY")
    print("No vectors were deleted, updated, or added.")


if __name__ == "__main__":
    main()
