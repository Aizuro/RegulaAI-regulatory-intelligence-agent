import hashlib
import re
from pathlib import Path
from difflib import SequenceMatcher

from src.legal_ingestion import build_legal_documents
from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
)


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sha(text):
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()[:16]


def key(d):
    m = d.metadata
    return (
        m.get("source"), m.get("page"), m.get("page_end"),
        m.get("pasal"), m.get("ayat"), sha(d.page_content)
    )


def meta_line(m):
    return (f"id={m.get('_id')} page={m.get('page')}-{m.get('page_end')} "
            f"pasal={m.get('pasal')} ayat={m.get('ayat')} "
            f"type={m.get('document_type')} reg={m.get('regulation_number')} "
            f"year={m.get('year')} chars={len(str(m.get('_content','')))}")


def main():
    expected, _ = build_legal_documents()
    model = get_embedding_model()
    vs = _open_vectorstore(
        model, DEFAULT_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME)
    raw = vs._collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids", [])
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])

    chroma = []
    for i, cid in enumerate(ids):
        m = dict(metas[i] or {})
        m["_id"] = cid
        m["_content"] = docs[i] or ""
        chroma.append(m)

    # Reproduce exact matching logic from the prior mapping diagnostic.
    exact_map = {}
    for m in chroma:
        k = (m.get("source"), m.get("page"), m.get("page_end"),
             m.get("pasal"), m.get("ayat"), sha(m.get("_content")))
        exact_map.setdefault(k, []).append(m)

    unmatched = []
    ambiguous = []
    for d in expected:
        m = d.metadata
        k = (m.get("source"), m.get("page"), m.get("page_end"),
             m.get("pasal"), m.get("ayat"), sha(d.page_content))
        cands = exact_map.get(k, [])
        if len(cands) == 0:
            unmatched.append(d)
        elif len(cands) > 1:
            ambiguous.append((d, cands))

    low = []
    likely = []
    for d in unmatched:
        m = d.metadata
        same_source = [c for c in chroma if c.get("source") == m.get("source")]
        scored = []
        for c in same_source:
            if c.get("page") != m.get("page") or c.get("page_end") != m.get("page_end"):
                continue
            score = SequenceMatcher(None, norm(
                d.page_content), norm(c.get("_content"))).ratio()
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]
        best = top[0][0] if top else 0
        if best < 0.30:
            low.append((d, top))
        else:
            likely.append((d, top))

    print("=" * 96)
    print("CHROMA UNMATCHED DETAIL DIAGNOSTIC (READ ONLY)")
    print("=" * 96)
    print(f"Chroma vectors : {len(chroma)}")
    print(f"Expected docs  : {len(expected)}")
    print(f"Unmatched      : {len(unmatched)}")
    print(f"Ambiguous      : {len(ambiguous)}")
    print()

    print("=" * 96)
    print("A. LOW-SIMILARITY / STRUCTURAL OUTLIERS")
    print("=" * 96)
    for d, top in low:
        print("-" * 96)
        print("EXPECTED:")
        print(
            meta_line({**d.metadata, "_content": d.page_content, "_id": "<expected>"}))
        print(f"sha={sha(d.page_content)}")
        print(norm(d.page_content)[:1800])
        print("\nCANDIDATES:")
        if not top:
            print("  NONE")
        for score, c in top:
            print(
                f"  score={score:.4f} {meta_line(c)} sha={sha(c['_content'])}")
            print("  " + norm(c["_content"])[:1200])

    print("\n" + "=" * 96)
    print("B. ALL UNMATCHED HIGH-CONFIDENCE CANDIDATES")
    print("=" * 96)
    for d, top in sorted(likely, key=lambda x: (x[0].metadata.get("source", ""), x[0].metadata.get("page", 0), x[0].metadata.get("pasal") or "")):
        print("-" * 96)
        print("EXPECTED:")
        print(
            meta_line({**d.metadata, "_content": d.page_content, "_id": "<expected>"}))
        for score, c in top[:3]:
            print(
                f"  CANDIDATE score={score:.4f} {meta_line(c)} sha={sha(c['_content'])}")

    print("\n" + "=" * 96)
    print("C. AMBIGUOUS EXACT MATCHES")
    print("=" * 96)
    for d, cands in ambiguous:
        print("-" * 96)
        print("EXPECTED:")
        print(
            meta_line({**d.metadata, "_content": d.page_content, "_id": "<expected>"}))
        print(f"content_sha={sha(d.page_content)}")
        for idx, c in enumerate(cands, 1):
            print(
                f"  CANDIDATE #{idx}: {meta_line(c)} sha={sha(c['_content'])}")
            print(
                f"    metadata_signature: type={c.get('document_type')} reg={c.get('regulation_number')} year={c.get('year')} pasal={c.get('pasal')} ayat={c.get('ayat')} page={c.get('page')}-{c.get('page_end')}")

    print("\n" + "=" * 96)
    print("STATUS: DIAGNOSTIC ONLY")
    print("No vectors were deleted, updated, or added.")
    print("=" * 96)


if __name__ == "__main__":
    main()
