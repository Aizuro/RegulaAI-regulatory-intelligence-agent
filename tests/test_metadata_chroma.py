"""
Chroma Metadata Diagnostic

Read-only diagnostic:
- Tidak indexing ulang
- Tidak embedding
- Tidak mengubah retriever.py
- Tidak mengubah vectorstore
"""

from __future__ import annotations

from embeddings_v2 import (
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_COLLECTION_NAME,
    legal_vectorstore_exists,
    load_legal_vectorstore,
)


def print_metadata(md: dict, label: str = ""):
    print(f"\n{label}")
    print("-" * 78)
    for key in sorted(md):
        value = md[key]
        print(f"{key:25} = {value!r}    TYPE={type(value).__name__}")


def run_filter(vs, label, filt):
    print(f"\n[{label}]")
    print(f"Filter: {filt}")
    try:
        docs = vs.similarity_search(
            "UU Nomor 1 Tahun 2024",
            k=5,
            filter=filt,
        )
        print(f"Result count: {len(docs)}")
        for i, doc in enumerate(docs, 1):
            md = doc.metadata or {}
            print(
                f"{i}. source={md.get('source')!r} | "
                f"document_type={md.get('document_type')!r} | "
                f"regulation_number={md.get('regulation_number')!r} "
                f"({type(md.get('regulation_number')).__name__}) | "
                f"year={md.get('year')!r} "
                f"({type(md.get('year')).__name__}) | "
                f"pasal={md.get('pasal')!r}"
            )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def main():
    print("=" * 78)
    print("CHROMA METADATA DIAGNOSTIC")
    print("=" * 78)
    print("READ-ONLY: no indexing, no embedding, no vectorstore modification.")

    if not legal_vectorstore_exists(DEFAULT_PERSIST_DIRECTORY):
        raise FileNotFoundError("Vectorstore tidak ditemukan.")

    vs = load_legal_vectorstore(
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )

    collection = vs._collection
    count = collection.count()

    print(f"\nCollection : {DEFAULT_COLLECTION_NAME}")
    print(f"Vectors    : {count}")

    if count != 776:
        print("WARNING: expected 776 vectors.")

    # ------------------------------------------------------------------
    # 1. Inspect raw metadata directly from Chroma
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1. RAW CHROMA METADATA")
    print("=" * 78)

    raw = collection.get(
        limit=10,
        include=["metadatas"],
    )

    metadatas = raw.get("metadatas") or []

    for i, md in enumerate(metadatas, 1):
        print_metadata(md or {}, f"RAW RECORD #{i}")

    # ------------------------------------------------------------------
    # 2. Find representative metadata by source
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. REPRESENTATIVE DOCUMENT METADATA")
    print("=" * 78)

    representative_sources = [
        "2024uu001.pdf",
        "2024uu003.pdf",
        "2025perpres004.pdf",
        "UUD1945.pdf",
    ]

    raw_all = collection.get(
        include=["metadatas"],
    )

    all_mds = raw_all.get("metadatas") or []

    for source in representative_sources:
        matches = [
            md for md in all_mds
            if md and md.get("source") == source
        ]

        print(f"\nSOURCE = {source!r}")
        print(f"Records found = {len(matches)}")

        if matches:
            print_metadata(matches[0], "First matching record")

            # Also report distinct types/values for key fields.
            for key in [
                "document_type",
                "regulation_number",
                "year",
                "pasal",
                "ayat",
            ]:
                values = {
                    (repr(md.get(key)), type(md.get(key)).__name__)
                    for md in matches
                }
                print(f"{key:25} distinct = {sorted(values)[:15]}")

    # ------------------------------------------------------------------
    # 3. Single-field filters
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. SINGLE-FIELD FILTER TESTS")
    print("=" * 78)

    run_filter(
        vs,
        "document_type = UU",
        {"document_type": {"$eq": "UU"}},
    )

    run_filter(
        vs,
        "regulation_number = INTEGER 1",
        {"regulation_number": {"$eq": 1}},
    )

    run_filter(
        vs,
        "regulation_number = STRING '1'",
        {"regulation_number": {"$eq": "1"}},
    )

    run_filter(
        vs,
        "year = INTEGER 2024",
        {"year": {"$eq": 2024}},
    )

    run_filter(
        vs,
        "year = STRING '2024'",
        {"year": {"$eq": "2024"}},
    )

    # ------------------------------------------------------------------
    # 4. Two-field filters
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. MULTI-FIELD FILTER TESTS")
    print("=" * 78)

    run_filter(
        vs,
        "document_type + regulation_number",
        {
            "$and": [
                {"document_type": {"$eq": "UU"}},
                {"regulation_number": {"$eq": 1}},
            ]
        },
    )

    run_filter(
        vs,
        "document_type + year",
        {
            "$and": [
                {"document_type": {"$eq": "UU"}},
                {"year": {"$eq": 2024}},
            ]
        },
    )

    run_filter(
        vs,
        "UU + reg=1 + year=2024",
        {
            "$and": [
                {"document_type": {"$eq": "UU"}},
                {"regulation_number": {"$eq": 1}},
                {"year": {"$eq": 2024}},
            ]
        },
    )

    # ------------------------------------------------------------------
    # 5. Pasal diagnostic
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. PASAL METADATA DIAGNOSTIC")
    print("=" * 78)

    run_filter(
        vs,
        "UU 1/2024 + pasal=17",
        {
            "$and": [
                {"document_type": {"$eq": "UU"}},
                {"regulation_number": {"$eq": 1}},
                {"year": {"$eq": 2024}},
                {"pasal": {"$eq": "17"}},
            ]
        },
    )

    run_filter(
        vs,
        "UU 1/2024 + pasal=16B",
        {
            "$and": [
                {"document_type": {"$eq": "UU"}},
                {"regulation_number": {"$eq": 1}},
                {"year": {"$eq": 2024}},
                {"pasal": {"$eq": "16B"}},
            ]
        },
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)
    print("Gunakan hasil bagian 3-5 untuk menentukan apakah masalahnya:")
    print("1. tipe metadata,")
    print("2. multi-field filter, atau")
    print("3. nilai metadata Pasal/Ayat yang tersimpan.")
    print("=" * 78)


if __name__ == "__main__":
    main()
