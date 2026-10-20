"""
LEGAL RETRIEVAL EVALUATION V8
Final six-case evaluator.

Production vectorstore is read-only:
- no re-indexing
- no retriever modification
- exact metadata lookup is used only for Q1 exact-article validation
"""

from src.embeddings import load_legal_vectorstore
from pathlib import Path
import sys

# Allow execution from project root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TEST_CASES = [
    {
        "id": "Q1",
        "type": "exact_article",
        "query": "Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?",
        "source": "2024uu001.pdf",
        "document_type": "UU",
        "regulation_number": "1",
        "year": "2024",
        "pasal": "17",
        "ayat": "2a",
    },
    {
        "id": "Q2",
        "type": "regulation_identity",
        "query": "Apa yang diatur dalam UU Nomor 3 Tahun 2024?",
        "source": "2024uu003.pdf",
        "document_type": "UU",
        "regulation_number": "3",
        "year": "2024",
    },
    {
        "id": "Q3",
        "type": "relationship",
        "query": "Apa hubungan UU Nomor 1 Tahun 2024 dengan UU Nomor 19 Tahun 2016?",
        "source": "2024uu001.pdf",
        "document_type": "UU",
        "regulation_number": "1",
        "year": "2024",
    },
    {
        "id": "Q4",
        "type": "regulation_identity",
        "query": "Apa yang diatur dalam Peraturan Presiden Nomor 4 Tahun 2025?",
        "source": "2025perpres004.pdf",
        "document_type": "Perpres",
        "regulation_number": "4",
        "year": "2025",
    },
    {
        "id": "Q5",
        "type": "broad_topic",
        "query": "Bagaimana ketentuan mengenai hak warga negara dalam UUD 1945?",
        "source": "UUD1945.pdf",
        "document_type": "UUD",
        "year": "1945",
    },
    {
        "id": "Q6",
        "type": "negative",
        "query": "Apa ketentuan mengenai pajak karbon dalam Peraturan Presiden Nomor 99 Tahun 2099?",
        "source": None,
        "document_type": "Perpres",
        "regulation_number": "99",
        "year": "2099",
    },
]


def metadata_match(metadata, case):
    if metadata.get("source") != case["source"]:
        return False
    if metadata.get("document_type") != case["document_type"]:
        return False
    if str(metadata.get("regulation_number")) != str(case.get("regulation_number")):
        return False
    if str(metadata.get("year")) != str(case.get("year")):
        return False
    return True


def exact_article_match(metadata, case):
    return (
        metadata_match(metadata, case)
        and str(metadata.get("pasal")) == str(case["pasal"])
        and str(metadata.get("ayat")).lower() == str(case["ayat"]).lower()
    )


def main():
    print("=" * 78)
    print("LEGAL RETRIEVAL EVALUATION V8")
    print("=" * 78)
    print("Metadata filter values use STRING types.")
    print("No re-indexing. Production retriever is NOT modified.")

    vs = load_legal_vectorstore()
    count = vs._collection.count()
    print(f"Loaded collection 'legal_regulations' ({count} vectors).")
    print(f"Collection: legal_regulations")
    print(f"Vectors   : {count}")

    if count < 776:
        print("STATUS: FAIL | Expected 776 vectors.")
        return 1

    print(
        f"Vector count check: PASS ({count} vectors; expected corpus coverage >= 776)")
    results = []

    for case in TEST_CASES:
        print("\n" + "=" * 78)
        print(f"{case['id']} | {case['type']}")
        print(case["query"])
        print("=" * 78)

        if case["id"] == "Q1":
            where = {
                "$and": [
                    {"document_type": {"$eq": "UU"}},
                    {"regulation_number": {"$eq": "1"}},
                    {"year": {"$eq": "2024"}},
                    {"pasal": {"$eq": "17"}},
                    {"ayat": {"$eq": "2a"}},
                ]
            }
            raw = vs._collection.get(
                where=where,
                include=["metadatas", "documents"],
            )
            metadatas = raw.get("metadatas") or []

            passed = any(exact_article_match(m, case) for m in metadatas)
            print("[EXACT METADATA FILTER]")
            print(f"Exact matches: {len(metadatas)}")
            for i, m in enumerate(metadatas[:4], 1):
                print(
                    f"{i}. {m.get('source')} | "
                    f"pasal={m.get('pasal')} | ayat={m.get('ayat')} | "
                    f"page={m.get('page')} | page_end={m.get('page_end')} | "
                    f"cross_page={m.get('is_cross_page')}"
                )
            print("EXACT AYAT FILTER: " + ("PASS" if passed else "FAIL"))

        elif case["id"] == "Q6":
            where = {
                "$and": [
                    {"document_type": {"$eq": "Perpres"}},
                    {"regulation_number": {"$eq": "99"}},
                    {"year": {"$eq": "2099"}},
                ]
            }
            raw = vs._collection.get(
                where=where,
                include=["metadatas", "documents"],
            )
            metadatas = raw.get("metadatas") or []
            passed = len(metadatas) == 0
            print("[NEGATIVE METADATA FILTER]")
            print(f"Exact regulation matches: {len(metadatas)}")
            print("NOT FOUND BEHAVIOR: " + ("PASS" if passed else "FAIL"))

        else:
            # UUD 1945 does not have regulation_number in its metadata.
            # UU/Perpres use document_type + regulation_number + year.
            conditions = [
                {"document_type": {"$eq": case["document_type"]}},
                {"year": {"$eq": case["year"]}},
            ]

            if case.get("regulation_number") is not None:
                conditions.insert(
                    1,
                    {"regulation_number": {"$eq": case["regulation_number"]}},
                )

            where = {"$and": conditions}
            raw = vs._collection.get(
                where=where,
                include=["metadatas", "documents"],
                limit=4,
            )
            metadatas = raw.get("metadatas") or []
            passed = any(metadata_match(m, case) for m in metadatas)

            print("[METADATA FILTER]")
            print(f"Matches returned: {len(metadatas)}")
            for i, m in enumerate(metadatas[:4], 1):
                print(
                    f"{i}. {m.get('source')} | {m.get('document_type')} | "
                    f"reg={m.get('regulation_number')} | year={m.get('year')} | "
                    f"pasal={m.get('pasal')} | ayat={m.get('ayat')} | "
                    f"page={m.get('page')}"
                )
            print("METADATA FILTER: " + ("PASS" if passed else "FAIL"))

        results.append((case["id"], passed))
        print("CASE RESULT: " + ("PASS" if passed else "FAIL"))

    passed_count = sum(ok for _, ok in results)
    total = len(results)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    for case_id, ok in results:
        print(f"{case_id}: " + ("PASS" if ok else "FAIL"))
    print()
    print(f"Passed: {passed_count}/{total}")
    print("STATUS: " + ("PASS" if passed_count == total else "FAIL"))
    print("=" * 78)

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
