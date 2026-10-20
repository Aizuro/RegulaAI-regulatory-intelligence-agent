"""
RAG V2 End-to-End Test

Tests the production RAG chain against the existing 776-vector legal
collection. No indexing or retriever modification is performed.
"""

from src.rag_chain import setup_vectorstore, build_rag_chain
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TEST_CASES = [
    {
        "id": "Q1",
        "query": "Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?",
        "expected_source": "2024uu001.pdf",
        "expected_pasal": "17",
        "expected_ayat": "2a",
    },
    {
        "id": "Q2",
        "query": "Apa yang diatur dalam UU Nomor 3 Tahun 2024?",
        "expected_source": "2024uu003.pdf",
    },
    {
        "id": "Q3",
        "query": "Apa hubungan UU Nomor 1 Tahun 2024 dengan UU Nomor 19 Tahun 2016?",
        "expected_source": "2024uu001.pdf",
    },
    {
        "id": "Q4",
        "query": "Apa yang diatur dalam Peraturan Presiden Nomor 4 Tahun 2025?",
        "expected_source": "2025perpres004.pdf",
    },
    {
        "id": "Q5",
        "query": "Bagaimana ketentuan mengenai hak warga negara dalam UUD 1945?",
        "expected_source": "UUD1945.pdf",
    },
]


def source_matches(doc, expected_source):
    source = str((doc.metadata or {}).get("source", ""))
    return source.replace("\\", "/").split("/")[-1] == expected_source


def article_matches(doc, pasal, ayat):
    metadata = doc.metadata or {}
    return (
        str(metadata.get("pasal")) == str(pasal)
        and str(metadata.get("ayat")).lower() == str(ayat).lower()
    )


def main():
    print("=" * 78)
    print("RAG V2 END-TO-END EVALUATION")
    print("=" * 78)
    print("Existing production vectorstore only.")
    print("No re-indexing. Production retriever is NOT modified.")

    try:
        vectorstore = setup_vectorstore()
        count = vectorstore._collection.count()
        print(f"Loaded collection: legal_regulations")
        print(f"Vectors: {count}")

        if count < 776:
            print("STATUS: FAIL | Expected 776 vectors.")
            return 1

        print(
            f"Vector count check: PASS "
            f"({count} vectors; expected corpus coverage >= 776)"
        )

        rag_chain, retriever = build_rag_chain(vectorstore)

    except Exception as exc:
        print("\nSETUP ERROR:")
        traceback.print_exc()
        return 1

    results = []

    for case in TEST_CASES:
        print("\n" + "=" * 78)
        print(f"{case['id']}")
        print(case["query"])
        print("=" * 78)

        try:
            result = rag_chain.invoke(case["query"])

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            answer_ok = isinstance(answer, str) and bool(answer.strip())
            sources_ok = isinstance(sources, list) and len(sources) > 0

            source_ok = any(
                source_matches(doc, case["expected_source"])
                for doc in sources
            )

            article_ok = True
            if "expected_pasal" in case:
                article_ok = any(
                    source_matches(doc, case["expected_source"])
                    and article_matches(
                        doc,
                        case["expected_pasal"],
                        case["expected_ayat"],
                    )
                    for doc in sources
                )

            passed = answer_ok and sources_ok and source_ok and article_ok

            print("\n[ANSWER]")
            print(answer.strip())

            print("\n[SOURCES]")
            for i, doc in enumerate(sources, 1):
                metadata = doc.metadata or {}
                print(
                    f"{i}. {metadata.get('source')} | "
                    f"type={metadata.get('document_type')} | "
                    f"reg={metadata.get('regulation_number')} | "
                    f"year={metadata.get('year')} | "
                    f"pasal={metadata.get('pasal')} | "
                    f"ayat={metadata.get('ayat')} | "
                    f"page={metadata.get('page')} | "
                    f"page_end={metadata.get('page_end')}"
                )

            print("\n[CHECK]")
            print(f"Answer generated : {'PASS' if answer_ok else 'FAIL'}")
            print(f"Sources returned : {'PASS' if sources_ok else 'FAIL'}")
            print(
                f"Expected source  : "
                f"{'PASS' if source_ok else 'FAIL'}"
            )

            if "expected_pasal" in case:
                print(
                    f"Exact article   : "
                    f"{'PASS' if article_ok else 'FAIL'}"
                )

            print(f"CASE RESULT: {'PASS' if passed else 'FAIL'}")
            results.append((case["id"], passed))

        except Exception:
            print("\nCASE ERROR:")
            traceback.print_exc()
            results.append((case["id"], False))

    passed_count = sum(ok for _, ok in results)
    total = len(results)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    for case_id, ok in results:
        print(f"{case_id}: {'PASS' if ok else 'FAIL'}")

    print()
    print(f"Passed: {passed_count}/{total}")
    print("STATUS: " + ("PASS" if passed_count == total else "FAIL"))
    print("=" * 78)

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
