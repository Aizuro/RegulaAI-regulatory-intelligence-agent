from src.document_comparison import compare_retrieved_documents
from src.document_retrieval import retrieve_regulation
import sys
from pathlib import Path

# Pastikan src dapat di-import ketika file dijalankan langsung.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


REGULATION_A = {
    "type": "UU",
    "number": "1",
    "year": "2024",
    "title": "UU Nomor 1 Tahun 2024",
}

REGULATION_B = {
    "type": "UU",
    "number": "3",
    "year": "2024",
    "title": "UU Nomor 3 Tahun 2024",
}


def _documents_from_result(result):
    """
    Support both object-style and dict-style V8.1 retrieval results.
    """
    if hasattr(result, "documents"):
        return result.documents

    if isinstance(result, dict):
        return result.get("documents", [])

    raise TypeError(
        f"Unsupported retrieval result type: {type(result).__name__}"
    )


def test_v82_real_local_retrieval_and_comparison():
    """
    Real integration test:

    V8.1 local retrieval
        ->
    V8.2 comparison adapter
        ->
    V3 comparison engine
    """

    result_a = retrieve_regulation(REGULATION_A)
    result_b = retrieve_regulation(REGULATION_B)

    documents_a = _documents_from_result(result_a)
    documents_b = _documents_from_result(result_b)

    assert documents_a, "UU-1-2024 retrieval returned no documents."
    assert documents_b, "UU-3-2024 retrieval returned no documents."

    report = compare_retrieved_documents(
        regulation_a=REGULATION_A,
        documents_a=documents_a,
        regulation_b=REGULATION_B,
        documents_b=documents_b,
    )

    assert report["comparison_ready"] is True
    assert report["type"] == "structural_comparison"

    assert report["counts"]["documents_a"] > 0
    assert report["counts"]["documents_b"] > 0

    print("\n=== V8.2 REAL INTEGRATION ===")
    print(f"A documents : {report['counts']['documents_a']}")
    print(f"B documents : {report['counts']['documents_b']}")
    print(f"Aligned     : {report['counts']['aligned_provisions']}")
    print(f"Only A      : {report['counts']['only_in_a']}")
    print(f"Only B      : {report['counts']['only_in_b']}")
    print(f"Status      : {report['status']}")
    print("V8.2 REAL INTEGRATION: PASS")
