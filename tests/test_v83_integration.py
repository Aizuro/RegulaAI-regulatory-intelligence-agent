from src.document_comparison import compare_retrieved_documents
from src.document_retrieval import retrieve_regulation
from src.impact_analysis import analyze_comparison_impact


REGULATION_A = {
    "type": "UU",
    "number": "1",
    "year": "2024",
    "title": "Regulasi A - V8.3 integration test",
}

REGULATION_B = {
    "type": "UU",
    "number": "3",
    "year": "2024",
    "title": "Regulasi B - V8.3 integration test",
}


def _documents(result):
    if isinstance(result, dict):
        return result.get("documents") or []
    return getattr(result, "documents", []) or []


def test_v83_real_retrieval_comparison_and_impact():
    result_a = retrieve_regulation(REGULATION_A)
    result_b = retrieve_regulation(REGULATION_B)

    docs_a = _documents(result_a)
    docs_b = _documents(result_b)

    assert docs_a
    assert docs_b

    comparison = compare_retrieved_documents(
        regulation_a=REGULATION_A,
        documents_a=docs_a,
        regulation_b=REGULATION_B,
        documents_b=docs_b,
    )

    assert comparison["comparison_ready"] is True
    assert comparison["type"] == "structural_comparison"

    analysis = analyze_comparison_impact(comparison)

    print("\n=== V8.3 REAL INTEGRATION ===")
    print(f"A documents       : {len(docs_a)}")
    print(f"B documents       : {len(docs_b)}")
    print(f"Aligned           : {comparison['counts']['aligned_provisions']}")
    print(f"Only A            : {comparison['counts']['only_in_a']}")
    print(f"Only B            : {comparison['counts']['only_in_b']}")
    print(f"Impact level      : {analysis.impact_level}")
    print(f"Summary           : {analysis.summary}")
    print(f"Changes           : {analysis.changes}")
    print(f"Potential impacts : {analysis.potential_impacts}")
    print(f"Affected areas    : {analysis.affected_areas}")
    print(f"Limitations       : {analysis.limitations}")
    print(f"Evidence          : {analysis.evidence_provisions}")
    print(f"Further research  : {analysis.requires_further_research}")
    print("V8.3 REAL INTEGRATION: PASS")

    assert analysis.summary
    assert isinstance(analysis.changes, list)
    assert isinstance(analysis.potential_impacts, list)
    assert isinstance(analysis.limitations, list)
    assert isinstance(analysis.evidence_provisions, list)
