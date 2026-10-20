from __future__ import annotations

"""
V8.4 real integration test.

Flow:
V8.4 impact tool
    -> V8.1 document retrieval
    -> V8.2 structural comparison
    -> V8.3 grounded impact analysis

This test validates the actual contracts returned by the existing V8.2
comparison engine. It does not modify production code.
"""

from src.v84_impact_tool import analyze_regulation_impact


def test_v84_real_impact_pipeline():
    """
    Run the complete V8.4 impact-analysis pipeline against real local
    regulation documents.

    Assertions focus on the structural contract and grounded evidence rather
    than a particular legal conclusion.
    """

    result = analyze_regulation_impact(
        regulation_a={
            "type": "UU",
            "number": "1",
            "year": "2024",
        },
        regulation_b={
            "type": "UU",
            "number": "3",
            "year": "2024",
        },
    )

    assert isinstance(result, dict)

    # V8.2 comparison evidence must be present.
    comparison = result.get("comparison")
    assert isinstance(comparison, dict)
    assert comparison.get("status") == "ready"
    assert comparison.get("comparison_ready") is True

    # V8.2 uses aligned_provisions / only_in_a / only_in_b.
    counts = comparison.get("counts")
    assert isinstance(counts, dict)
    assert counts.get("documents_a", 0) > 0
    assert counts.get("documents_b", 0) > 0

    aligned = comparison.get("aligned_provisions")
    only_in_a = comparison.get("only_in_a")
    only_in_b = comparison.get("only_in_b")

    assert isinstance(aligned, list)
    assert isinstance(only_in_a, list)
    assert isinstance(only_in_b, list)

    # Sanity-check that the comparison contains actual structural evidence.
    assert counts.get("aligned_provisions") == len(aligned)
    assert counts.get("only_in_a") == len(only_in_a)
    assert counts.get("only_in_b") == len(only_in_b)

    # Every aligned provision should expose the structural fields produced
    # by V8.2.
    for provision in aligned:
        assert isinstance(provision, dict)
        assert isinstance(provision.get("provision"), str)
        assert isinstance(provision.get("key"), dict)
        assert "text_similarity" in provision
        assert "textually_different" in provision

    # V8.3 impact-analysis result must be present.
    impact = result.get("impact")
    assert isinstance(impact, dict)

    assert impact.get("impact_level") in {
        "unknown",
        "low",
        "medium",
        "high",
    }
    assert isinstance(impact.get("summary"), str)
    assert impact["summary"].strip()

    assert isinstance(impact.get("changes"), list)
    assert isinstance(impact.get("potential_impacts"), list)
    assert isinstance(impact.get("affected_areas"), list)
    assert isinstance(impact.get("limitations"), list)
    assert isinstance(impact.get("evidence_provisions"), list)
    assert isinstance(impact.get("requires_further_research"), bool)

    # If the model returns evidence, every evidence label must correspond to
    # a provision actually present in the V8.2 comparison.
    available_provisions = {
        item.get("provision")
        for item in aligned
        if isinstance(item, dict) and item.get("provision")
    }
    available_provisions.update(
        item.get("provision")
        for item in only_in_a
        if isinstance(item, dict) and item.get("provision")
    )
    available_provisions.update(
        item.get("provision")
        for item in only_in_b
        if isinstance(item, dict) and item.get("provision")
    )

    for evidence in impact["evidence_provisions"]:
        assert evidence in available_provisions

    print("\n=== V8.4 REAL INTEGRATION ===")
    print(f"A documents       : {counts.get('documents_a')}")
    print(f"B documents       : {counts.get('documents_b')}")
    print(f"Aligned           : {len(aligned)}")
    print(f"Only A            : {len(only_in_a)}")
    print(f"Only B            : {len(only_in_b)}")
    print(f"Impact level      : {impact.get('impact_level')}")
    print(f"Summary           : {impact.get('summary')}")
    print(f"Evidence          : {impact.get('evidence_provisions')}")
    print(f"Further research  : {impact.get('requires_further_research')}")
    print("V8.4 REAL INTEGRATION: PASS")
