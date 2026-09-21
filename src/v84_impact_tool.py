
from __future__ import annotations

from typing import Any, Dict

from document_retrieval import retrieve_regulation
from document_comparison import compare_retrieved_documents
from impact_analysis import analyze_comparison_impact


def analyze_regulation_impact(
    regulation_a: Dict[str, Any],
    regulation_b: Dict[str, Any],
    llm=None,
) -> Dict[str, Any]:
    """V8.4 agent-facing tool: retrieve -> compare -> grounded impact analysis."""

    result_a = retrieve_regulation(regulation_a)
    result_b = retrieve_regulation(regulation_b)

    docs_a = list(result_a.documents or [])
    docs_b = list(result_b.documents or [])

    if not docs_a or not docs_b:
        return {
            "status": "missing_source",
            "comparison_ready": False,
            "regulation_a": regulation_a,
            "regulation_b": regulation_b,
            "impact": None,
            "message": "One or both requested regulations are unavailable.",
        }

    comparison = compare_retrieved_documents(
        regulation_a=regulation_a,
        documents_a=docs_a,
        regulation_b=regulation_b,
        documents_b=docs_b,
    )

    if not comparison.get("comparison_ready"):
        return {
            "status": "comparison_not_ready",
            "comparison_ready": False,
            "regulation_a": regulation_a,
            "regulation_b": regulation_b,
            "comparison": comparison,
            "impact": None,
        }

    impact = analyze_comparison_impact(comparison, llm=llm)

    return {
        "status": "ok",
        "comparison_ready": True,
        "regulation_a": regulation_a,
        "regulation_b": regulation_b,
        "comparison": comparison,
        "impact": impact.model_dump() if hasattr(impact, "model_dump") else impact,
    }
