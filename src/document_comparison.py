"""V8.2 document comparison adapter.

Reuses the existing V3 ``comparison_engine.build_comparison_report``.
This module only adapts V8.1 DocumentRetrievalResult objects into the
serialized shape expected by the existing comparison engine.

The comparison engine remains deterministic and does not perform legal
interpretation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from comparison_engine import build_comparison_report


def _serialize_document(doc: Any) -> Dict[str, Any]:
    """Convert a LangChain Document-like object into comparison input."""
    metadata = dict(getattr(doc, "metadata", {}) or {})
    content = str(
        getattr(doc, "page_content", None)
        or getattr(doc, "content", None)
        or ""
    )
    return {
        "content": content,
        "metadata": metadata,
        **metadata,
    }


def _serialize_documents(documents: Iterable[Any]) -> list[Dict[str, Any]]:
    return [_serialize_document(doc) for doc in documents]


def build_v82_comparison_input(
    *,
    regulation_a: Dict[str, str],
    documents_a: Iterable[Any],
    regulation_b: Dict[str, str],
    documents_b: Iterable[Any],
) -> Dict[str, Any]:
    """Build the existing V3 comparison-engine input contract."""
    docs_a = _serialize_documents(documents_a)
    docs_b = _serialize_documents(documents_b)

    return {
        "regulation_a": dict(regulation_a),
        "regulation_b": dict(regulation_b),
        "documents_a": docs_a,
        "documents_b": docs_b,
        "comparison_ready": bool(docs_a and docs_b),
        "missing_regulations": [],
        "status": "ready" if docs_a and docs_b else "missing_source",
    }


def compare_retrieved_documents(
    *,
    regulation_a: Dict[str, str],
    documents_a: Iterable[Any],
    regulation_b: Dict[str, str],
    documents_b: Iterable[Any],
) -> Dict[str, Any]:
    """Run the existing deterministic V3 comparison engine on V8.1 results."""
    comparison_input = build_v82_comparison_input(
        regulation_a=regulation_a,
        documents_a=documents_a,
        regulation_b=regulation_b,
        documents_b=documents_b,
    )

    return build_comparison_report(comparison_input)
