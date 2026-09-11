"""
Legal tools V1 — tools for the Regulatory Intelligence Agent.

These tools operate on the existing legal Chroma collection.
No ingestion or re-indexing is performed here.
"""

from typing import Any, Dict, List, Optional
import re

from langchain_core.documents import Document


def _metadata_filter(
    document_type: Optional[str] = None,
    regulation_number: Optional[str] = None,
    year: Optional[str] = None,
    pasal: Optional[str] = None,
    ayat: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    conditions = []

    values = [
        ("document_type", document_type),
        ("regulation_number", regulation_number),
        ("year", year),
        ("pasal", pasal),
        ("ayat", ayat),
    ]

    for key, value in values:
        if value is not None and value != "":
            conditions.append({key: {"$eq": str(value)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _serialize_document(doc: Document) -> Dict[str, Any]:
    m = doc.metadata
    return {
        "content": doc.page_content,
        "source": m.get("source"),
        "document_type": m.get("document_type"),
        "regulation_number": m.get("regulation_number"),
        "year": m.get("year"),
        "title": m.get("title"),
        "bab": m.get("bab"),
        "pasal": m.get("pasal"),
        "ayat": m.get("ayat"),
        "page": m.get("page"),
        "page_end": m.get("page_end"),
        "is_cross_page": m.get("is_cross_page", False),
    }


def search_regulations(
    vectorstore,
    query: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic search over the legal corpus.

    Use this when the agent needs to discover relevant regulations
    or provisions and the exact legal identifier is not yet known.
    """
    if not query or not query.strip():
        raise ValueError("query tidak boleh kosong.")
    if k < 1:
        raise ValueError("k harus >= 1.")

    docs = vectorstore.similarity_search(query.strip(), k=k)
    return [_serialize_document(doc) for doc in docs]


def get_regulation(
    vectorstore,
    document_type: Optional[str] = None,
    regulation_number: Optional[str] = None,
    year: Optional[str] = None,
    pasal: Optional[str] = None,
    ayat: Optional[str] = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Exact metadata lookup for a known regulation/provision.

    Examples:
      get_regulation(vectorstore, "UU", "1", "2024")
      get_regulation(vectorstore, "UU", "1", "2024", "17", "2a")
    """
    if k < 1:
        raise ValueError("k harus >= 1.")

    metadata_filter = _metadata_filter(
        document_type=document_type,
        regulation_number=regulation_number,
        year=year,
        pasal=pasal,
        ayat=ayat,
    )

    if metadata_filter is None:
        raise ValueError(
            "get_regulation membutuhkan minimal satu identifier legal."
        )

    # A neutral query is used because the metadata filter determines identity.
    docs = vectorstore.similarity_search(
        "ketentuan peraturan",
        k=k,
        filter=metadata_filter,
    )
    return [_serialize_document(doc) for doc in docs]


def compare_regulations(
    vectorstore,
    regulation_a: Dict[str, str],
    regulation_b: Dict[str, str],
    k: int = 20,
) -> Dict[str, Any]:
    """
    Retrieve two regulations and report whether both are available.

    This tool performs retrieval only. It does not ask the LLM to invent
    missing regulations or perform the legal interpretation itself.
    """
    required = ("document_type", "regulation_number", "year")

    for label, regulation in (
        ("regulation_a", regulation_a),
        ("regulation_b", regulation_b),
    ):
        missing = [key for key in required if not regulation.get(key)]
        if missing:
            raise ValueError(
                f"{label} kurang identifier wajib: {', '.join(missing)}"
            )

    docs_a = get_regulation(
        vectorstore,
        document_type=regulation_a["document_type"],
        regulation_number=regulation_a["regulation_number"],
        year=regulation_a["year"],
        k=k,
    )

    docs_b = get_regulation(
        vectorstore,
        document_type=regulation_b["document_type"],
        regulation_number=regulation_b["regulation_number"],
        year=regulation_b["year"],
        k=k,
    )

    missing_regulations = []

    if not docs_a:
        missing_regulations.append(regulation_a)

    if not docs_b:
        missing_regulations.append(regulation_b)

    return {
        "regulation_a": regulation_a,
        "regulation_b": regulation_b,
        "documents_a": docs_a,
        "documents_b": docs_b,
        "comparison_ready": bool(docs_a and docs_b),
        "missing_regulations": missing_regulations,
        "status": "ready" if docs_a and docs_b else "missing_source",
    }
