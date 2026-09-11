"""Deterministic structural comparison for retrieved legal evidence.

This module does NOT decide legal meaning. It aligns retrieved provisions by
metadata and reports presence/absence plus a lightweight textual similarity
signal. The LLM remains responsible for natural-language explanation, while
this report provides an auditable intermediate representation.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def _docs(result: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    values = result.get(key) or []
    return [v for v in values if isinstance(v, dict)]


def _provision_key(doc: Dict[str, Any]) -> Tuple[str, str, str]:
    md = doc.get("metadata") or doc
    return (
        str(md.get("bab") or ""),
        str(md.get("pasal") or ""),
        str(md.get("ayat") or ""),
    )


def _text(doc: Dict[str, Any]) -> str:
    return str(doc.get("content") or doc.get("page_content") or "").strip()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a.lower(), b.lower()).ratio(), 4)


def _label(key: Tuple[str, str, str]) -> str:
    bab, pasal, ayat = key
    parts = []
    if bab:
        parts.append(f"Bab {bab}")
    if pasal:
        parts.append(f"Pasal {pasal}")
    if ayat:
        parts.append(f"Ayat {ayat}")
    return ", ".join(parts) if parts else "Unspecified provision"


def build_comparison_report(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build an auditable structural comparison from compare_regulations output."""
    a_docs = _docs(result, "documents_a")
    b_docs = _docs(result, "documents_b")

    a_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    b_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for doc in a_docs:
        a_map.setdefault(_provision_key(doc), doc)
    for doc in b_docs:
        b_map.setdefault(_provision_key(doc), doc)

    a_keys = set(a_map)
    b_keys = set(b_map)
    common = sorted(a_keys & b_keys)
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)

    aligned = []
    for key in common:
        a = a_map[key]
        b = b_map[key]
        sim = _similarity(_text(a), _text(b))
        aligned.append({
            "provision": _label(key),
            "key": {"bab": key[0], "pasal": key[1], "ayat": key[2]},
            "source_a": a.get("source"),
            "source_b": b.get("source"),
            "page_a": a.get("page"),
            "page_b": b.get("page"),
            "text_similarity": sim,
            "textually_different": sim < 0.98,
        })

    return {
        "type": "structural_comparison",
        "interpretation": "not_performed",
        "regulation_a": result.get("regulation_a"),
        "regulation_b": result.get("regulation_b"),
        "comparison_ready": bool(result.get("comparison_ready")),
        "counts": {
            "documents_a": len(a_docs),
            "documents_b": len(b_docs),
            "aligned_provisions": len(common),
            "only_in_a": len(only_a),
            "only_in_b": len(only_b),
        },
        "only_in_a": [{"key": {"bab": k[0], "pasal": k[1], "ayat": k[2]}, "provision": _label(k)} for k in only_a],
        "only_in_b": [{"key": {"bab": k[0], "pasal": k[1], "ayat": k[2]}, "provision": _label(k)} for k in only_b],
        "aligned_provisions": aligned,
        "missing_regulations": result.get("missing_regulations", []),
        "status": result.get("status"),
    }
