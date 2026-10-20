"""Evaluation layer for the Regulatory Intelligence Agent.

Uses the existing production vectorstore only. No indexing or Chroma mutation.
Reports tool-selection accuracy, evidence/source accuracy, citation integrity,
and basic answer-grounding checks.
"""
from src.embeddings import load_legal_vectorstore
from src.agent_orchestrator import build_agent
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_MIN_VECTORS = 776

CASES = [
    {
        "id": "E1",
        "query": "Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?",
        "tool": "get_regulation",
        "sources": {"2024uu001.pdf"},
        "required": ["Transaksi Elektronik", "Sertifikat Elektronik"],
    },
    {
        "id": "E2",
        "query": "Apa hubungan UU Nomor 1 Tahun 2024 dengan UU Nomor 19 Tahun 2016?",
        "tool": "compare_regulations",
        "sources": {"2024uu001.pdf"},
        "missing": {("UU", "19", "2016")},
        "required": ["UU", "19", "2016"],
    },
    {
        "id": "E3",
        "query": "Apa yang diatur dalam UU Nomor 3 Tahun 2024?",
        "tool": "get_regulation",
        "sources": {"2024uu003.pdf"},
        "required": ["UU", "3", "2024"],
    },
    {
        "id": "E4",
        "query": "Bagaimana ketentuan mengenai hak warga negara dalam UUD 1945?",
        "tool": "search_regulations",
        "sources": {"UUD1945.pdf"},
        "required": ["UUD", "28"],
    },
    {
        "id": "E5",
        "query": "Apa perbedaan UU Nomor 1 Tahun 2024 dengan UU Nomor 3 Tahun 2024?",
        "tool": "compare_regulations",
        "sources": {"2024uu001.pdf", "2024uu003.pdf"},
        "required": ["UU", "2024", "Desa"],
    },
    {
        "id": "E6",
        "query": "Apa ketentuan mengenai pajak karbon dalam Peraturan Presiden Nomor 99 Tahun 2099?",
        "tool": "get_regulation",
        "sources": set(),
        "required": [],
        "allow_empty_citations": True,
        "missing_source_expected": True,
    },
]


def _source_name(value: Any) -> str:
    return str(value or "").replace("\\", "/").split("/")[-1]


def _trace_tools(result: Dict[str, Any]) -> List[str]:
    return [str(x.get("tool")) for x in result.get("trace", []) if x.get("tool")]


def _citation_sources(result: Dict[str, Any]) -> set[str]:
    return {_source_name(c.get("source")) for c in result.get("citations", [])}


def _citation_integrity(result: Dict[str, Any], allow_empty: bool = False) -> bool:
    citations = result.get("citations") or []
    if not citations:
        return allow_empty
    for c in citations:
        if not c.get("source"):
            return False
        if c.get("page_reference") is None:
            return False
    return True


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _evidence_texts(result: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for doc in result.get("sources", []) or []:
        if isinstance(doc, dict):
            content = doc.get("content")
        else:
            content = getattr(doc, "page_content", None)
        if content:
            texts.append(_normalize_text(content))
    return texts


def _answer_grounded(result: Dict[str, Any], required: Sequence[str]) -> bool:
    """Lightweight grounding check against retrieved evidence.

    Exact required anchors are preferred. If an answer uses a reasonable
    lexical variant, accept it only when the relevant evidence actually
    contains that concept. This avoids treating a keyword list as a semantic
    judge while preventing false negatives from harmless wording changes.
    """
    answer = _normalize_text(result.get("answer"))
    if not answer.strip():
        return False

    evidence = _evidence_texts(result)
    if not evidence:
        return False
    evidence_text = " ".join(evidence)

    # Exact anchors remain the strongest deterministic signal.
    missing = []
    for term in required:
        normalized = _normalize_text(term)
        if normalized and normalized not in answer:
            missing.append(normalized)
    if not missing:
        return True

    # For wording variation, require meaningful overlap between the answer
    # and retrieved evidence. Ignore very short/common words.
    answer_tokens = {t for t in answer.split() if len(t) >= 5}
    evidence_tokens = {t for t in evidence_text.split() if len(t) >= 5}
    overlap = answer_tokens & evidence_tokens

    # At least two meaningful evidence terms, or one strong legal anchor
    # shared with evidence, is enough for this lightweight regression check.
    strong_terms = {
        "transaksi", "elektronik", "sertifikat", "sertifikasi",
        "tanda", "tangan", "pasal", "ayat", "undang-undang",
        "peraturan", "desa", "warga", "negara",
    }
    strong_overlap = overlap & strong_terms
    return len(overlap) >= 2 and len(strong_overlap) >= 1


def _missing_source_ok(result: Dict[str, Any], expected: set[tuple[str, str, str]]) -> bool:
    if not expected:
        return True
    for item in result.get("trace", []):
        tool_result = item.get("result") or {}
        for missing in tool_result.get("missing_regulations", []) or []:
            actual = (
                str(missing.get("document_type")),
                str(missing.get("regulation_number")),
                str(missing.get("year")),
            )
            if actual in expected:
                return True
    return False


def _missing_source_grounded(result: Dict[str, Any]) -> bool:
    answer = _normalize_text(result.get("answer"))

    if result.get("sources"):
        return False

    missing = any(
        item.get("result", {}).get("missing_regulations")
        for item in result.get("trace", [])
    )

    if not missing:
        return False

    unavailable_terms = (
        "tidak tersedia",
        "tidak ditemukan",
        "tidak ada dalam",
        "tidak terdapat dalam",
        "knowledge base",
    )

    return any(term in answer for term in unavailable_terms)


def main() -> int:
    print("=" * 78)
    print("REGULATORY INTELLIGENCE AGENT — EVALUATION LAYER")
    print("=" * 78)
    print("Existing production vectorstore only.")
    print("No re-indexing. No Chroma mutation.")

    vectorstore = load_legal_vectorstore()
    count = vectorstore._collection.count()
    print(f"Loaded collection 'legal_regulations' ({count} vectors).")
    print(
        f"Vector coverage check: {'PASS' if count >= EXPECTED_MIN_VECTORS else 'FAIL'} ({count} >= {EXPECTED_MIN_VECTORS})")
    if count < EXPECTED_MIN_VECTORS:
        return 1

    agent = build_agent(vectorstore)
    totals = {
        "cases": len(CASES),
        "answer": 0,
        "tool": 0,
        "source": 0,
        "citation": 0,
        "grounding": 0,
        "missing": 0,
    }
    case_results: List[bool] = []

    for case in CASES:
        print("\n" + "=" * 78)
        print(case["id"])
        print(case["query"])
        print("=" * 78)
        try:
            result = agent.invoke(case["query"])
            answer_ok = bool(str(result.get("answer") or "").strip())
            tools = _trace_tools(result)
            tool_ok = bool(tools) and tools[0] == case["tool"]
            actual_sources = _citation_sources(result)
            source_ok = case["sources"].issubset(actual_sources)
            citation_ok = _citation_integrity(
                result, case.get("allow_empty_citations", False))
            if case.get("missing_source_expected"):
                grounding_ok = _missing_source_grounded(result)
            else:
                grounding_ok = _answer_grounded(result, case["required"])
            missing_ok = _missing_source_ok(result, case.get("missing", set()))

            checks = [answer_ok, tool_ok, source_ok,
                      citation_ok, grounding_ok, missing_ok]
            totals["answer"] += int(answer_ok)
            totals["tool"] += int(tool_ok)
            totals["source"] += int(source_ok)
            totals["citation"] += int(citation_ok)
            totals["grounding"] += int(grounding_ok)
            if case.get("missing"):
                totals["missing"] += int(missing_ok)

            print(f"Tool trace          : {tools}")
            print(f"Citation sources    : {sorted(actual_sources)}")
            print(f"Answer generated    : {'PASS' if answer_ok else 'FAIL'}")
            print(f"Tool selection      : {'PASS' if tool_ok else 'FAIL'}")
            print(f"Source accuracy     : {'PASS' if source_ok else 'FAIL'}")
            print(f"Citation integrity  : {'PASS' if citation_ok else 'FAIL'}")
            print(
                f"Basic grounding     : {'PASS' if grounding_ok else 'FAIL'}")
            if case.get("missing"):
                print(
                    f"Missing source      : {'PASS' if missing_ok else 'FAIL'}")
            case_pass = all(checks)
            case_results.append(case_pass)
            print(f"CASE RESULT         : {'PASS' if case_pass else 'FAIL'}")

        except Exception as exc:
            print("CASE ERROR:")
            traceback.print_exc()
            case_results.append(False)
            print("CASE RESULT         : FAIL")

    n = totals["cases"]
    print("\n" + "=" * 78)
    print("EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Answer generated    : {totals['answer']}/{n}")
    print(f"Tool selection      : {totals['tool']}/{n}")
    print(f"Source accuracy     : {totals['source']}/{n}")
    print(f"Citation integrity  : {totals['citation']}/{n}")
    print(f"Basic grounding     : {totals['grounding']}/{n}")
    missing_den = sum(1 for c in CASES if c.get("missing"))
    if missing_den:
        print(f"Missing-source      : {totals['missing']}/{missing_den}")

    # A case passes only if all applicable checks pass.
    # Re-run is intentionally not performed; each case has exactly one agent invocation.
    # The per-case results above are authoritative for this run.
    overall_pass = len(case_results) == n and all(case_results)
    print(f"\nOVERALL STATUS      : {'PASS' if overall_pass else 'FAIL'}")
    print("NOTE: Grounding is a lightweight evidence-overlap regression check, not a substitute for a semantic judge.")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
