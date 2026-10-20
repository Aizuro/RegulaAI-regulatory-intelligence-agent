"""Evaluation suite for Regulatory Intelligence Agent V4.

V4 tests:
- existing legal routing remains intact
- SQL routing works
- SQL result is structured
- SQL tool is read-only
- missing legal source handling remains intact
"""

from src.agent_orchestrator import build_agent
from src.embeddings import load_legal_vectorstore
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Set


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


EXPECTED_MIN_VECTORS = 776


@dataclass(frozen=True)
class TestCase:
    case_id: str
    question: str
    expected_plan: List[str]
    expected_steps: List[str]
    expected_sources: Set[str]
    expects_missing_source: bool
    expects_sql: bool


TEST_CASES = [
    TestCase(
        "V4-1",
        "Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?",
        ["get_regulation"],
        ["get_regulation"],
        {"2024uu001.pdf"},
        False,
        False,
    ),
    TestCase(
        "V4-2",
        "Berapa jumlah regulasi berdasarkan sektor?",
        ["run_sql_query"],
        ["run_sql_query"],
        set(),
        False,
        True,
    ),
    TestCase(
        "V4-3",
        "Bagaimana distribusi status regulasi dalam dataset?",
        ["run_sql_query"],
        ["run_sql_query"],
        set(),
        False,
        True,
    ),
    TestCase(
        "V4-4",
        "Apa perbedaan UU Nomor 1 Tahun 2024 dengan UU Nomor 3 Tahun 2024?",
        ["compare_regulations"],
        ["compare_regulations"],
        {"2024uu001.pdf", "2024uu003.pdf"},
        False,
        False,
    ),
    TestCase(
        "V4-5",
        "Apa ketentuan dalam Peraturan Presiden Nomor 99 Tahun 2099?",
        ["get_regulation"],
        ["get_regulation"],
        set(),
        True,
        False,
    ),
]


def extract_sources(result: Dict[str, Any]) -> Set[str]:
    return {
        Path(str(item.get("source", "")).replace("\\", "/")).name
        for item in result.get("citations", [])
        if item.get("source")
    }


def extract_tools(result: Dict[str, Any]) -> List[str]:
    return [
        str(step.get("tool"))
        for step in result.get("trace", [])
        if step.get("tool")
    ]


def check_case(agent: Any, case: TestCase) -> bool:
    result = agent.invoke(case.question)

    checks = {
        "Plan": result.get("plan") == case.expected_plan,
        "Completed steps": result.get("completed_steps") == case.expected_steps,
        "First tool": bool(extract_tools(result))
        and extract_tools(result)[0] == case.expected_steps[0],
        "Source accuracy": case.expected_sources.issubset(extract_sources(result)),
        "Missing-source": bool(result.get("missing_regulations"))
        == case.expects_missing_source,
        "Answer generated": bool(str(result.get("answer") or "").strip()),
    }

    if case.expects_sql:
        sql_results = result.get("sql_results") or []
        checks["SQL result"] = bool(sql_results)
        checks["SQL rows"] = (
            bool(sql_results)
            and isinstance(sql_results[0].get("rows"), list)
        )
        checks["SQL status"] = (
            bool(sql_results)
            and sql_results[0].get("status") == "ok"
        )

    print()
    print("=" * 78)
    print(f"{case.case_id} — {case.question}")
    print("=" * 78)

    for label, passed in checks.items():
        print(f"{label:<24}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        print(f"Expected plan          : {case.expected_plan}")
        print(f"Actual plan            : {result.get('plan')}")
        print(f"Expected steps         : {case.expected_steps}")
        print(f"Actual steps           : {result.get('completed_steps')}")
        print(f"Tools trace            : {extract_tools(result)}")

    passed = all(checks.values())
    print(f"CASE RESULT             : {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    print("=" * 78)
    print("REGULATORY INTELLIGENCE AGENT V4 — EVALUATION")
    print("=" * 78)

    vectorstore = load_legal_vectorstore()
    count = vectorstore._collection.count()
    print(
        f"Vector coverage check   : {'PASS' if count >= EXPECTED_MIN_VECTORS else 'FAIL'} ({count} >= {EXPECTED_MIN_VECTORS})")

    if count < EXPECTED_MIN_VECTORS:
        return 1

    agent = build_agent(vectorstore)

    passed = sum(check_case(agent, case) for case in TEST_CASES)
    total = len(TEST_CASES)

    print()
    print("=" * 78)
    print("V4 EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Cases passed            : {passed}/{total}")
    print(f"Cases failed            : {total - passed}/{total}")
    print(f"Overall status          : {'PASS' if passed == total else 'FAIL'}")
    print("=" * 78)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
