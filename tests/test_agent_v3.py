"""Evaluation suite for Regulatory Intelligence Agent V3.

This file focuses on V3 orchestration behavior:
- planning
- executed steps
- first selected tool
- source coverage
- missing-source handling
- answer generation
- structural comparison report

Run from the project root:
    python testing/test_agent_v3.py
"""

from src.agent_orchestrator import build_agent
from src.embeddings import load_legal_vectorstore
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Set


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPECTED_MIN_VECTORS = 776


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    case_id: str
    question: str
    expected_plan: List[str]
    expected_steps: List[str]
    expected_sources: Set[str]
    expects_missing_source: bool
    expects_comparison: bool


TEST_CASES = [
    TestCase(
        case_id="V3-1",
        question="Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?",
        expected_plan=["get_regulation"],
        expected_steps=["get_regulation"],
        expected_sources={"2024uu001.pdf"},
        expects_missing_source=False,
        expects_comparison=False,
    ),
    TestCase(
        case_id="V3-2",
        question=(
            "Apa perbedaan UU Nomor 1 Tahun 2024 "
            "dengan UU Nomor 3 Tahun 2024?"
        ),
        expected_plan=["compare_regulations"],
        expected_steps=["compare_regulations"],
        expected_sources={"2024uu001.pdf", "2024uu003.pdf"},
        expects_missing_source=False,
        expects_comparison=True,
    ),
    TestCase(
        case_id="V3-3",
        question=(
            "Apa perbedaan UU Nomor 1 Tahun 2024 "
            "dengan UU Nomor 3 Tahun 2024 terkait transaksi elektronik?"
        ),
        expected_plan=[
            "compare_regulations",
            "semantic_follow_up",
        ],
        expected_steps=[
            "compare_regulations",
            "search_regulations",
        ],
        expected_sources={"2024uu001.pdf", "2024uu003.pdf"},
        expects_missing_source=False,
        expects_comparison=True,
    ),
    TestCase(
        case_id="V3-4",
        question=(
            "Apa ketentuan mengenai pajak karbon dalam "
            "Peraturan Presiden Nomor 99 Tahun 2099?"
        ),
        expected_plan=["get_regulation"],
        expected_steps=["get_regulation"],
        expected_sources=set(),
        expects_missing_source=True,
        expects_comparison=False,
    ),
    TestCase(
        case_id="V3-5",
        question="Bagaimana ketentuan mengenai hak warga negara dalam UUD 1945?",
        expected_plan=["search_regulations"],
        expected_steps=["search_regulations"],
        expected_sources={"UUD1945.pdf"},
        expects_missing_source=False,
        expects_comparison=False,
    ),
]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 78


def print_header(title: str) -> None:
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def print_check(label: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{label:<24}: {status}")


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_sources(result: Dict[str, Any]) -> Set[str]:
    """Return normalized source filenames from the agent citations."""

    sources = set()

    for citation in result.get("citations", []):
        source = str(citation.get("source") or "").strip()

        if not source:
            continue

        sources.add(Path(source.replace("\\", "/")).name)

    return sources


def extract_tools(result: Dict[str, Any]) -> List[str]:
    """Return tools recorded in the agent trace."""

    return [
        str(step.get("tool"))
        for step in result.get("trace", [])
        if step.get("tool")
    ]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_plan(result: Dict[str, Any], case: TestCase) -> bool:
    return result.get("plan") == case.expected_plan


def check_steps(result: Dict[str, Any], case: TestCase) -> bool:
    return result.get("completed_steps") == case.expected_steps


def check_first_tool(result: Dict[str, Any], case: TestCase) -> bool:
    tools = extract_tools(result)

    if not tools:
        return False

    return tools[0] == case.expected_steps[0]


def check_sources(result: Dict[str, Any], case: TestCase) -> bool:
    actual_sources = extract_sources(result)
    return case.expected_sources.issubset(actual_sources)


def check_missing_source(result: Dict[str, Any], case: TestCase) -> bool:
    has_missing_source = bool(result.get("missing_regulations"))
    return has_missing_source == case.expects_missing_source


def check_answer(result: Dict[str, Any]) -> bool:
    answer = result.get("answer")
    return bool(str(answer or "").strip())


def check_comparison_report(
    result: Dict[str, Any],
    case: TestCase,
) -> bool:
    if not case.expects_comparison:
        return True

    comparison = result.get("comparison")

    return (
        isinstance(comparison, dict)
        and comparison.get("type") == "structural_comparison"
    )


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_case(agent: Any, case: TestCase) -> bool:
    print()
    print_header(f"{case.case_id} — {case.question}")

    result = agent.invoke(case.question)

    checks = {
        "Plan": check_plan(result, case),
        "Completed steps": check_steps(result, case),
        "First tool": check_first_tool(result, case),
        "Source accuracy": check_sources(result, case),
        "Missing-source": check_missing_source(result, case),
        "Answer generated": check_answer(result),
    }

    if case.expects_comparison:
        checks["Comparison report"] = check_comparison_report(result, case)

    for label, passed in checks.items():
        print_check(label, passed)

    if not all(checks.values()):
        print()
        print("Expected plan :", case.expected_plan)
        print("Actual plan   :", result.get("plan"))

        print("Expected steps:", case.expected_steps)
        print("Actual steps  :", result.get("completed_steps"))

        print("Expected src  :", sorted(case.expected_sources))
        print("Actual src    :", sorted(extract_sources(result)))

        print("Tools trace   :", extract_tools(result))

        if case.expects_comparison:
            comparison = result.get("comparison")
            print(
                "Comparison type:",
                comparison.get("type") if isinstance(
                    comparison, dict) else None,
            )

    passed = all(checks.values())
    print(f"CASE RESULT             : {'PASS' if passed else 'FAIL'}")

    return passed


def load_vectorstore():
    """Load the existing production Chroma vectorstore."""

    vectorstore = load_legal_vectorstore()
    vector_count = vectorstore._collection.count()

    print(f"Loaded collection 'legal_regulations' ({vector_count} vectors).")

    if vector_count < EXPECTED_MIN_VECTORS:
        print(
            "Vector coverage check: "
            f"FAIL ({vector_count} < {EXPECTED_MIN_VECTORS})"
        )
        return None

    print(
        "Vector coverage check: "
        f"PASS ({vector_count} >= {EXPECTED_MIN_VECTORS})"
    )

    return vectorstore


def main() -> int:
    print_header("REGULATORY INTELLIGENCE AGENT V3 — EVALUATION")

    vectorstore = load_vectorstore()

    if vectorstore is None:
        return 1

    agent = build_agent(vectorstore)

    passed_cases = 0

    for case in TEST_CASES:
        try:
            if run_case(agent, case):
                passed_cases += 1
        except Exception as exc:
            print()
            print(f"CASE ERROR              : {type(exc).__name__}: {exc}")
            print("CASE RESULT             : FAIL")

    total_cases = len(TEST_CASES)
    overall_pass = passed_cases == total_cases

    print()
    print_header("V3 EVALUATION SUMMARY")

    print(f"Cases passed            : {passed_cases}/{total_cases}")
    print(
        f"Cases failed            : {total_cases - passed_cases}/{total_cases}")
    print(
        f"Overall status          : "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    print(SEPARATOR)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
