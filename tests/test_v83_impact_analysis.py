import json

from langchain_core.documents import Document

from src.document_comparison import compare_retrieved_documents
from src.impact_analysis import (
    ImpactAnalysis,
    analyze_comparison_impact,
    build_impact_prompt,
    validate_impact_evidence,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def invoke(self, prompt):
        return FakeResponse(self.result)


def _doc(content, *, pasal="", ayat="", page=1, source="test.pdf"):
    return Document(
        page_content=content,
        metadata={
            "document_type": "UU",
            "regulation_number": "1",
            "year": "2024",
            "bab": "",
            "pasal": pasal,
            "ayat": ayat,
            "page": page,
            "source": source,
        },
    )


def _comparison_report():
    return compare_retrieved_documents(
        regulation_a={"document_type": "UU",
                      "regulation_number": "1", "year": "2024"},
        documents_a=[_doc("Ketentuan lama", pasal="10", source="old.pdf")],
        regulation_b={"document_type": "UU",
                      "regulation_number": "3", "year": "2024"},
        documents_b=[_doc("Ketentuan baru", pasal="10", source="new.pdf")],
    )


def _valid_json_output():
    return json.dumps({
        "impact_level": "medium",
        "summary": "Terdapat perbedaan teks pada provision yang ter-align.",
        "changes": ["Pasal 10 memiliki teks yang berbeda."],
        "potential_impacts": ["Potensi perubahan perlu diverifikasi terhadap isi lengkap."],
        "affected_areas": ["Perlu ditentukan berdasarkan konteks regulasi lengkap."],
        "limitations": ["Evidence hanya menunjukkan perbedaan struktural."],
        "evidence_provisions": ["Pasal 10"],
        "requires_further_research": True,
    })


def test_impact_prompt_contains_structural_evidence_and_grounding_rules():
    prompt = build_impact_prompt(_comparison_report())
    assert "comparison_report" in prompt
    assert "textually_different" in prompt
    assert "Jangan mengarang isi pasal" in prompt
    assert "only_in_a" in prompt
    assert "only_in_b" in prompt
    assert "HANYA satu JSON object" in prompt


def test_impact_analysis_returns_structured_contract():
    result = analyze_comparison_impact(
        _comparison_report(),
        llm=FakeLLM(_valid_json_output()),
    )
    assert isinstance(result, ImpactAnalysis)
    # The comparison fixture carries structural metadata but no explicit
    # before/after provision text, so substantive impact must remain unknown.
    assert result.impact_level == "unknown"
    assert result.evidence_provisions == ["Pasal 10"]
    assert result.potential_impacts == []
    assert result.affected_areas == []
    assert result.requires_further_research is True


def test_impact_analysis_requires_comparison_ready():
    try:
        analyze_comparison_impact(
            {"comparison_ready": False},
            llm=FakeLLM(_valid_json_output()),
        )
    except ValueError as exc:
        assert "comparison_ready=True" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_impact_prompt_preserves_removed_semantics():
    report = {
        "type": "structural_comparison",
        "comparison_ready": True,
        "status": "ready",
        "regulation_a": {},
        "regulation_b": {},
        "counts": {
            "documents_a": 10,
            "documents_b": 8,
            "aligned_provisions": 0,
            "only_in_a": 2,
            "only_in_b": 0,
        },
        "only_in_a": [
            {
                "key": {"bab": "", "pasal": "20", "ayat": ""},
                "provision": "Pasal 20",
            }
        ],
        "only_in_b": [],
        "aligned_provisions": [],
        "missing_regulations": [],
    }
    prompt = build_impact_prompt(report)
    assert "only_in_a" in prompt
    assert "dicabut" in prompt
    assert "revoked/abolished" in prompt


def test_impact_evidence_must_exist_in_comparison_report():
    report = {
        "comparison_ready": True,
        "only_in_a": [{"provision": "Pasal 10"}],
        "only_in_b": [],
        "aligned_provisions": [],
    }
    valid = ImpactAnalysis(
        impact_level="unknown",
        summary="Cukup.",
        changes=[],
        potential_impacts=[],
        affected_areas=[],
        limitations=[],
        evidence_provisions=["Pasal 10"],
        requires_further_research=True,
    )
    assert validate_impact_evidence(report, valid) is valid


def test_impact_evidence_rejects_unknown_provision():
    report = {
        "comparison_ready": True,
        "only_in_a": [{"provision": "Pasal 10"}],
        "only_in_b": [],
        "aligned_provisions": [],
    }
    invalid = ImpactAnalysis(
        impact_level="unknown",
        summary="Cukup.",
        changes=[],
        potential_impacts=[],
        affected_areas=[],
        limitations=[],
        evidence_provisions=["Pasal 99"],
        requires_further_research=True,
    )
    try:
        validate_impact_evidence(report, invalid)
    except ValueError as exc:
        assert "Pasal 99" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_invalid_llm_json_is_rejected():
    try:
        analyze_comparison_impact(
            _comparison_report(),
            llm=FakeLLM("not valid json"),
        )
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
