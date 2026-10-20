
from __future__ import annotations

import json

from src.impact_analysis import ImpactAnalysis, analyze_comparison_impact
from src.agent_orchestrator import RegulatoryIntelligenceAgent, _tool_schemas
import src.v84_impact_tool as v84_impact_tool


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
        return FakeResponse(json.dumps({
            "impact_level": "unknown",
            "summary": "Evidence struktural belum cukup untuk menentukan dampak substantif.",
            "changes": ["Pasal 27"],
            "potential_impacts": ["Kemungkinan terdapat perbedaan penerapan yang perlu diverifikasi."],
            "affected_areas": [],
            "limitations": ["Teks lengkap ketentuan belum tersedia."],
            "evidence_provisions": ["Pasal 27"],
            "requires_further_research": True,
        }))


def comparison():
    return {
        "comparison_ready": True,
        "only_in_a": [{"provision": "Pasal 5"}],
        "only_in_b": [{"provision": "Pasal 8"}],
        "aligned_provisions": [
            {"provision": "Pasal 27", "textually_different": True}
        ],
    }


def test_v84_impact_tool_is_registered():
    names = {item["name"] for item in _tool_schemas()}
    assert "analyze_regulation_impact" in names


def test_v84_impact_analysis_uses_existing_comparison_evidence():
    result = analyze_comparison_impact(comparison(), llm=FakeLLM())
    assert isinstance(result, ImpactAnalysis)
    assert result.impact_level == "unknown"
    assert result.evidence_provisions == ["Pasal 27"]


def test_v84_plan_routes_impact_question_to_new_tool():
    question = "Bandingkan UU Nomor 1 Tahun 2024 dan UU Nomor 3 Tahun 2024, apa dampaknya?"
    assert RegulatoryIntelligenceAgent._build_plan(question) == [
        "analyze_regulation_impact"
    ]


def test_v84_tool_adapter_composes_retrieval_comparison_and_impact(monkeypatch):
    monkeypatch.setattr(
        v84_impact_tool,
        "retrieve_regulation",
        lambda identity: type("R", (), {
            "documents": [{"content": f"Pasal 27 {identity['regulation_number']}",
                           "metadata": {"pasal": "27"}}]
        })(),
    )
    monkeypatch.setattr(
        v84_impact_tool,
        "compare_retrieved_documents",
        lambda **kwargs: comparison(),
    )
    monkeypatch.setattr(
        v84_impact_tool,
        "analyze_comparison_impact",
        lambda report, llm=None: ImpactAnalysis(
            impact_level="unknown",
            summary="insufficient evidence",
            changes=["Pasal 27"],
            potential_impacts=[],
            affected_areas=[],
            limitations=["limited"],
            evidence_provisions=["Pasal 27"],
            requires_further_research=True,
        ),
    )

    result = v84_impact_tool.analyze_regulation_impact(
        {"document_type": "UU", "regulation_number": "1", "year": "2024"},
        {"document_type": "UU", "regulation_number": "3", "year": "2024"},
    )

    assert result["status"] == "ok"
    assert result["comparison_ready"] is True
    assert result["impact"]["impact_level"] == "unknown"
