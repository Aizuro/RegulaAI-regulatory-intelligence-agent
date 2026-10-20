import importlib.util
import json
import sys
import types
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("impact_analysis.py")

fake_llm_chain = types.ModuleType("llm_chain")
fake_llm_chain.get_llm = lambda: None
sys.modules["llm_chain"] = fake_llm_chain

spec = importlib.util.spec_from_file_location("impact_analysis_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def comparison_report():
    return {
        "type": "structural_comparison",
        "interpretation": "not_performed",
        "comparison_ready": True,
        "counts": {
            "documents_a": 142,
            "documents_b": 101,
            "aligned_provisions": 2,
            "only_in_a": 63,
            "only_in_b": 55,
        },
        "aligned_provisions": [
            {"provision": "Unspecified provision", "text_similarity": 0.1217},
            {"provision": "Pasal 27", "text_similarity": 0.0166},
        ],
        "only_in_a": [{"provision": "Pasal 13"}],
        "only_in_b": [{"provision": "Pasal 26"}],
        "missing_regulations": [],
        "status": "ready",
    }


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt):
        return types.SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def test_structural_report_clears_speculative_impacts():
    payload = {
        "impact_level": "unknown",
        "summary": "Perbedaan signifikan berdampak pada kepatuhan.",
        "changes": ["Pasal 13 hanya di A", "Pasal 26 hanya di B"],
        "potential_impacts": ["Dapat memengaruhi kepatuhan pelaku usaha"],
        "affected_areas": ["Prosedur administratif"],
        "limitations": ["Tidak ada analisis isi pasal"],
        "evidence_provisions": ["Pasal 13", "Pasal 26", "Pasal 27"],
        "requires_further_research": False,
    }

    result = module.analyze_comparison_impact(
        comparison_report(),
        llm=FakeLLM(payload),
    )

    assert result.impact_level == "unknown"
    assert result.potential_impacts == []
    assert result.affected_areas == []
    assert result.requires_further_research is True
    assert "Dampak hukum substantif belum dapat ditentukan" in result.summary


def test_structural_report_preserves_valid_evidence():
    payload = {
        "impact_level": "unknown",
        "summary": "Ada perbedaan struktural.",
        "changes": ["Pasal 13 hanya di A"],
        "potential_impacts": [],
        "affected_areas": [],
        "limitations": ["Belum ada isi teks pasal."],
        "evidence_provisions": ["Pasal 13", "Pasal 27"],
        "requires_further_research": True,
    }

    result = module.analyze_comparison_impact(
        comparison_report(),
        llm=FakeLLM(payload),
    )

    assert result.evidence_provisions == ["Pasal 13", "Pasal 27"]


def test_invalid_evidence_is_rejected():
    payload = {
        "impact_level": "unknown",
        "summary": "Tidak cukup evidence.",
        "changes": [],
        "potential_impacts": [],
        "affected_areas": [],
        "limitations": [],
        "evidence_provisions": ["Pasal 999"],
        "requires_further_research": True,
    }

    try:
        module.analyze_comparison_impact(
            comparison_report(),
            llm=FakeLLM(payload),
        )
    except ValueError as exc:
        assert "outside comparison evidence" in str(exc)
    else:
        raise AssertionError("Expected invalid evidence to be rejected")


def test_prompt_explicitly_blocks_speculation():
    prompt = module.build_impact_prompt(comparison_report())
    assert "potential_impacts=[]" in prompt
    assert "affected_areas=[]" in prompt
    assert "structural comparison" in prompt.lower()
