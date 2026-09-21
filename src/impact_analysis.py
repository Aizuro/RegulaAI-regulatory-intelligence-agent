from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field
from llm_chain import get_llm

ImpactLevel = Literal["unknown", "low", "medium", "high"]


class ImpactAnalysis(BaseModel):
    impact_level: ImpactLevel
    summary: str = Field(min_length=1)
    changes: list[str] = Field(default_factory=list)
    potential_impacts: list[str] = Field(default_factory=list)
    affected_areas: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_provisions: list[str] = Field(default_factory=list)
    requires_further_research: bool


SYSTEM_PROMPT = """Kamu adalah Regulatory Intelligence Analyst untuk regulasi Indonesia.

Analisis hanya berdasarkan comparison report yang diberikan.
Jangan mengarang isi pasal, kewajiban, sanksi, tanggal berlaku, pihak terdampak,
atau dampak hukum.

Aturan:
1. Gunakan Bahasa Indonesia formal dan ringkas.
2. Jika evidence tidak cukup untuk menentukan dampak substantif, gunakan
   impact_level='unknown'.
3. only_in_a/only_in_b bukan bukti otomatis bahwa ketentuan dicabut atau baru
   berlaku.
4. textually_different bukan bukti otomatis adanya perubahan hukum substantif.
5. REMOVED FROM SOURCE SNAPSHOT bukan berarti regulasi dicabut secara hukum.
   Only-in-a/removed from source snapshot juga bukan otomatis berarti
   revoked/abolished.
6. potential_impacts harus ditulis sebagai kemungkinan, bukan fakta.
7. evidence_provisions hanya boleh berisi provision label yang benar-benar ada
   dalam comparison report.
8. Jangan memberikan nasihat hukum final.
"""


def _compact_comparison_report(
    comparison_report: Dict[str, Any],
    max_one_sided: int = 10,
) -> Dict[str, Any]:
    """
    Reduce prompt size and bound the amount of evidence the LLM needs to
    reason over. Counts are preserved, while only a small sample of one-sided
    provisions is included to avoid excessively long JSON responses.
    """
    return {
        "type": comparison_report.get("type"),
        "interpretation": comparison_report.get("interpretation"),
        "regulation_a": comparison_report.get("regulation_a"),
        "regulation_b": comparison_report.get("regulation_b"),
        "comparison_ready": comparison_report.get("comparison_ready"),
        "counts": comparison_report.get("counts", {}),
        "aligned_provisions": comparison_report.get("aligned_provisions", []),
        "only_in_a": (comparison_report.get("only_in_a", []) or [])[:max_one_sided],
        "only_in_b": (comparison_report.get("only_in_b", []) or [])[:max_one_sided],
        "missing_regulations": comparison_report.get("missing_regulations", []),
        "status": comparison_report.get("status"),
    }


def build_impact_prompt(comparison_report: Dict[str, Any]) -> str:
    schema = {
        "impact_level": "unknown|low|medium|high",
        "summary": "string, maksimal 2 kalimat",
        "changes": ["maksimal 3 item, string singkat"],
        "potential_impacts": ["maksimal 3 item, string singkat"],
        "affected_areas": ["maksimal 3 item, string singkat"],
        "limitations": ["maksimal 3 item, string singkat"],
        "evidence_provisions": ["maksimal 5 item, harus persis dari evidence"],
        "requires_further_research": True,
    }

    compact_report = _compact_comparison_report(comparison_report)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "comparison_report (ringkas):\n"
        f"{json.dumps(compact_report, ensure_ascii=False, default=str)}\n\n"
        "Kembalikan HANYA satu JSON object yang valid. "
        "Jangan gunakan markdown code fence. "
        "Jangan menambahkan teks sebelum atau sesudah JSON.\n"
        "PENTING: output harus sangat ringkas agar JSON selesai sepenuhnya. "
        "Jangan mengulang daftar provision dari input. "
        "Gunakan maksimal 3 item untuk changes, potential_impacts, "
        "affected_areas, dan limitations. "
        "Gunakan maksimal 5 evidence_provisions. "
        "Summary maksimal 2 kalimat.\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def _parse_json_response(result: Any) -> Dict[str, Any]:
    content = getattr(result, "content", result)

    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    if not isinstance(content, str):
        raise ValueError("LLM response must contain JSON text.")

    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid JSON response from LLM: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Impact analysis JSON must be an object.")

    return parsed


def _available_provisions(report: Dict[str, Any]) -> set[str]:
    values: set[str] = set()

    for key in ("aligned_provisions", "only_in_a", "only_in_b"):
        for item in report.get(key, []) or []:
            if isinstance(item, dict):
                for field in ("provision", "label", "key"):
                    value = item.get(field)
                    if value:
                        values.add(str(value))
            elif item:
                values.add(str(item))

    for item in report.get("textually_different", []) or []:
        if isinstance(item, dict):
            for field in ("provision", "label", "key"):
                value = item.get(field)
                if value:
                    values.add(str(value))

    return values


def validate_impact_evidence(
    comparison_report: Dict[str, Any],
    analysis: ImpactAnalysis,
) -> ImpactAnalysis:
    available = _available_provisions(comparison_report)
    unknown = [x for x in analysis.evidence_provisions if x not in available]

    if unknown:
        raise ValueError(
            "Impact analysis referenced provisions outside comparison evidence: "
            + ", ".join(unknown)
        )

    return analysis


def analyze_comparison_impact(
    comparison_report: Dict[str, Any],
    llm=None,
) -> ImpactAnalysis:
    if not comparison_report.get("comparison_ready"):
        raise ValueError("comparison_ready=True is required before impact analysis.")

    model = llm or get_llm()
    result = model.invoke(build_impact_prompt(comparison_report))
    parsed = _parse_json_response(result)
    analysis = ImpactAnalysis.model_validate(parsed)

    return validate_impact_evidence(comparison_report, analysis)
