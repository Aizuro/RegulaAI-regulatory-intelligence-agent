from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field
from src.llm_chain import get_llm

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
5. REMOVED FROM SOURCE SNAPSHOT bukan berarti regulasi dicabut secara hukum
   (removed from a source snapshot does not prove the regulation was
   revoked/abolished).
6. potential_impacts dan affected_areas HARUS kosong jika comparison report
   hanya berisi struktur/provision labels tanpa isi teks pasal yang relevan.
7. Jika impact_level='unknown' dan tidak ada substantive text evidence,
   jangan mengisi potential_impacts atau affected_areas dengan dugaan umum.
8. Jangan menggunakan kata seperti "signifikan", "berdampak", atau klaim
   substantif lain jika evidence hanya menunjukkan perbedaan struktural.
9. evidence_provisions hanya boleh berisi provision label yang benar-benar ada
   dalam comparison report.
10. Jangan memberikan nasihat hukum final.
"""


def _compact_comparison_report(
    comparison_report: Dict[str, Any],
    max_one_sided: int = 10,
) -> Dict[str, Any]:
    """Keep the LLM prompt bounded while preserving comparison evidence."""
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
        "summary": "string",
        "changes": ["string"],
        "potential_impacts": ["string"],
        "affected_areas": ["string"],
        "limitations": ["string"],
        "evidence_provisions": ["string"],
        "requires_further_research": True,
    }
    compact_report = _compact_comparison_report(comparison_report)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Input variable: comparison_report\n"
        "Comparison report (JSON):\n"
        f"{json.dumps(compact_report, ensure_ascii=False, default=str)}\n\n"
        "Kembalikan HANYA satu JSON object yang valid, tanpa markdown code "
        "fence dan tanpa teks sebelum atau sesudah JSON.\n"
        "Jika report hanya menyediakan structural comparison tanpa isi teks "
        "pasal, gunakan impact_level='unknown', potential_impacts=[], dan "
        "affected_areas=[].\n"
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
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc.msg}.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Impact analysis JSON must be an object.")
    return parsed


def _available_provisions(report: Dict[str, Any]) -> set[str]:
    values: set[str] = set()

    for key in ("aligned_provisions", "only_in_a", "only_in_b"):
        for item in report.get(key, []) or []:
            if isinstance(item, dict):
                value = item.get("provision") or item.get("label")
                if value:
                    values.add(str(value))
            elif item:
                values.add(str(item))

    for item in report.get("textually_different", []) or []:
        if isinstance(item, dict):
            value = item.get("provision") or item.get("label")
            if value:
                values.add(str(value))

    return values


def _has_substantive_text_evidence(report: Dict[str, Any]) -> bool:
    """
    Structural comparison metadata is not substantive legal evidence.

    Return True only when the comparison explicitly carries text content for
    the compared provisions. This prevents generic impact speculation when the
    report contains only labels, counts, similarity scores, or page numbers.
    """
    text_fields = {
        "text_a",
        "text_b",
        "source_text",
        "target_text",
        "text_before",
        "text_after",
        "old_text",
        "new_text",
    }

    for key in ("aligned_provisions", "only_in_a", "only_in_b", "textually_different"):
        for item in report.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            if any(
                isinstance(item.get(field), str) and item.get(field).strip()
                for field in text_fields
            ):
                return True

    return False


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

    # Current V8.2 output is structural: it contains provision labels,
    # similarity, and page metadata but not the actual provision text.
    # In that situation, force the analysis to remain explicitly non-substantive.
    if not _has_substantive_text_evidence(comparison_report):
        # If the caller already supplied a compliant non-substantive result,
        # preserve object identity and avoid rewriting its summary unnecessarily.
        if (
            analysis.impact_level == "unknown"
            and not analysis.potential_impacts
            and not analysis.affected_areas
            and analysis.requires_further_research
        ):
            return analysis

        counts = comparison_report.get("counts", {})
        a = counts.get("only_in_a", len(
            comparison_report.get("only_in_a", []) or []))
        b = counts.get("only_in_b", len(
            comparison_report.get("only_in_b", []) or []))
        aligned = counts.get(
            "aligned_provisions",
            len(comparison_report.get("aligned_provisions", []) or []),
        )

        analysis = analysis.model_copy(
            update={
                "impact_level": "unknown",
                "summary": (
                    "Perbandingan struktural menunjukkan adanya perbedaan "
                    f"ketentuan antara kedua regulasi ({aligned} provision "
                    f"ter-align, {a} hanya di regulasi A, dan {b} hanya di "
                    "regulasi B). Dampak hukum substantif belum dapat "
                    "ditentukan berdasarkan evidence yang tersedia."
                ),
                "potential_impacts": [],
                "affected_areas": [],
                "requires_further_research": True,
            }
        )

    return analysis


def analyze_comparison_impact(
    comparison_report: Dict[str, Any],
    llm=None,
) -> ImpactAnalysis:
    if not comparison_report.get("comparison_ready"):
        raise ValueError(
            "Comparison report is not ready; comparison_ready=True is required."
        )

    model = llm or get_llm()
    result = model.invoke(build_impact_prompt(comparison_report))
    parsed = _parse_json_response(result)
    analysis = ImpactAnalysis.model_validate(parsed)

    return validate_impact_evidence(comparison_report, analysis)
