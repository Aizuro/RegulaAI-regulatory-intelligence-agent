"""V7.4 regulatory monitoring AI analysis.

Keeps the monitoring-analysis contract and prompt logic in Python/FastAPI.
n8n remains responsible for orchestration and routing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from src.llm_chain import get_llm


MonitoringEvent = Literal[
    "new_regulation",
    "updated_regulation",
    "removed_regulation",
]


class MonitoringAnalysis(BaseModel):
    """Structured result returned by the regulatory analysis LLM."""

    relevance: Literal["unknown", "low", "medium", "high"]
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    impact: str = Field(min_length=1)
    should_notify: bool


SYSTEM_PROMPT = """Kamu adalah Regulatory Intelligence Analyst untuk regulasi Indonesia.

Tugasmu menganalisis event monitoring regulasi berdasarkan DATA YANG DIBERIKAN SAJA.
Jangan mengarang isi regulasi, pasal, kewajiban, tanggal, atau dampak yang tidak
terdapat dalam data.

Aturan:
1. Gunakan Bahasa Indonesia yang formal dan ringkas.
2. Jika informasi yang tersedia tidak cukup untuk menentukan sesuatu, gunakan
   relevance='unknown' atau jelaskan keterbatasannya secara eksplisit.
3. Untuk UPDATED, bandingkan previous_regulation dengan regulation dan jelaskan
   perubahan yang benar-benar terlihat dari data.
4. Untuk NEW, jelaskan bahwa ini adalah regulasi baru berdasarkan data yang tersedia.
5. Untuk REMOVED, jelaskan bahwa regulasi tidak lagi muncul pada snapshot sumber.
6. should_notify=true hanya jika event tampak cukup penting untuk diberi perhatian;
   jangan menganggap semua event harus dinotifikasi.
7. Jangan memberikan nasihat hukum final. Analisis ini adalah ringkasan monitoring.
8. Jangan menyebut adanya snapshot sebelumnya kecuali previous_regulation benar-benar
   diberikan dalam DATA MONITORING.
9. Jika published_date bernilai null atau tidak tersedia, jangan menyebut tanggal
   publikasi dan jangan menyimpulkan tanggal tersebut dari sumber lain.
10. Jangan menambahkan fakta, tanggal, status hukum, isi pasal, kewajiban, atau dampak
    yang tidak dapat ditelusuri ke DATA MONITORING yang diberikan.
11. Selalu kembalikan SEMUA field yang diwajibkan schema:
   - relevance
   - summary
   - key_points
   - impact
   - should_notify
12. Jangan menghilangkan field apa pun meskipun informasi tidak cukup.
   Jika informasi tidak cukup:
   - relevance = "unknown"
   - summary harus menjelaskan keterbatasan data
   - key_points dapat berupa []
   - impact harus menjelaskan keterbatasan data
   - should_notify = false
13. Untuk UPDATED, bandingkan regulation dengan previous_regulation.
    Hanya nyatakan perubahan yang benar-benar terlihat dari kedua data tersebut.
    Jika hanya metadata yang berubah dan tidak ada informasi isi regulasi, jangan menilai dampaknya sebagai minimal/rendah. Nyatakan bahwa dampak tidak dapat ditentukan.
    Jangan mengarang perubahan isi, pasal, kewajiban, atau dampak.
14. Untuk REMOVED, event hanya menunjukkan bahwa regulasi tidak lagi muncul
    pada snapshot sumber.
    Jangan menyimpulkan bahwa regulasi sudah tidak berlaku secara hukum,
    dicabut, dibatalkan, atau tidak memiliki revisi kecuali informasi tersebut
    secara eksplisit tersedia dalam data monitoring.
"""


def _build_prompt(event: MonitoringEvent, regulation: Dict[str, Any], previous_regulation: Dict[str, Any] | None) -> str:
    payload = {
        "event": event,
        "regulation": regulation,
    }

    if previous_regulation is not None:
        payload["previous_regulation"] = previous_regulation

    return (
        SYSTEM_PROMPT
        + "\nDATA MONITORING:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nKembalikan HANYA satu JSON object yang valid, tanpa markdown code fence "
        "dan tanpa teks sebelum atau sesudah JSON.\n"
        "Gunakan field berikut:\n"
        '{"relevance":"unknown|low|medium|high",'
        '"summary":"string",'
        '"key_points":["string"],'
        '"impact":"string",'
        '"should_notify":true}'
    )


def analyze_monitoring_event(
    event: MonitoringEvent,
    regulation: Dict[str, Any],
    previous_regulation: Dict[str, Any] | None = None,
    llm=None,
) -> MonitoringAnalysis:
    """Analyze one monitoring event using the configured LLM.

    ``previous_regulation`` is required for UPDATED events and ignored for NEW/REMOVED
    only when it is absent. The LLM is instructed to stay grounded in supplied data.
    """
    if event == "updated_regulation" and previous_regulation is None:
        raise ValueError(
            "previous_regulation wajib tersedia untuk updated_regulation."
        )

    model = llm or get_llm()
    prompt = _build_prompt(event, regulation, previous_regulation)

    result = model.invoke(prompt)

    if hasattr(result, "content"):
        content = result.content
    else:
        content = result

    if not isinstance(content, str):
        raise ValueError(
            f"LLM monitoring analysis harus berupa JSON string, "
            f"mendapatkan {type(content).__name__}."
        )

    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM mengembalikan JSON monitoring yang tidak valid: {exc}"
        ) from exc

    return MonitoringAnalysis.model_validate(parsed)    
