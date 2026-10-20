"""Guard deterministik untuk pertanyaan pengguna (tanpa LLM, tanpa dependensi berat).

Tiga tugas:
  1. detect_uud_provision      -> "Pasal 23A UUD 1945" dirutekan ke get_regulation,
                                  bukan diserahkan ke LLM (yang bisa menjawab dari ingatan).
  2. sanitize_question         -> membuang klausa yang mencoba mengubah aturan sistem
                                  ("abaikan instruksi sebelumnya", "tampilkan system prompt").
  3. mentions_legal_reference  -> pertanyaan yang menyebut Pasal/UU/UUD/Perpres wajib
                                  didukung hasil tool, tidak boleh dijawab dari ingatan model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REFUSAL_INJECTION = (
    "Maaf, saya tidak dapat memenuhi permintaan itu. Saya hanya dapat membantu "
    "pertanyaan tentang peraturan perundang-undangan Indonesia."
)

# --------------------------------------------------------------------------- UUD 1945
_UUD_RE = re.compile(
    r"\b(?:uud|undang[-\s]?undang\s+dasar(?:\s+negara\s+republik\s+indonesia)?)"
    r"\s*(?:tahun\s*)?1945\b",
    re.IGNORECASE,
)
_PASAL_RE = re.compile(r"\bpasal\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
_AYAT_RE = re.compile(r"\bayat\s*\(?\s*(\d+[A-Za-z]?)\s*\)?", re.IGNORECASE)


def detect_uud_provision(question: str) -> Optional[Dict[str, str]]:
    """Kembalikan argumen get_regulation untuk 'Pasal X [ayat Y] UUD 1945', atau None.

    Konvensi metadata di vectorstore: pasal huruf besar ('23A'), ayat huruf kecil ('2a').
    Jika beberapa Pasal disebut, hanya yang pertama dipakai (batasan yang diketahui).
    """
    if not _UUD_RE.search(question):
        return None
    pasal = _PASAL_RE.search(question)
    if not pasal:
        return None
    args = {"document_type": "UUD", "year": "1945", "pasal": pasal.group(1).upper()}
    ayat = _AYAT_RE.search(question)
    if ayat:
        args["ayat"] = ayat.group(1).lower()
    return args


# --------------------------------------------------------------------------- injeksi
_INJECTION_PATTERNS = [
    r"\babaikan\s+(?:semua\s+|seluruh\s+|segala\s+)?(?:aturan|instruksi|perintah|prompt|batasan)",
    r"\blupakan\s+(?:semua\s+|seluruh\s+)?(?:aturan|instruksi|perintah|batasan)",
    r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|your)\s+"
    r"(?:instructions?|rules?|prompts?)",
    r"\bdisregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|your)\s+(?:instructions?|rules?)",
    r"\bsystem\s*prompt\b",
    r"\bprompt\s+sistem\b",
    r"\binstruksi\s+sistem\b",
    r"\b(?:bertindaklah|berperilakulah|berpura-?puralah)\s+sebagai\b",
    r"\bjailbreak\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Pemisah klausa: tanda baca akhir kalimat, atau kata sambung urutan ("lalu", "setelah itu").
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|\s*,?\s+(?:lalu|kemudian|setelah\s+itu|selanjutnya|and\s+then|then)\s+",
    re.IGNORECASE,
)


@dataclass
class GuardResult:
    clean_question: str
    injection_detected: bool = False
    dropped_clauses: List[str] = field(default_factory=list)

    @property
    def refuse(self) -> bool:
        """True jika seluruh pertanyaan hanyalah upaya injeksi."""
        return self.injection_detected and not self.clean_question.strip()


def sanitize_question(question: str) -> GuardResult:
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(question or "") if c and c.strip()]
    kept, dropped = [], []
    for clause in clauses:
        (dropped if _INJECTION_RE.search(clause) else kept).append(clause.strip())
    if not dropped:
        return GuardResult(clean_question=(question or "").strip())
    return GuardResult(
        clean_question=" ".join(kept).strip(),
        injection_detected=True,
        dropped_clauses=dropped,
    )


# --------------------------------------------------------------------------- rujukan hukum
_LEGAL_REF_RE = re.compile(
    r"\b(?:pasal|ayat|uud|undang[-\s]?undang|uu|perpres|perppu|perpu|"
    r"peraturan\s+(?:presiden|pemerintah|menteri|daerah))\b",
    re.IGNORECASE,
)


def mentions_legal_reference(question: str) -> bool:
    return bool(_LEGAL_REF_RE.search(question or ""))


# --------------------------------------------------------------------------- nasihat hukum
REPLY_LEGAL_ADVICE = (
    "Saya menyediakan informasi peraturan perundang-undangan, bukan nasihat hukum, "
    "dan tidak dapat menjamin hasil suatu perkara. Untuk kasus Anda, sebaiknya "
    "konsultasikan dengan advokat atau pengacara berlisensi. Jika Anda menyebutkan "
    "peraturan atau pasal tertentu, saya dapat membantu menjelaskan isinya."
)

_ADVICE_PATTERNS = [
    r"\b(?:pasti|dijamin|jamin(?:kan)?|pastikan)\b.{0,40}\b(?:menang|berhasil|lolos|bebas|untung)\b",
    r"\b(?:berikan|beri|minta|butuh|mau)\b.{0,30}\bkepastian\s+hukum\b",
    r"\bapakah\s+saya\s+(?:akan\s+)?(?:menang|kalah|bersalah|dihukum|dipenjara)\b",
    r"\bnasihat\s+hukum\s+(?:untuk|buat)\s+saya\b",
]
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), re.IGNORECASE)


def is_legal_advice_request(question: str) -> bool:
    """True jika pengguna meminta jaminan hasil/nasihat personal TANPA menyebut
    peraturan tertentu. Pertanyaan yang menyebut Pasal/UU tetap diproses normal."""
    q = question or ""
    return bool(_ADVICE_RE.search(q)) and not mentions_legal_reference(q)
