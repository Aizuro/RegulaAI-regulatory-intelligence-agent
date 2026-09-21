"""Legal Chunk Quality/Cleaning Layer V1.1.

Conservative cleanup after legal_parser.py + legal_chunker.py.

Design principle:
    For legal documents, false-positive deletion is worse than leaving OCR/layout
    noise. Only remove artifacts when the WHOLE line strongly matches a known
    page-layout pattern.

Raw `content` is never mutated. Cleaned text is stored in `clean_content`.
"""

import re
from copy import deepcopy
from typing import Any, Dict, List


_SK_NO_RE = re.compile(
    r"^SK\s*No\.?\s*[A-Za-z0-9 .:/_-]{2,}$",
    re.IGNORECASE,
)

# Standalone page numbers / page markers such as:
#   -4-
#   4
#   4-
_PAGE_NUMBER_RE = re.compile(
    r"^[\-–—]?\s*\d{1,3}\s*[\-–—]?$"
)

# Strong header/footer signatures. IMPORTANT:
# These match short standalone header/footer lines only.
# They intentionally do NOT match arbitrary prose beginning with
# "Presiden" or "Republik Indonesia".
_HEADER_EXACT_RE = re.compile(
    r"^(?:"
    r"PRESIDEN"
    r"|FRESIDEN"
    r"|REPUBLIK\s+INDONESIA"
    r"|REPUBLTK\s+INDONESIA"
    r"|REPUBUK\s+INDONESIA"
    r"|REPIJBUK\s+INDONESIA"
    r"|REPIJBLIK\s+INDONESIA"
    r"|REPUELIK\s+INDONESIA"
    r"|REPUELIK\s+TNDONESIA"
    r"|REPUBLIC\s+INDONESIA"
    r"|SALINAN"
    r")"
    r"\s*[,.:;]?$",
    re.IGNORECASE,
)

# Known standalone OCR header fragments observed in this corpus.
_KNOWN_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"ihtrEILtriIi"
    r"|LIK\s+iNf\.ftIf\+TA-l"
    r"|NEPUBLIK\s+INDONESIA"
    r"|BUK\s+INDONESIA"
    r"|FEPUELIK\s+INDONESIA"
    r")$",
    re.IGNORECASE,
)


def _is_layout_artifact(line: str) -> bool:
    """Return True only for high-confidence layout artifacts."""
    s = line.strip()

    if not s:
        return False

    if _SK_NO_RE.fullmatch(s):
        return True

    if _PAGE_NUMBER_RE.fullmatch(s):
        return True

    if _HEADER_EXACT_RE.fullmatch(s):
        return True

    if _KNOWN_FRAGMENT_RE.fullmatch(s):
        return True

    return False


def clean_chunk_content(text: str) -> str:
    """Remove only high-confidence page-layout artifacts."""
    if not text:
        return ""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept = []

    for line in lines:
        if _is_layout_artifact(line):
            continue
        kept.append(line.rstrip())

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def assess_chunk_quality(raw: str, cleaned: str) -> Dict[str, Any]:
    raw = raw or ""
    cleaned = cleaned or ""

    removed = max(0, len(raw) - len(cleaned))

    return {
        "raw_chars": len(raw),
        "clean_chars": len(cleaned),
        "removed_chars": removed,
        "changed": raw != cleaned,
        "cleaning_version": "v1.1",
    }


def clean_legal_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return deep copies with cleaned content and quality metadata."""
    result = []

    for chunk in chunks:
        item = deepcopy(chunk)

        raw = item.get("content", "")
        cleaned = clean_chunk_content(raw)

        item["clean_content"] = cleaned
        item["quality"] = assess_chunk_quality(raw, cleaned)

        result.append(item)

    return result
