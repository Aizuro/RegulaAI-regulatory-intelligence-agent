"""Legal Chunker V2.

Post-processing for blocks emitted by legal_parser.py.
Focus: merge legal blocks that continue across adjacent PDF pages.
Raw source content is preserved; only structural metadata and chunk grouping are added.
"""

from copy import deepcopy
from typing import Any, Dict, List
import re


def _page_number(block: Dict[str, Any]):
    value = block.get("page")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _same_legal_unit(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True when two blocks represent the same Pasal/Ayat unit."""
    if a.get("block_type") != "ayat" or b.get("block_type") != "ayat":
        return False
    return (
        a.get("pasal") is not None
        and a.get("pasal") == b.get("pasal")
        and a.get("ayat") is not None
        and a.get("ayat") == b.get("ayat")
    )


def _adjacent_pages(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    pa = _page_number(a)
    pb = _page_number(b)
    return pa is not None and pb is not None and pb == pa + 1


def _remove_repeated_ayat_marker(text: str, ayat: str) -> str:
    """Remove a repeated leading ayat marker from a continuation block."""
    if not text:
        return text
    escaped = re.escape(str(ayat))
    pattern = rf"^\s*\({escaped}\)\s*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)


def _merge_content(first: str, continuation: str, ayat: str) -> str:
    continuation = _remove_repeated_ayat_marker(continuation or "", ayat)
    first = first or ""
    if not first:
        return continuation
    if not continuation:
        return first
    return first.rstrip() + "\n" + continuation.lstrip()


def merge_cross_page_continuations(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent-page blocks with the same Pasal + Ayat identity.

    This intentionally does not merge blocks merely because their text looks similar.
    Both blocks must be ayat blocks, have identical Pasal/Ayat metadata, and be on
    consecutive pages. This avoids merging repeated references or unrelated text.
    """
    if not blocks:
        return []

    result: List[Dict[str, Any]] = []

    for original in blocks:
        block = deepcopy(original)

        if result and _same_legal_unit(result[-1], block) and _adjacent_pages(result[-1], block):
            previous = result[-1]
            previous["content"] = _merge_content(
                previous.get("content", ""),
                block.get("content", ""),
                previous.get("ayat"),
            )
            previous["page_end"] = block.get("page")
            previous["source_pages"] = sorted(set(
                [str(x) for x in previous.get(
                    "source_pages", [previous.get("page")])]
                + [str(block.get("page"))]
            ))
            previous["is_cross_page"] = True
            previous["merged_blocks"] = previous.get("merged_blocks", 1) + 1
            continue

        block.setdefault("page_end", block.get("page"))
        block.setdefault("source_pages", [str(block.get("page"))])
        block.setdefault("is_cross_page", False)
        block.setdefault("merged_blocks", 1)
        result.append(block)

    return result


def chunk_legal_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Public V2 entry point."""
    return merge_cross_page_continuations(blocks)
