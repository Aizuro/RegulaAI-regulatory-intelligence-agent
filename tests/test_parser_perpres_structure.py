"""
Focused parser diagnostic for 2025perpres004.pdf.

Run from project root:
    py .\src\test_parser_perpres_structure.py

This diagnostic:
- DOES NOT touch Chroma/vectorstore.
- DOES NOT embed or index anything.
- Prints parser blocks for pages 14-17.
- Focuses on Pasal 20, 21, 30, 30A, and 31.
"""

from src.legal_parser import enrich_documents
from src.document_loader import load_single_pdf
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


PDF_PATH = PROJECT_ROOT / "data" / "documents" / "2025perpres004.pdf"
TARGET_PAGES = {14, 15, 16, 17}


def print_block(index, block):
    print("-" * 80)
    print(f"BLOCK #{index}")
    print(f"page              : {block.get('page')}")
    print(f"block_type        : {block.get('block_type')}")
    print(f"pasal             : {block.get('pasal')!r}")
    print(f"ayat              : {block.get('ayat')!r}")
    print(f"target_pasal      : {block.get('target_pasal')!r}")
    print(f"amendment_article : {block.get('amendment_article')!r}")
    content = (block.get("content") or "").replace("\n", " ")
    print(f"content           : {content[:700]}")


def main():
    print("=" * 80)
    print("FOCUSED PARSER STRUCTURE DIAGNOSTIC")
    print("=" * 80)
    print(f"PDF: {PDF_PATH}")
    print("Chroma: NOT USED")
    print()

    if not PDF_PATH.exists():
        print(f"ERROR: PDF tidak ditemukan: {PDF_PATH}")
        return 1

    loaded = load_single_pdf(
        str(PDF_PATH),
        enable_ocr_fallback=True,
        ocr_language="ind",
    )

    print(f"Loaded pages: {len(loaded)}")
    print()

    blocks, identity = enrich_documents(loaded)

    print("IDENTITY")
    print(identity)
    print()

    selected = [
        (idx, block)
        for idx, block in enumerate(blocks, start=1)
        if block.get("page") in TARGET_PAGES
    ]

    print(f"Blocks on pages {sorted(TARGET_PAGES)}: {len(selected)}")
    print()

    for idx, block in selected:
        print_block(idx, block)

    print()
    print("=" * 80)
    print("TARGET SUMMARY")
    print("=" * 80)

    checks = {
        "Pasal 20": [],
        "Pasal 21": [],
        "Pasal 30": [],
        "Pasal 30A": [],
        "Pasal 31": [],
    }

    for idx, block in selected:
        content = block.get("content") or ""
        pasal = str(block.get("pasal") or "")
        target = str(block.get("target_pasal") or "")

        if pasal == "20" or "Pasal 20" in content or "Pasal 2O" in content:
            checks["Pasal 20"].append((idx, block))

        if target == "21" or "Pasal 21" in content or "Pasal 2l" in content:
            checks["Pasal 21"].append((idx, block))

        if pasal == "30" or "Pasal 30" in content or "Pasal 3O" in content:
            checks["Pasal 30"].append((idx, block))

        if "Pasal 30A" in content or "Pasal 3OA" in content:
            checks["Pasal 30A"].append((idx, block))

        if pasal == "31" or target == "31" or "Pasal 31" in content:
            checks["Pasal 31"].append((idx, block))

    for name, items in checks.items():
        print(f"{name:10s}: {len(items)} candidate block(s)")
        for idx, block in items:
            print(
                f"  #{idx}: page={block.get('page')}, "
                f"type={block.get('block_type')}, "
                f"pasal={block.get('pasal')!r}, "
                f"target={block.get('target_pasal')!r}, "
                f"ayat={block.get('ayat')!r}"
            )

    print()
    print("=" * 80)
    print("STATUS: DIAGNOSTIC ONLY")
    print("Tidak ada perubahan pada Chroma/vectorstore.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
