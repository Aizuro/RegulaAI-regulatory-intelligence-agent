"""
Regression test for legal parser metadata.

Run from project root:
    py .\src\test_parser_regression.py

This test reads the actual corpus through the production ingestion pipeline
and verifies that the parser no longer produces the four known bad Pasal
metadata values in 2025perpres004.pdf.

IMPORTANT:
- This test does NOT touch Chroma/vectorstore.
- This test does NOT embed anything.
- This test does NOT modify source PDFs.
"""

from src.legal_ingestion import build_legal_documents
from pathlib import Path
import sys

# Make src imports work when executed from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


CORPUS_DIR = PROJECT_ROOT / "data" / "documents"
TARGET_SOURCE = "2025perpres004.pdf"

EXPECTED_TOTAL = 776
BAD_PASAL_VALUES = {"2O", "2l", "3O", "3OA"}


def main():
    print("=" * 80)
    print("LEGAL PARSER REGRESSION TEST")
    print("=" * 80)
    print(f"Corpus : {CORPUS_DIR}")
    print("Chroma : NOT USED")
    print()

    if not CORPUS_DIR.exists():
        print(f"ERROR: corpus folder not found: {CORPUS_DIR}")
        return 1

    documents, reports = build_legal_documents(
        folder_path=str(CORPUS_DIR),
        enable_ocr_fallback=True,
        ocr_language="ind",
    )

    print(f"Total documents/chunks : {len(documents)}")
    print()

    failures = []

    # ------------------------------------------------------------------
    # 1. Total corpus regression
    # ------------------------------------------------------------------
    if len(documents) != EXPECTED_TOTAL:
        failures.append(
            f"TOTAL_COUNT expected {EXPECTED_TOTAL}, got {len(documents)}"
        )
        print(
            f"[FAIL] Total count: expected {EXPECTED_TOTAL}, got {len(documents)}")
    else:
        print(f"[PASS] Total count: {len(documents)}")

    # ------------------------------------------------------------------
    # 2. Find the target Perpres records
    # ------------------------------------------------------------------
    target_docs = [
        doc for doc in documents
        if doc.metadata.get("source") == TARGET_SOURCE
    ]

    print(f"[INFO] {TARGET_SOURCE}: {len(target_docs)} documents")

    # ------------------------------------------------------------------
    # 3. No known bad OCR Pasal metadata
    # ------------------------------------------------------------------
    bad_records = [
        doc for doc in target_docs
        if str(doc.metadata.get("pasal", "")) in BAD_PASAL_VALUES
        or str(doc.metadata.get("target_pasal", "")) in BAD_PASAL_VALUES
    ]

    if bad_records:
        failures.append(
            f"BAD_OCR_METADATA found {len(bad_records)} records"
        )
        print(f"[FAIL] Bad OCR Pasal metadata: {len(bad_records)}")
        for doc in bad_records:
            m = doc.metadata
            print(
                f"  pasal={m.get('pasal')!r}, "
                f"target_pasal={m.get('target_pasal')!r}, "
                f"page={m.get('page')}-{m.get('page_end')}, "
                f"content={doc.page_content[:180]!r}"
            )
    else:
        print("[PASS] No bad OCR Pasal metadata (2O/2l/3O/3OA)")

    # ------------------------------------------------------------------
    # 4. Check Pasal 20
    # ------------------------------------------------------------------
    pasal20 = [
        doc for doc in target_docs
        if str(doc.metadata.get("pasal")) == "20"
        and "Kantor Bersama Samsat" in doc.page_content
    ]

    if not pasal20:
        failures.append("PASAL_20 not found")
        print("[FAIL] Pasal 20 substantive block not found")
    else:
        print(
            f"[PASS] Pasal 20 detected: {len(pasal20)} block(s), "
            f"page(s)={sorted(set(str(d.metadata.get('page')) for d in pasal20))}"
        )

    # ------------------------------------------------------------------
    # 5. Amendment instruction for Pasal 21 must not inherit Pasal 20
    # ------------------------------------------------------------------
    pasal21_instruction = [
        doc for doc in target_docs
        if "Pasal 21 diubah" in doc.page_content
        and str(doc.metadata.get("target_pasal")) == "21"
    ]

    inherited = [
        doc for doc in pasal21_instruction
        if doc.metadata.get("pasal") is not None
    ]

    if not pasal21_instruction:
        failures.append("PASAL_21_TARGET not found")
        print("[FAIL] Pasal 21 amendment instruction not found")
    elif inherited:
        failures.append(
            f"PASAL_21_INHERITED_PASAL found {len(inherited)} record(s)"
        )
        print("[FAIL] Pasal 21 instruction still has inherited pasal:")
        for doc in inherited:
            print(doc.metadata)
    else:
        print("[PASS] Pasal 21 amendment instruction does not inherit Pasal")

    # ------------------------------------------------------------------
    # 6. Check Pasal 30
    # ------------------------------------------------------------------
    pasal30 = [
        doc for doc in target_docs
        if str(doc.metadata.get("pasal")) == "30"
        and "Pelaksana Kantor Bersama Samsat" in doc.page_content
    ]

    if not pasal30:
        failures.append("PASAL_30 not found")
        print("[FAIL] Pasal 30 substantive block not found")
    else:
        print(
            f"[PASS] Pasal 30 detected: {len(pasal30)} block(s), "
            f"page(s)={sorted(set(str(d.metadata.get('page')) for d in pasal30))}"
        )

    # ------------------------------------------------------------------
    # 7. Amendment instruction inserting Pasal 30A must not inherit Pasal 30
    # ------------------------------------------------------------------
    pasal30a_instruction = [
        doc for doc in target_docs
        if "Pasal 3OA" in doc.page_content
        and str(doc.metadata.get("target_pasal")) == "30"
        and doc.metadata.get("block_type") == "amendment_instruction"
    ]

    inherited30 = [
        doc for doc in pasal30a_instruction
        if doc.metadata.get("pasal") is not None
    ]

    if not pasal30a_instruction:
        failures.append("PASAL_30A_INSTRUCTION not found")
        print("[FAIL] Pasal 30A insertion instruction not found")
    elif inherited30:
        failures.append(
            f"PASAL_30A_INHERITED_PASAL found {len(inherited30)} record(s)"
        )
        print("[FAIL] Pasal 30A instruction still has inherited pasal:")
        for doc in inherited30:
            print(doc.metadata)
    else:
        print("[PASS] Pasal 30A insertion instruction does not inherit Pasal 30")

    # ------------------------------------------------------------------
    # 8. Print source report for quick sanity check
    # ------------------------------------------------------------------
    target_report = next(
        (r for r in reports if r.get("source") == TARGET_SOURCE),
        None,
    )

    if target_report:
        print()
        print("[INFO] Target source report:")
        print(f"  blocks   : {target_report.get('blocks')}")
        print(f"  chunks   : {target_report.get('chunks')}")
        print(f"  documents: {target_report.get('documents')}")

    print()
    print("=" * 80)

    if failures:
        print("STATUS: FAIL")
        print()
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        print()
        print("JANGAN re-index Chroma dulu.")
        return 1

    print("STATUS: PASS")
    print("Parser + ingestion regression passed.")
    print("Aman untuk lanjut ke migrasi 4 metadata lama di Chroma.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
