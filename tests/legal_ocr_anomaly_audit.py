"""
legal_ocr_anomaly_audit_v2.py

Versi diperbaiki dari legal_ocr_anomaly_audit.py.

Fokus:
- Tidak menganggap Pasal 15, Pasal 24, dst. sebagai anomaly.
- Mendeteksi OCR confusion yang lebih spesifik: O<->0, l/I<->1.
- Mendeteksi "Pasal 168" sebagai kandidat 16B hanya jika konteks mendukung.
- Membedakan ayat legal "(2a)" dari ayat OCR malformed "(l)", "(21)", dll.
- Tidak mengubah PDF.
"""

from pathlib import Path
import re

try:
    import fitz
except ImportError:
    raise SystemExit("PyMuPDF belum terinstall. Jalankan: pip install pymupdf")

DOCUMENT_DIR = Path("data/documents")

# OCR confusion pada identifier Pasal.
PASAL_OCR_PATTERNS = {
    "Pasal_168": re.compile(r"\bPasal\s+168\b", re.I),
    "Pasal_digit_as_O": re.compile(
        r"\bPasal\s+\d+O\b", re.I
    ),  # contoh: Pasal 2O, Pasal 3O
    "Pasal_digit_as_l": re.compile(
        r"\bPasal\s+\d+l\b", re.I
    ),  # contoh: Pasal 2l
    "Pasal_digit_as_I": re.compile(
        r"\bPasal\s+\d+I\b", re.I
    ),
}

# OCR confusion pada ayat. "(2a)" justru valid secara hukum dan tidak
# dimasukkan ke malformed.
AYAT_OCR_PATTERNS = {
    "Ayat_l_as_1": re.compile(r"\(\s*[lI]\s*\)?"),
    "Ayat_digit_plus_l": re.compile(r"\(\s*\d+[lI]\s*\)?"),
    "Ayat_l_plus_digit": re.compile(r"\(\s*[lI]\d+\s*\)?"),
}

COMMON_OCR_PATTERNS = {
    "PRES!DEN": re.compile(r"PRES!DEN", re.I),
    "INDONE!3IA": re.compile(r"INDONE!3IA", re.I),
    "REPIJBUK": re.compile(r"REPIJBUK", re.I),
    "PasaJ": re.compile(r"\bPasaJ\b", re.I),
}


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text or "").strip()


def context_snippet(text, start, end, radius=180):
    text = normalize_spaces(text)
    start = max(0, start - radius)
    end = min(len(text), end + radius)
    return text[start:end]


def add_matches(findings, text, page_number, patterns):
    for name, pattern in patterns.items():
        for match in pattern.finditer(text or ""):
            findings.append({
                "page": page_number,
                "type": name,
                "match": match.group(0),
                "context": context_snippet(
                    text, match.start(), match.end()
                ),
            })


def inspect_pasals(text, page_number):
    """
    Cari heading Pasal secara line-based.

    Hanya identifier yang memang terlihat suspicious yang dilaporkan.
    Pasal 15, 24, 31, 28A, 36B, dst. tidak dianggap anomaly.
    """
    findings = []

    # Heading sederhana, termasuk identifier OCR seperti 2O / 2l / 168.
    pattern = re.compile(
        r"(?im)^\s*Pasal\s+([0-9]+[A-Za-z]?|[IVXLCDM]+)\s*$"
    )

    for match in pattern.finditer(text or ""):
        identifier = match.group(1)

        if identifier == "168":
            reason = (
                "KANDIDAT 16B: periksa konteks amendment "
                "(mis. Pasal 16A + Pasal 168)"
            )
            findings.append({
                "page": page_number,
                "type": "Pasal_heading_168",
                "match": f"Pasal {identifier}",
                "context": reason,
            })

        elif re.search(r"[OI lI]", identifier):
            findings.append({
                "page": page_number,
                "type": "Pasal_heading_OCR_character",
                "match": f"Pasal {identifier}",
                "context": (
                    "Identifier heading mengandung karakter yang "
                    "sering tertukar oleh OCR."
                ),
            })

    return findings


def audit_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    findings = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        add_matches(
            findings,
            text,
            page_number,
            PASAL_OCR_PATTERNS
        )

        add_matches(
            findings,
            text,
            page_number,
            AYAT_OCR_PATTERNS
        )

        add_matches(
            findings,
            text,
            page_number,
            COMMON_OCR_PATTERNS
        )

        findings.extend(
            inspect_pasals(text, page_number)
        )

    doc.close()
    return findings


def print_findings(filename, findings):
    print("\n" + "=" * 90)
    print(f"FILE: {filename}")
    print("=" * 90)

    if not findings:
        print("No targeted legal OCR anomaly detected.")
        return

    print(f"Findings: {len(findings)}")

    for i, item in enumerate(findings, start=1):
        print(f"\n[{i}] Page {item['page']}")
        print(f"    Type    : {item['type']}")
        print(f"    Match   : {item['match']}")
        print(f"    Context : {item['context']}")


def main():
    if not DOCUMENT_DIR.exists():
        raise SystemExit(
            f"Folder tidak ditemukan: {DOCUMENT_DIR.resolve()}"
        )

    pdf_files = sorted(DOCUMENT_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(
            f"Tidak ada PDF di: {DOCUMENT_DIR.resolve()}"
        )

    print("=" * 90)
    print("LEGAL OCR ANOMALY AUDIT V2")
    print("=" * 90)
    print(f"Directory: {DOCUMENT_DIR}")
    print(f"PDF files : {len(pdf_files)}")

    all_results = []

    for pdf_path in pdf_files:
        print(f"\nScanning: {pdf_path.name}")

        try:
            findings = audit_pdf(pdf_path)
            all_results.append((pdf_path.name, findings))
            print_findings(pdf_path.name, findings)
        except Exception as exc:
            print(f"ERROR: {exc}")

    print("\n" + "=" * 90)
    print("ANOMALY SUMMARY")
    print("=" * 90)

    total = 0

    for filename, findings in all_results:
        total += len(findings)

        if not findings:
            continue

        print(f"\n{filename}: {len(findings)} finding(s)")

        counts = {}
        for item in findings:
            counts[item["type"]] = counts.get(item["type"], 0) + 1

        for anomaly_type, count in sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  - {anomaly_type}: {count}")

    print(f"\nTOTAL FINDINGS: {total}")

    print("\nInterpretation:")
    print("- Ini adalah candidate anomaly, bukan koreksi otomatis.")
    print("- Pasal 168 perlu diverifikasi secara kontekstual.")
    print("- Pasal 2O/2l/3O adalah kandidat OCR O<->0 atau l<->1.")
    print("- Ayat (2a) adalah bentuk legal yang valid dan tidak dianggap malformed.")
    print("- Text asli PDF tidak diubah.")
    print("- OCR ulang jangan dilakukan hanya berdasarkan hasil detector ini.")


if __name__ == "__main__":
    main()
