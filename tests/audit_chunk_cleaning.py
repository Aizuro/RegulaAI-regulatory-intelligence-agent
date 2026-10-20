"""
Audit Cleaning Layer V1

Tujuan:
- Membandingkan content (raw) vs clean_content
- Menampilkan bagian yang dihapus oleh cleaner
- Memeriksa apakah penghapusan benar-benar layout/OCR noise
- Mengambil sampel dari beberapa dokumen

Jalankan:
    python audit_chunk_cleaning.py
"""

from pathlib import Path
from collections import defaultdict

from src.document_loader import load_single_pdf
from src.legal_parser import extract_legal_blocks
from src.legal_chunker import chunk_legal_blocks
from src.legal_chunk_cleaner import clean_legal_chunks


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/documents")

PDF_FILES = [
    "1959uut001.pdf",
    "1959uut003.pdf",
    "1959uut004.pdf",
    "1959uut005.pdf",
    "2024uu001.pdf",
    "2024uu002.pdf",
    "2024uu003.pdf",
    "2024UU005.pdf",
    "2025perpres004.pdf",
    "UUD1945.pdf",
]

# Maksimal contoh per dokumen
SAMPLES_PER_DOCUMENT = 3


# ============================================================
# HELPERS
# ============================================================

def get_metadata(chunk):
    """
    Mendukung chunk berupa dict.
    """
    return chunk.get("metadata", {})


def get_content(chunk):
    return chunk.get("content", "")


def get_clean_content(chunk):
    return chunk.get("clean_content", "")


def print_separator(char="=", width=90):
    print(char * width)


def show_diff(raw, cleaned):
    """
    Menampilkan baris yang ada di raw tetapi hilang
    setelah cleaning.
    """

    raw_lines = raw.splitlines()
    clean_lines = cleaned.splitlines()

    clean_set = set(
        line.strip()
        for line in clean_lines
        if line.strip()
    )

    removed = []

    for line in raw_lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped not in clean_set:
            removed.append(line)

    if not removed:
        print("  Tidak ada baris yang terdeteksi dihapus.")
        return

    print("  BARIS YANG DIHAPUS:")

    for line in removed:
        print(f"    [-] {line}")


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(pdf_path):
    print()
    print_separator()
    print(f"FILE: {pdf_path.name}")
    print_separator()

    # --------------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------------

    documents = load_single_pdf(str(pdf_path))

    # --------------------------------------------------------
    # 2. Split / preprocessing
    # --------------------------------------------------------

    # documents = split_documents(documents)

    # --------------------------------------------------------
    # 3. Legal Parser
    # --------------------------------------------------------

    blocks = extract_legal_blocks(documents)

    # --------------------------------------------------------
    # 4. Legal Chunker
    # --------------------------------------------------------

    chunks = chunk_legal_blocks(blocks)

    # --------------------------------------------------------
    # 5. Cleaning
    # --------------------------------------------------------

    cleaned_chunks = clean_legal_chunks(chunks)

    # --------------------------------------------------------
    # Ambil hanya chunk yang berubah
    # --------------------------------------------------------

    changed = [
        chunk
        for chunk in cleaned_chunks
        if chunk.get("quality", {}).get("changed", False)
    ]

    print(f"Total chunks     : {len(cleaned_chunks)}")
    print(f"Changed chunks   : {len(changed)}")

    if not changed:
        print("Tidak ada chunk yang berubah.")
        return

    # --------------------------------------------------------
    # Group berdasarkan jenis block
    # --------------------------------------------------------

    groups = defaultdict(list)

    for chunk in changed:
        metadata = get_metadata(chunk)

        block_type = metadata.get(
            "block_type",
            "unknown"
        )

        groups[block_type].append(chunk)

    # --------------------------------------------------------
    # Pilih beberapa contoh
    # --------------------------------------------------------

    selected = []

    # Prioritaskan jenis yang berbeda
    for block_type, items in groups.items():
        selected.extend(items[:SAMPLES_PER_DOCUMENT])

    # Jangan terlalu banyak
    selected = selected[:SAMPLES_PER_DOCUMENT * 2]

    # --------------------------------------------------------
    # Print sample
    # --------------------------------------------------------

    for index, chunk in enumerate(selected, start=1):

        metadata = get_metadata(chunk)

        raw = get_content(chunk)
        cleaned = get_clean_content(chunk)

        print()
        print_separator("-")
        print(f"SAMPLE #{index}")
        print_separator("-")

        print(f"Block type : {metadata.get('block_type')}")
        print(f"Pasal      : {metadata.get('pasal')}")
        print(f"Ayat       : {metadata.get('ayat')}")
        print(f"Bab        : {metadata.get('bab')}")
        print(f"Page       : {metadata.get('page')}")
        print(f"Page end   : {metadata.get('page_end')}")

        quality = chunk.get("quality", {})

        print(f"Raw chars  : {quality.get('raw_chars')}")
        print(f"Clean chars: {quality.get('clean_chars')}")
        print(f"Removed    : {quality.get('removed_chars')}")

        print()
        print("RAW CONTENT")
        print("-" * 40)
        print(raw)

        print()
        print("CLEAN CONTENT")
        print("-" * 40)
        print(cleaned)

        print()
        print("REMOVED / DIFFERENCE")
        print("-" * 40)

        show_diff(raw, cleaned)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print_separator()
    print("LEGAL CHUNK CLEANING AUDIT V1")
    print_separator()

    missing = []

    for filename in PDF_FILES:

        pdf_path = DATA_DIR / filename

        if not pdf_path.exists():
            missing.append(str(pdf_path))
            continue

        process_pdf(pdf_path)

    # --------------------------------------------------------
    # Missing files
    # --------------------------------------------------------

    if missing:

        print()
        print_separator()
        print("WARNING: FILE TIDAK DITEMUKAN")
        print_separator()

        for path in missing:
            print(f"- {path}")

    print()
    print_separator()
    print("AUDIT SELESAI")
    print_separator()


if __name__ == "__main__":
    main()
