from pathlib import Path

from src.document_loader import load_single_pdf
from src.legal_parser import extract_legal_blocks
from src.legal_chunker import chunk_legal_blocks
from src.legal_chunk_cleaner import clean_legal_chunks

PDF_DIR = Path("data/documents")


def main():
    print("=" * 80)
    print("LEGAL CHUNK QUALITY/CLEANING V1 TEST")
    print("=" * 80)

    total = 0
    changed = 0

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        docs = load_single_pdf(str(pdf))
        blocks = extract_legal_blocks(docs)
        chunks = chunk_legal_blocks(blocks)
        cleaned = clean_legal_chunks(chunks)
        total += len(cleaned)
        changed += sum(1 for x in cleaned if x["quality"]["changed"])
        print(
            f"{pdf.name:22} blocks={len(blocks):3} chunks={len(chunks):3} cleaned={sum(1 for x in cleaned if x['quality']['changed']):3}")

    # Target the known cross-page chunk.
    docs = load_single_pdf(str(PDF_DIR / "2024uu001.pdf"))
    chunks = clean_legal_chunks(chunk_legal_blocks(extract_legal_blocks(docs)))
    targets = [
        x for x in chunks
        if x.get("pasal") == "17" and x.get("ayat") == "2a"
    ]
    print()
    print("TARGET 2024uu001 Pasal 17 Ayat 2a")
    print(f"count: {len(targets)}")
    if targets:
        t = targets[0]
        print(f"pages: {t.get('source_pages')}")
        print(f"changed: {t['quality']['changed']}")
        print(f"raw_chars: {t['quality']['raw_chars']}")
        print(f"clean_chars: {t['quality']['clean_chars']}")
        print("clean preview:")
        print(t["clean_content"][:600])

    print()
    print(f"TOTAL CHUNKS: {total}")
    print(f"CHUNKS CLEANED: {changed}")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
