from pathlib import Path

from src.document_loader import load_single_pdf
from src.legal_parser import extract_legal_blocks


PDF_PATH = Path("data/documents/2024uu001.pdf")


def main():
    print("=" * 80)
    print("TEST LEGAL PARSER")
    print("=" * 80)

    docs = load_single_pdf(str(PDF_PATH))

    print(f"Pages loaded : {len(docs)}")

    blocks = extract_legal_blocks(docs)

    print(f"Legal blocks : {len(blocks)}")
    print()

    print("CHECK PASAL 16A / 16B / 18A")
    print("-" * 80)

    found = False

    for i, block in enumerate(blocks):
        pasal = block.get("pasal")

        if pasal in {"16A", "16B", "18A"}:
            found = True

            print(f"Block #{i}")
            print(f"  Page    : {block.get('page')}")
            print(f"  Pasal   : {pasal}")
            print(f"  Ayat    : {block.get('ayat')}")
            print(f"  Type    : {block.get('block_type')}")
            print(f"  Content : {block.get('content', '')[:250]!r}")
            print()

    if not found:
        print("TIDAK ADA Pasal 16A / 16B / 18A yang ditemukan.")


if __name__ == "__main__":
    main()
