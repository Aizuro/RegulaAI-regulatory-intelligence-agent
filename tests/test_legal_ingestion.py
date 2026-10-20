"""Dependency-light regression test for the legal ingestion pipeline.

Run from the project root after placing these modules under src/.
Requires the project's LangChain dependencies.
"""

from pathlib import Path

from src.legal_ingestion import build_legal_documents


PDFS = [
    "1959uut001.pdf",
    "1959uut003.pdf",
    "1959uut004.pdf",
    "1959uut005.pdf",
    "2024UU005.pdf",
    "2024uu001.pdf",
    "2024uu002.pdf",
    "2024uu003.pdf",
    "2025perpres004.pdf",
    "UUD1945.pdf",
]


def main():
    root = Path("data/documents")
    assert root.exists(), f"Folder tidak ditemukan: {root}"

    docs, reports = build_legal_documents(
        str(root),
        enable_ocr_fallback=False,
    )

    report_by_source = {r["source"]: r for r in reports}

    print("=" * 80)
    print("LEGAL INGESTION PIPELINE V1 REGRESSION")
    print("=" * 80)

    for filename in PDFS:
        assert filename in report_by_source, f"Missing report: {filename}"
        r = report_by_source[filename]
        print(
            f"{filename:20} "
            f"blocks={r['blocks']:3} "
            f"chunks={r['chunks']:3} "
            f"documents={r['documents']:3}"
        )
        assert r["documents"] > 0

    assert docs, "Tidak ada Document hasil ingestion."

    # Core metadata checks.
    for doc in docs:
        md = doc.metadata
        assert doc.page_content.strip(), "Ada embedding document dengan content kosong."
        assert md.get("source"), "Metadata source hilang."
        assert md.get("block_type"), "Metadata block_type hilang."
        assert md.get("cleaning_version") == "v1.1"
        assert md.get("ingestion_version") == "legal-ingestion-v1"
        assert isinstance(md.get("page"), int)
        assert isinstance(md.get("page_end"), int)
        assert isinstance(md.get("is_cross_page"), bool)
        assert isinstance(md.get("source_pages"), str)

    target = [
        d for d in docs
        if d.metadata.get("source") == "2024uu001.pdf"
        and d.metadata.get("pasal") == "17"
        and d.metadata.get("ayat") == "2a"
    ]

    print("\nTARGET 2024uu001 Pasal 17 Ayat 2a")
    print("count:", len(target))
    assert len(target) == 1
    t = target[0]
    print("source_pages:", t.metadata.get("source_pages"))
    print("is_cross_page:", t.metadata.get("is_cross_page"))
    print("chars:", len(t.page_content))

    assert t.metadata.get("source_pages") == "6,7"
    assert t.metadata.get("is_cross_page") is True
    assert "Transaksi Elektronik yang memiliki risiko tinggi" in t.page_content

    print("\nTOTAL FINAL DOCUMENTS:", len(docs))
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
