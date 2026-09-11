"""Resumable legal indexing pipeline V2."""
import argparse
import sys

from legal_ingestion import build_legal_documents
from embeddings import index_legal_documents_resumable

DEFAULT_DOCUMENTS_PATH = "data/documents"
DEFAULT_VECTORSTORE_PATH = "vectorstore"
DEFAULT_COLLECTION_NAME = "legal_regulations"


def validate_documents(documents):
    if not documents:
        raise ValueError("Tidak ada legal Documents yang siap di-index.")

    required = [
        "source", "document_type", "year", "block_type",
        "page", "ingestion_version",
    ]

    for index, doc in enumerate(documents):
        if not doc.page_content.strip():
            raise ValueError(
                f"Document index {index} memiliki page_content kosong.")

        md = doc.metadata or {}
        missing = [key for key in required if key not in md]
        if missing:
            raise ValueError(
                f"Document index {index} missing metadata: {missing}"
            )

        if md.get("document_type") != "UUD" and "regulation_number" not in md:
            raise ValueError(
                f"Document index {index} ({md.get('source')}) "
                "missing metadata: ['regulation_number']"
            )


def run(dry_run=False):
    print("=" * 72)
    print("LEGAL INDEXING PIPELINE V2 — RESUMABLE")
    print("=" * 72)

    documents, reports = build_legal_documents(
        folder_path=DEFAULT_DOCUMENTS_PATH,
        enable_ocr_fallback=True,
        ocr_language="ind",
    )
    validate_documents(documents)

    print(f"PDF ditemukan   : {len(reports)}")
    print(f"Legal Documents : {len(documents)}")
    for r in reports:
        print(
            f"{r['source']:<24} blocks={r['blocks']:>4} "
            f"chunks={r['chunks']:>4} documents={r['documents']:>4}"
        )

    if dry_run:
        print("STATUS: DRY RUN PASS")
        return

    print()
    print("Mode RESUME aktif: vector yang sudah tersimpan tidak di-embed ulang.")

    vs = index_legal_documents_resumable(
        documents=documents,
        persist_directory=DEFAULT_VECTORSTORE_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
        batch_size=50,
        requests_per_minute=100,
        max_retries=5,
    )

    count = vs._collection.count()
    if count != len(documents):
        raise RuntimeError(
            f"Jumlah vector tidak sama dengan jumlah legal Documents: "
            f"{count} != {len(documents)}"
        )
    print(f"STATUS: PASS | {count} vectors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run(dry_run=args.dry_run)
    except Exception as exc:
        print("=" * 72)
        print("STATUS: FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
