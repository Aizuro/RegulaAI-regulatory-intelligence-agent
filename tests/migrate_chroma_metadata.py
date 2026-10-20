from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
    index_legal_documents_resumable,
)
from src.legal_ingestion import build_legal_documents


def main():
    print("=" * 80)
    print("CHROMA TARGETED METADATA MIGRATION - RESUME")
    print("=" * 80)

    model = get_embedding_model()
    vectorstore = _open_vectorstore(
        model,
        DEFAULT_PERSIST_DIRECTORY,
        DEFAULT_COLLECTION_NAME,
    )

    before = vectorstore._collection.count()
    print(f"Vectors before resume: {before}")

    if before != 772:
        raise RuntimeError(
            f"ABORT: expected 772 vectors after interrupted migration, found {before}."
        )

    print("\nRebuilding legal Documents through current ingestion pipeline...")
    documents, reports = build_legal_documents()

    print(f"Documents from ingestion: {len(documents)}")
    if len(documents) != 776:
        raise RuntimeError(
            f"ABORT: expected 776 legal Documents, found {len(documents)}."
        )

    print("\nResuming existing resumable indexer...")
    print("Only missing deterministic IDs will be embedded.")

    result = index_legal_documents_resumable(
        documents,
        persist_directory=DEFAULT_PERSIST_DIRECTORY,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    final_count = result._collection.count()
    print(f"\nVectors after resume: {final_count}")

    if final_count != 776:
        raise RuntimeError(
            f"MIGRATION INCOMPLETE: expected 776 vectors, found {final_count}."
        )

    print("\nSTATUS: PASS")
    print("Targeted metadata migration resumed successfully.")


if __name__ == "__main__":
    main()
