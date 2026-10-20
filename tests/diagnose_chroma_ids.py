from src.embeddings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    get_embedding_model,
    _open_vectorstore,
    _document_ids,
)
from src.legal_ingestion import build_legal_documents


def main():
    print("=" * 80)
    print("CHROMA ID / METADATA DIAGNOSTIC")
    print("=" * 80)

    model = get_embedding_model()
    vs = _open_vectorstore(
        model, DEFAULT_PERSIST_DIRECTORY, DEFAULT_COLLECTION_NAME)
    collection = vs._collection

    existing = collection.get(include=["documents", "metadatas"])
    existing_ids = set(existing.get("ids") or [])
    print(f"Chroma vectors : {len(existing_ids)}")

    print("\nBuilding current legal Documents...")
    documents, reports = build_legal_documents()
    expected_ids = _document_ids(documents)
    expected_set = set(expected_ids)
    print(f"Expected documents : {len(documents)}")
    print(f"Expected IDs       : {len(expected_set)}")

    matched = expected_set & existing_ids
    missing = expected_set - existing_ids
    extra = existing_ids - expected_set

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Matched IDs : {len(matched)}")
    print(f"Missing IDs : {len(missing)}")
    print(f"Extra IDs   : {len(extra)}")

    if missing:
        print("\nMISSING IDs:")
        for doc, doc_id in zip(documents, expected_ids):
            if doc_id in missing:
                md = doc.metadata
                print(
                    f"  {doc_id} | {md.get('source')} | "
                    f"type={md.get('document_type')} | "
                    f"reg={md.get('regulation_number')} | year={md.get('year')} | "
                    f"pasal={md.get('pasal')} | ayat={md.get('ayat')} | "
                    f"page={md.get('page')}-{md.get('page_end')}"
                )

    if extra:
        print("\nEXTRA IDs IN CHROMA:")
        records = collection.get(ids=list(extra), include=[
                                 "documents", "metadatas"])
        ids = records.get("ids") or []
        metadatas = records.get("metadatas") or []
        for i, extra_id in enumerate(ids):
            md = metadatas[i] if i < len(metadatas) else {}
            print(
                f"  {extra_id} | {md.get('source')} | "
                f"type={md.get('document_type')} | "
                f"reg={md.get('regulation_number')} | year={md.get('year')} | "
                f"pasal={md.get('pasal')} | ayat={md.get('ayat')} | "
                f"page={md.get('page')}-{md.get('page_end')}"
            )

    print("\nSTATUS: DIAGNOSTIC ONLY")
    print("No vectors were deleted or modified.")


if __name__ == "__main__":
    main()
