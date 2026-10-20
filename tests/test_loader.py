from src.document_loader import load_multiple_pdfs

documents = load_multiple_pdfs(
    "data/documents",
    enable_ocr_fallback=True
)

for doc in documents:
    print(
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("extraction_method"),
        doc.metadata.get("ocr_used"),
        doc.metadata.get("extraction_quality")
    )
