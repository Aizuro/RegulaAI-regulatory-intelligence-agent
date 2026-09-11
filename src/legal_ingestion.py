"""Legal ingestion pipeline V1.

Pipeline:
    PDF -> document_loader -> legal_parser -> legal_chunker -> cleaner
         -> LangChain Documents with legal metadata

This module deliberately keeps embedding/Chroma operations in embeddings.py.
Its job is to produce clean, metadata-rich Documents ready for vectorization.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from langchain.schema import Document

from document_loader import load_single_pdf
from legal_parser import enrich_documents
from legal_chunker import chunk_legal_blocks
from legal_chunk_cleaner import clean_legal_chunks


PIPELINE_VERSION = "legal-ingestion-v1"


def _safe_metadata_value(value: Any) -> Any:
    """Convert metadata into Chroma-safe primitive values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _build_metadata(
    chunk: Dict[str, Any],
    identity: Dict[str, Any],
    source_filename: str,
    source_doc_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a flat, Chroma-compatible metadata dictionary."""
    source_doc_metadata = source_doc_metadata or {}
    quality = chunk.get("quality") or {}

    values = {
        # Document identity
        "source": source_filename,
        "document_type": identity.get("document_type"),
        "regulation_number": identity.get("regulation_number"),
        "year": identity.get("year"),
        "title": identity.get("title"),

        # Legal structure
        "bab": chunk.get("bab"),
        "pasal": chunk.get("pasal"),
        "ayat": chunk.get("ayat"),
        "amendment_article": chunk.get("amendment_article"),
        "target_pasal": chunk.get("target_pasal"),
        "block_type": chunk.get("block_type"),

        # Provenance
        "page": chunk.get("page"),
        "page_end": chunk.get("page_end", chunk.get("page")),
        "source_pages": chunk.get("source_pages", [chunk.get("page")]),
        "is_cross_page": bool(chunk.get("is_cross_page", False)),
        "merged_blocks": chunk.get("merged_blocks", 1),

        # Cleaning / extraction diagnostics
        "cleaning_version": quality.get("cleaning_version", "v1.1"),
        "extraction_method": source_doc_metadata.get("extraction_method"),
        "extraction_quality": source_doc_metadata.get("extraction_quality"),
        "ocr_used": source_doc_metadata.get("ocr_used", False),
    }

    # Chroma metadata cannot contain None. Keep only populated fields.
    metadata: Dict[str, Any] = {}
    for key, value in values.items():
        safe = _safe_metadata_value(value)
        if safe is not None and safe != "":
            metadata[key] = safe

    return metadata


def _page_metadata_map(documents: List[Any]) -> Dict[int, Dict[str, Any]]:
    """Map 1-based parser page numbers to loader metadata."""
    result: Dict[int, Dict[str, Any]] = {}
    for index, doc in enumerate(documents, start=1):
        result[index] = dict(getattr(doc, "metadata", {}) or {})
    return result


def build_legal_documents(
    folder_path: str = "data/documents",
    *,
    enable_ocr_fallback: bool = True,
    ocr_language: str = "ind",
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """Run the complete legal preprocessing pipeline.

    Returns:
        documents: clean LangChain Documents ready for embedding.
        reports: one diagnostic report per source PDF.

    The original parser/chunker `content` is not embedded. Only
    `clean_content` becomes Document.page_content.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"Tidak ada PDF di folder: {folder_path}")

    final_documents: List[Document] = []
    reports: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        loaded = load_single_pdf(
            str(pdf_path),
            enable_ocr_fallback=enable_ocr_fallback,
            ocr_language=ocr_language,
        )

        blocks, identity = enrich_documents(loaded)
        chunks = chunk_legal_blocks(blocks)
        cleaned_chunks = clean_legal_chunks(chunks)
        page_metadata = _page_metadata_map(loaded)

        source_documents = 0
        skipped_empty = 0

        for chunk in cleaned_chunks:
            clean_content = (chunk.get("clean_content") or "").strip()
            if not clean_content:
                skipped_empty += 1
                continue

            metadata = _build_metadata(
                chunk,
                identity,
                pdf_path.name,
                page_metadata.get(int(chunk.get("page")), {})
                if chunk.get("page") is not None else {},
            )

            # Stable local ID is useful for deterministic ingestion later.
            metadata["ingestion_version"] = PIPELINE_VERSION

            final_documents.append(
                Document(
                    page_content=clean_content,
                    metadata=metadata,
                )
            )
            source_documents += 1

        reports.append({
            "source": pdf_path.name,
            "identity": identity,
            "blocks": len(blocks),
            "chunks": len(chunks),
            "documents": source_documents,
            "skipped_empty": skipped_empty,
        })

    return final_documents, reports
