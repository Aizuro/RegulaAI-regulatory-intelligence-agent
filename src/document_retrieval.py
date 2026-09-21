"""V8.1 Document Retrieval Layer.

Retrieves regulatory documents from the existing Chroma corpus and, when the
requested regulation is not available locally, fetches the PDF from its source
URL and runs the existing legal ingestion pipeline.

Design:
- Reuse V1-V6 ingestion, embeddings and Chroma components.
- Do not modify the V7 monitoring state.
- Do not automatically index freshly fetched documents into the main vectorstore.
  V8.1 retrieval is read-oriented so V8.2 can compare current vs previous versions
  without silently mixing versions in the existing corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.documents import Document

from embeddings import load_legal_vectorstore
from legal_ingestion import build_legal_documents


@dataclass
class RegulationIdentity:
    document_type: str
    regulation_number: str
    year: str

    @property
    def regulation_key(self) -> str:
        return f"{self.document_type.upper()}-{self.regulation_number}-{self.year}"


@dataclass
class DocumentRetrievalResult:
    status: str
    source: str
    regulation_key: str
    documents: List[Document]
    source_url: Optional[str] = None
    retrieval_method: str = ""
    error: Optional[str] = None


def _validate_https_url(url: str) -> None:
    parsed = urlparse(str(url))
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("source URL harus menggunakan HTTPS yang valid.")


def _identity_from_regulation(regulation: Dict[str, Any]) -> RegulationIdentity:
    document_type = str(regulation.get("type") or "").strip()
    number = str(regulation.get("number") or "").strip()
    year = str(regulation.get("year") or "").strip()

    if not document_type or not number or not year:
        raise ValueError(
            "Regulation membutuhkan type, number, dan year untuk retrieval."
        )

    return RegulationIdentity(document_type, number, year)


def _load_existing_chunks(
    identity: RegulationIdentity,
    persist_directory: str = "vectorstore",
    collection_name: str = "legal_regulations",
) -> List[Document]:
    """Load all locally indexed chunks for an explicit regulation identity.

    This uses metadata filtering rather than semantic search because V8.1 needs
    the complete document representation for later document comparison.
    """
    vectorstore = load_legal_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    collection = vectorstore._collection
    where = {
        "$and": [
            {"document_type": {"$eq": identity.document_type}},
            {"regulation_number": {"$eq": identity.regulation_number}},
            {"year": {"$eq": identity.year}},
        ]
    }

    result = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    documents = []
    ids = result.get("ids") or []
    texts = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    for index, text in enumerate(texts):
        metadata = dict(metadatas[index] or {})
        metadata["retrieval_id"] = ids[index] if index < len(ids) else None
        metadata["retrieval_method"] = "local_chroma"
        documents.append(Document(page_content=text or "", metadata=metadata))

    # Chroma does not guarantee semantic order for collection.get(). Restore a
    # deterministic legal order for downstream comparison.
    documents.sort(
        key=lambda doc: (
            int(doc.metadata.get("page") or 0),
            int(doc.metadata.get("page_end") or doc.metadata.get("page") or 0),
            str(doc.metadata.get("bab") or ""),
            str(doc.metadata.get("pasal") or ""),
            str(doc.metadata.get("ayat") or ""),
        )
    )
    return documents


def _download_pdf(url: str, destination: Path, timeout: int = 30) -> None:
    _validate_https_url(url)

    request = Request(
        url,
        headers={
            "User-Agent": "Regula-V8.1-Document-Retrieval/1.0",
            "Accept": "application/pdf,*/*",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        data = response.read()

    # Do not trust Content-Type alone: JDIH endpoints can return redirects or
    # generic content types. The PDF signature is a stronger local check.
    if not data.startswith(b"%PDF"):
        raise ValueError(
            f"URL tidak mengembalikan PDF yang valid (Content-Type={content_type!r})."
        )

    destination.write_bytes(data)


def _fetch_and_ingest(
    regulation: Dict[str, Any],
    source_url: str,
) -> List[Document]:
    """Fetch one source PDF and reuse the existing ingestion pipeline."""
    identity = _identity_from_regulation(regulation)

    with TemporaryDirectory(prefix="regula_v81_") as tmp:
        pdf_path = Path(tmp) / f"{identity.regulation_key}.pdf"
        _download_pdf(source_url, pdf_path)

        documents, reports = build_legal_documents(
            folder_path=tmp,
            enable_ocr_fallback=True,
            ocr_language="ind",
        )

        if not documents:
            raise ValueError(
                f"PDF berhasil diunduh tetapi tidak menghasilkan legal documents "
                f"untuk {identity.regulation_key}."
            )

        # Preserve source URL and monitoring identity for V8.2/V8.3.
        for doc in documents:
            doc.metadata["regulation_key"] = identity.regulation_key
            doc.metadata["source_url"] = source_url
            doc.metadata["retrieval_method"] = "external_pdf"
            doc.metadata["retrieval_source"] = "external"

        return documents


def retrieve_regulation(
    regulation: Dict[str, Any],
    *,
    persist_directory: str = "vectorstore",
    collection_name: str = "legal_regulations",
    fetch_if_missing: bool = True,
) -> DocumentRetrievalResult:
    """Retrieve a regulation from local Chroma or its source URL.

    Priority:
    1. Complete local Chroma representation.
    2. External HTTPS PDF from regulation["url"], if enabled.

    The function never writes externally fetched documents into the existing
    Chroma collection.
    """
    identity = _identity_from_regulation(regulation)

    try:
        local_documents = _load_existing_chunks(
            identity,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except FileNotFoundError:
        local_documents = []

    if local_documents:
        return DocumentRetrievalResult(
            status="ok",
            source="local",
            regulation_key=identity.regulation_key,
            documents=local_documents,
            source_url=regulation.get("url"),
            retrieval_method="local_chroma_metadata",
        )

    if not fetch_if_missing:
        return DocumentRetrievalResult(
            status="not_found",
            source="none",
            regulation_key=identity.regulation_key,
            documents=[],
            source_url=regulation.get("url"),
            retrieval_method="local_chroma_metadata",
            error="Regulasi tidak ditemukan di local corpus.",
        )

    source_url = str(regulation.get("url") or "").strip()
    if not source_url:
        return DocumentRetrievalResult(
            status="not_found",
            source="none",
            regulation_key=identity.regulation_key,
            documents=[],
            retrieval_method="none",
            error="Regulasi tidak ada di local corpus dan source URL tidak tersedia.",
        )

    try:
        external_documents = _fetch_and_ingest(regulation, source_url)
        return DocumentRetrievalResult(
            status="ok",
            source="external",
            regulation_key=identity.regulation_key,
            documents=external_documents,
            source_url=source_url,
            retrieval_method="external_pdf_ingestion",
        )
    except Exception as exc:
        return DocumentRetrievalResult(
            status="error",
            source="external",
            regulation_key=identity.regulation_key,
            documents=[],
            source_url=source_url,
            retrieval_method="external_pdf_ingestion",
            error=f"{type(exc).__name__}: {exc}",
        )


def retrieve_relevant_chunks(
    retriever,
    query: str,
) -> List[Document]:
    """Small V8.1 adapter for existing semantic/legal-aware retrieval."""
    return retriever.invoke(query)
