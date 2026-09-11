# src/rag_chain.py
"""
RAG Chain V2 — legal-aware, single retrieval.

Changes from legacy RAG:
- Uses the production legal vectorstore from embeddings_v2.
- Does not auto-create/re-index the vectorstore at application startup.
- Retrieves documents exactly once per question.
- Returns both the LLM answer and the retrieved source Documents.
- Includes legal metadata in the LLM context and source display.
"""

from typing import List, Tuple, Dict, Any

from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma

from llm_chain import get_llm, get_prompt_template
from retriever import get_retriever
from embeddings import (
    load_legal_vectorstore,
    legal_vectorstore_exists,
    DEFAULT_PERSIST_DIRECTORY,
)


def _format_legal_document(doc: Document, index: int) -> str:
    """Format one retrieved legal chunk with its metadata."""
    metadata = doc.metadata or {}

    source = metadata.get("source", "Tidak diketahui")
    filename = str(source).replace("\\", "/").split("/")[-1]

    document_type = metadata.get("document_type")
    regulation_number = metadata.get("regulation_number")
    year = metadata.get("year")
    title = metadata.get("title")
    bab = metadata.get("bab")
    pasal = metadata.get("pasal")
    ayat = metadata.get("ayat")
    page = metadata.get("page")
    page_end = metadata.get("page_end")

    identity_parts = []
    if document_type:
        identity_parts.append(str(document_type))
    if regulation_number is not None and year is not None:
        identity_parts.append(f"No. {regulation_number}/{year}")
    elif year is not None:
        identity_parts.append(f"Tahun {year}")

    identity = " ".join(
        identity_parts) if identity_parts else "Tidak diketahui"

    if page is not None:
        try:
            page_start = int(page) + 1
        except (TypeError, ValueError):
            page_start = page

        if page_end is not None:
            try:
                page_finish = int(page_end) + 1
            except (TypeError, ValueError):
                page_finish = page_end
            page_display = (
                f"{page_start}-{page_finish}"
                if page_finish != page_start
                else str(page_start)
            )
        else:
            page_display = str(page_start)
    else:
        page_display = "Tidak diketahui"

    structure = []
    if bab is not None:
        structure.append(f"Bab {bab}")
    if pasal is not None:
        structure.append(f"Pasal {pasal}")
    if ayat is not None:
        structure.append(f"Ayat ({ayat})")

    lines = [
        f"[Dokumen {index}]",
        f"Sumber: {filename}",
        f"Regulasi: {identity}",
    ]

    if title:
        lines.append(f"Judul: {title}")

    if structure:
        lines.append("Struktur: " + " | ".join(structure))

    lines.append(f"Halaman: {page_display}")
    lines.append("")
    lines.append("Isi:")
    lines.append(doc.page_content.strip())

    return "\n".join(lines)


def format_retrieved_docs_for_context(source_documents: List[Document]) -> str:
    """Build legal-aware context for the LLM."""
    if not source_documents:
        return "Tidak ada dokumen yang ditemukan."

    return "\n\n---\n\n".join(
        _format_legal_document(doc, index)
        for index, doc in enumerate(source_documents, start=1)
    )


def _format_page_reference(metadata: Dict[str, Any]) -> str:
    """Build a page reference from authoritative document metadata."""
    page = metadata.get("page")
    page_end = metadata.get("page_end")

    if page is None:
        return "Halaman tidak diketahui"

    try:
        start = int(page) + 1
    except (TypeError, ValueError):
        start = page

    if page_end is None:
        return f"Halaman {start}"

    try:
        end = int(page_end) + 1
    except (TypeError, ValueError):
        end = page_end

    return f"Halaman {start}-{end}" if end != start else f"Halaman {start}"


def build_authoritative_citations(source_documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Build citations exclusively from retrieved Document metadata.

    The LLM answer is never used as the source of citation fields such as
    page, Pasal, or ayat.
    """
    citations = []
    seen = set()

    for doc in source_documents:
        metadata = doc.metadata or {}
        source = metadata.get("source", "Tidak diketahui")
        filename = str(source).replace("\\", "/").split("/")[-1]

        citation = {
            "source": filename,
            "document_type": metadata.get("document_type"),
            "regulation_number": metadata.get("regulation_number"),
            "year": metadata.get("year"),
            "title": metadata.get("title"),
            "bab": metadata.get("bab"),
            "pasal": metadata.get("pasal"),
            "ayat": metadata.get("ayat"),
            "page": metadata.get("page"),
            "page_end": metadata.get("page_end"),
            "page_reference": _format_page_reference(metadata),
        }

        key = tuple(citation.items())
        if key in seen:
            continue

        seen.add(key)
        citations.append(citation)

    return citations


def build_rag_chain(vectorstore: Chroma):
    """
    Build a single-retrieval RAG chain.

    The chain returns:
        {
            "answer": str,
            "sources": List[Document],
            "citations": List[Dict[str, Any]]
        }

    `citations` are generated exclusively from source metadata and are
    authoritative for source/page/Pasal/ayat references.


    Retrieval is executed once and the same retrieved documents are
    passed both to the prompt and to the final output.
    """
    llm = get_llm()
    prompt = get_prompt_template()
    retriever = get_retriever(vectorstore)

    # First stage: retrieve once and keep the Documents.
    retrieve = RunnableParallel(
        {
            "documents": retriever,
            "question": RunnablePassthrough(),
        }
    )

    # Second stage: build prompt context from those same Documents.
    answer_chain = (
        RunnableParallel(
            {
                "context": lambda x: format_retrieved_docs_for_context(
                    x["documents"]
                ),
                "question": lambda x: x["question"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both the answer and the exact Documents used as context.
    rag_chain = retrieve | RunnableParallel(
        {
            "answer": answer_chain,
            "sources": lambda x: x["documents"],
            "citations": lambda x: build_authoritative_citations(x["documents"]),
        }
    )

    return rag_chain, retriever


def ask(rag_chain, retriever, question: str) -> Dict[str, Any]:
    """
    Ask the RAG system.

    `retriever` is retained in the function signature for compatibility
    with the existing application interface, but is NOT invoked here.
    The chain already performed the single retrieval.
    """
    result = rag_chain.invoke(question)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "citations": result.get("citations", []),
    }


def setup_vectorstore(documents_folder: str = "data/documents") -> Chroma:
    """
    Load the existing legal vectorstore.

    `documents_folder` is retained for backward compatibility. V2 does not
    auto-ingest or re-index PDFs during application startup.
    """
    del documents_folder

    if not legal_vectorstore_exists(DEFAULT_PERSIST_DIRECTORY):
        raise FileNotFoundError(
            "Legal vectorstore tidak ditemukan. "
            "Jalankan pipeline legal indexing terlebih dahulu."
        )

    print("Legal vectorstore ditemukan, memuat dari disk...")
    return load_legal_vectorstore(DEFAULT_PERSIST_DIRECTORY)


def format_citations_for_display(citations: List[Dict[str, Any]]) -> str:
    """Format authoritative citations generated from Document metadata."""
    if not citations:
        return "Tidak ada sumber dokumen yang ditemukan."

    lines = []
    for citation in citations:
        parts = [str(citation.get("source", "Tidak diketahui"))]

        document_type = citation.get("document_type")
        number = citation.get("regulation_number")
        year = citation.get("year")

        if document_type:
            if number is not None and year is not None:
                parts.append(f"{document_type} No. {number}/{year}")
            elif year is not None:
                parts.append(f"{document_type} Tahun {year}")
            else:
                parts.append(str(document_type))

        pasal = citation.get("pasal")
        ayat = citation.get("ayat")
        if pasal is not None:
            reference = f"Pasal {pasal}"
            if ayat is not None:
                reference += f" ayat ({ayat})"
            parts.append(reference)

        parts.append(
            str(citation.get("page_reference", "Halaman tidak diketahui")))
        lines.append("📄 " + " — ".join(parts))

    return "\n".join(lines)


def format_sources_for_display(source_documents: List[Document]) -> str:
    """Format retrieved legal sources for the UI."""
    if not source_documents:
        return "Tidak ada sumber dokumen yang ditemukan."

    lines = []
    seen = set()

    for doc in source_documents:
        metadata = doc.metadata or {}

        source = metadata.get("source", "Tidak diketahui")
        filename = str(source).replace("\\", "/").split("/")[-1]

        document_type = metadata.get("document_type")
        regulation_number = metadata.get("regulation_number")
        year = metadata.get("year")
        pasal = metadata.get("pasal")
        ayat = metadata.get("ayat")
        page = metadata.get("page")
        page_end = metadata.get("page_end")

        identity_parts = []
        if document_type:
            identity_parts.append(str(document_type))
        if regulation_number is not None and year is not None:
            identity_parts.append(f"No. {regulation_number}/{year}")
        elif year is not None:
            identity_parts.append(f"Tahun {year}")

        identity = " ".join(identity_parts) if identity_parts else None

        if page is not None:
            try:
                page_start = int(page) + 1
            except (TypeError, ValueError):
                page_start = page

            if page_end is not None:
                try:
                    page_finish = int(page_end) + 1
                except (TypeError, ValueError):
                    page_finish = page_end
                page_display = (
                    f"{page_start}-{page_finish}"
                    if page_finish != page_start
                    else str(page_start)
                )
            else:
                page_display = str(page_start)
        else:
            page_display = "?"

        key = (
            filename,
            identity,
            pasal,
            ayat,
            page_display,
        )

        if key in seen:
            continue
        seen.add(key)

        parts = [filename]
        if identity:
            parts.append(identity)
        if pasal is not None:
            pasal_text = f"Pasal {pasal}"
            if ayat is not None:
                pasal_text += f" ayat ({ayat})"
            parts.append(pasal_text)

        parts.append(f"Halaman {page_display}")
        lines.append("📄 " + " — ".join(parts))

    return "\n".join(lines)
