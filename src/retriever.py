from typing import List, Optional, Dict, Any
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def _extract_legal_query(query: str) -> Dict[str, Optional[str]]:
    """
    Extract explicit legal identifiers from a user query.

    Examples:
      UU Nomor 1 Tahun 2024, Pasal 17 ayat (2a)
      Peraturan Presiden Nomor 4 Tahun 2025
      UUD 1945
    """
    q = query.lower()

    result: Dict[str, Optional[str]] = {
        "document_type": None,
        "regulation_number": None,
        "year": None,
        "pasal": None,
        "ayat": None,
    }

    # Document type
    if re.search(r"\bundang[- ]undang\s+darurat\b|\buu\s+darurat\b", q):
        result["document_type"] = "UU Darurat"
    elif re.search(r"\bundang[- ]undang\b|\buu\b", q):
        result["document_type"] = "UU"
    elif re.search(r"\bperaturan\s+presiden\b|\bperpres\b", q):
        result["document_type"] = "Perpres"
    elif re.search(r"\buud\s*1945\b|\bundang[- ]undang\s+dasar\s+1945\b", q):
        result["document_type"] = "UUD"

    # Regulation number + year
    number_match = re.search(
        r"(?:nomor|no\.?)\s*(\d+)\s*(?:tahun\s*(\d{4}))?",
        q,
    )
    if number_match:
        result["regulation_number"] = number_match.group(1)
        if number_match.group(2):
            result["year"] = number_match.group(2)

    # For UUD 1945, there is no regulation_number in metadata.
    if result["document_type"] == "UUD" and re.search(r"1945", q):
        result["year"] = "1945"
        result["regulation_number"] = None

    # Pasal: allow 13A, 16B, 28D, etc.
    pasal_match = re.search(r"\bpasal\s+(\d+[a-z]?)\b", q)
    if pasal_match:
        result["pasal"] = pasal_match.group(1).upper()

    # Ayat: allow (2a), (2), 2a
    ayat_match = re.search(
        r"\bayat\s*\(\s*(\d+[a-z]?)\s*\)|\bayat\s+(\d+[a-z]?)\b",
        q,
    )
    if ayat_match:
        result["ayat"] = (ayat_match.group(1) or ayat_match.group(2)).lower()

    return result


def _metadata_filter_from_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma filter only from identifiers explicitly present in the query.
    Returns None when there is not enough legal metadata to safely filter.
    """
    parsed = _extract_legal_query(query)
    conditions = []

    if parsed["document_type"] == "UU Darurat":
        # Corpus metadata uses document_type "UU"; distinguish the special
        # document only when additional identity metadata is available.
        conditions.append({"document_type": {"$eq": "UU"}})
    elif parsed["document_type"]:
        conditions.append({"document_type": {"$eq": parsed["document_type"]}})

    if parsed["regulation_number"] is not None:
        conditions.append(
            {"regulation_number": {"$eq": parsed["regulation_number"]}}
        )

    if parsed["year"] is not None:
        conditions.append({"year": {"$eq": parsed["year"]}})

    if parsed["pasal"] is not None:
        conditions.append({"pasal": {"$eq": parsed["pasal"]}})

    if parsed["ayat"] is not None:
        conditions.append({"ayat": {"$eq": parsed["ayat"]}})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def get_retriever(
    vectorstore: Chroma,
    k: int = 4,
    search_type: str = "mmr",
) -> BaseRetriever:
    """
    Production retriever.

    Legal-aware behavior:
    - If the query contains explicit legal identifiers, first attempt an exact
      metadata-filtered retrieval.
    - If the exact filter is too restrictive / returns no result, fall back to
      the existing semantic retrieval behavior.
    - Generic questions continue to use the existing MMR/similarity strategy.
    """

    if search_type not in {"mmr", "similarity"}:
        raise ValueError("search_type harus 'mmr' atau 'similarity'.")

    class LegalAwareRetriever(BaseRetriever):
        vectorstore: Any
        k: int = 4
        search_type: str = "mmr"

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            metadata_filter = _metadata_filter_from_query(query)

            # Exact legal lookup is preferred when the query explicitly names
            # enough metadata. This is especially important for Pasal/Ayat
            # citation queries where semantic similarity can rank an amendment
            # instruction above the actual target provision.
            if metadata_filter is not None:
                exact_docs = self.vectorstore.similarity_search(
                    query,
                    k=self.k,
                    filter=metadata_filter,
                )
                if exact_docs:
                    return exact_docs

            if self.search_type == "mmr":
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": self.k,
                        "fetch_k": self.k * 3,
                        "lambda_mult": 0.7,
                    },
                )
            else:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.k},
                )

            return retriever.invoke(query)

    return LegalAwareRetriever(
        vectorstore=vectorstore,
        k=k,
        search_type=search_type,
    )


def retrieve_documents(
    retriever: BaseRetriever,
    query: str,
) -> List[Document]:
    """Retrieve documents for a query."""
    return retriever.invoke(query)


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    Format retrieved legal documents as LLM context.
    """
    if not docs:
        return "Tidak ada dokumen yang ditemukan."

    formatted = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        source = metadata.get("source", "unknown")
        document_type = metadata.get("document_type", "")
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
            identity_parts.append(document_type)
        if regulation_number is not None:
            identity_parts.append(f"Nomor {regulation_number}")
        if year:
            identity_parts.append(f"Tahun {year}")

        location_parts = []
        if bab:
            location_parts.append(f"Bab {bab}")
        if pasal:
            location_parts.append(f"Pasal {pasal}")
        if ayat:
            location_parts.append(f"ayat ({ayat})")

        page_label = str(page) if page_end in (
            None, page) else f"{page}-{page_end}"

        header = f"[Dokumen {i}]"
        if identity_parts:
            header += " " + " ".join(identity_parts)
        if title:
            header += f" | {title}"
        if location_parts:
            header += " | " + " ".join(location_parts)
        header += f" | Halaman {page_label} | Sumber {source}"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(formatted)


def inspect_retrieval(
    retriever: BaseRetriever,
    query: str,
    k: Optional[int] = None,
) -> List[Document]:
    """Inspect retrieval results for debugging/validation."""
    docs = retrieve_documents(retriever, query)

    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        print(f"\nRank {i}")
        print(f"Source       : {metadata.get('source')}")
        print(f"Type         : {metadata.get('document_type')}")
        print(f"Regulation   : {metadata.get('regulation_number')}")
        print(f"Year         : {metadata.get('year')}")
        print(f"Pasal        : {metadata.get('pasal')}")
        print(f"Ayat         : {metadata.get('ayat')}")
        print(
            f"Page         : {metadata.get('page')}-{metadata.get('page_end')}")
        print(f"Content      : {doc.page_content[:300]}")

    return docs
