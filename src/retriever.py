# Tahap 5: Cari Dokumen yang Relevan

# src/retriever.py

from langchain_chroma import Chroma
from langchain.schema import BaseRetriever, Document
from typing import List


def get_retriever(
    vectorstore: Chroma,
    k: int = 4,
    search_type: str = "mmr"
) -> BaseRetriever:
    """
    Membuat retriever dari vectorstore yang sudah ada.

    Args:
        vectorstore : ChromaDB vectorstore object
        k           : Jumlah chunks yang diambil per query
        search_type : "similarity" atau "mmr"
                      - similarity : ambil k chunk paling mirip
                      - mmr        : ambil chunk yang relevan DAN beragam

    Returns:
        Retriever object siap pakai
    """
    if search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,           # Jumlah chunk yang dikembalikan ke LLM
                "fetch_k": k * 3, # Jumlah kandidat yang diperiksa dulu
                "lambda_mult": 0.7 # 0 = maksimal beragam, 1 = maksimal relevan
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    return retriever


def retrieve_documents(
    retriever: BaseRetriever,
    query: str
) -> List[Document]:
    """
    Mengambil dokumen relevan berdasarkan query.

    Args:
        retriever : Retriever object dari get_retriever()
        query     : Pertanyaan dari user

    Returns:
        List of Document yang relevan
    """
    documents = retriever.invoke(query)
    return documents


def format_retrieved_docs(documents: List[Document]) -> str:
    """
    Menggabungkan semua chunk menjadi satu string konteks
    untuk dikirim ke LLM.

    Args:
        documents : Hasil dari retrieve_documents()

    Returns:
        String konteks yang sudah diformat
    """
    context_parts = []

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 0)

        # Format setiap chunk dengan label sumbernya
        context_parts.append(
            f"[Dokumen {i} - Halaman {page + 1}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def inspect_retrieval(
    retriever: BaseRetriever,
    query: str
) -> None:
    """
    Menampilkan hasil retrieval secara detail untuk debugging.

    Args:
        retriever : Retriever object
        query     : Query yang ingin dites
    """
    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"{'='*50}")

    docs = retrieve_documents(retriever, query)

    print(f"Ditemukan {len(docs)} dokumen relevan:\n")

    for i, doc in enumerate(docs, 1):
        print(f"--- Dokumen {i} ---")
        print(f"Sumber  : {doc.metadata.get('source', 'N/A')}")
        print(f"Halaman : {doc.metadata.get('page', 0) + 1}")
        print(f"Isi     : {doc.page_content[:300]}...")
        print()

    print("KONTEKS YANG AKAN DIKIRIM KE LLM:")
    print("-" * 50)
    print(format_retrieved_docs(docs))