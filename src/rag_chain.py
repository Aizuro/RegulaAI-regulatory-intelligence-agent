# src/rag_chain.py

from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from typing import List

from src.llm_chain import get_llm, get_prompt_template
from src.retriever import get_retriever, format_retrieved_docs
from src.embeddings import load_vectorstore, create_vectorstore, vectorstore_exists
from src.document_loader import load_multiple_pdfs
from src.text_splitter import split_documents


def build_rag_chain(vectorstore: Chroma):
    """
    Merakit semua komponen menjadi satu RAG chain.

    Args:
        vectorstore: ChromaDB vectorstore yang sudah berisi dokumen

    Returns:
        Tuple: (rag_chain, retriever)
        - rag_chain  : chain siap pakai untuk menjawab pertanyaan
        - retriever  : disimpan terpisah agar bisa dipakai ambil sumber dokumen
    """
    llm = get_llm()
    prompt = get_prompt_template()
    retriever = get_retriever(vectorstore)

    # Cara chain ini bekerja saat dipanggil dengan {"question": "..."}:
    #
    # RunnableParallel menjalankan DUA cabang SECARA BERSAMAAN:
    #   Cabang 1 - "context" : pertanyaan → retriever → format_retrieved_docs
    #   Cabang 2 - "question": pertanyaan diteruskan apa adanya (passthrough)
    #
    # Hasilnya digabung: {"context": "...", "question": "..."}
    # Lalu dikirim ke prompt → llm → output parser

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_retrieved_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask(rag_chain, retriever, question: str) -> dict:
    """
    Fungsi utama untuk mengajukan pertanyaan ke RAG chain.
    Mengembalikan jawaban DAN dokumen sumber yang digunakan.

    Args:
        rag_chain : hasil dari build_rag_chain()
        retriever : hasil dari build_rag_chain()
        question  : pertanyaan dari user (string)

    Returns:
        dict berisi:
          - "answer"   : jawaban dari LLM (string)
          - "sources"  : list of Document yang dijadikan konteks
    """
    # Jalankan chain → dapat jawaban teks
    answer = rag_chain.invoke(question)

    # Jalankan retriever lagi → dapat dokumen sumbernya
    # (chain tidak mengembalikan dokumen sumber secara langsung)
    source_documents = retriever.invoke(question)

    return {
        "answer": answer,
        "sources": source_documents
    }


def setup_vectorstore(documents_folder: str = "data/documents") -> Chroma:
    """
    Setup vectorstore: buat baru jika belum ada, load jika sudah ada.
    Fungsi ini yang dipanggil pertama kali saat aplikasi start.

    Args:
        documents_folder: folder berisi PDF yang akan diindeks

    Returns:
        Chroma vectorstore siap pakai
    """
    if vectorstore_exists():
        print("Vectorstore ditemukan, memuat dari disk...")
        return load_vectorstore()

    print("Vectorstore belum ada, membuat dari awal...")
    documents = load_multiple_pdfs(documents_folder)

    if not documents:
        raise ValueError(
            f"Tidak ada dokumen PDF di folder '{documents_folder}'. "
            "Tambahkan file PDF terlebih dahulu."
        )

    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)

    return vectorstore


def format_sources_for_display(source_documents: List[Document]) -> str:
    """
    Memformat dokumen sumber menjadi teks ringkas untuk ditampilkan ke user.
    Dipakai di UI Streamlit nanti (Tahap 8).

    Args:
        source_documents: list of Document dari hasil retriever

    Returns:
        String ringkasan sumber yang mudah dibaca
    """
    if not source_documents:
        return "Tidak ada sumber dokumen yang ditemukan."

    lines = []
    seen = set()  # Untuk menghindari sumber duplikat

    for doc in source_documents:
        source = doc.metadata.get("source", "Tidak diketahui")
        page = doc.metadata.get("page", 0) + 1  # +1 karena index mulai dari 0

        # Ambil nama file saja, bukan full path
        filename = source.split("/")[-1]
        key = f"{filename}_hal{page}"

        if key not in seen:
            seen.add(key)
            lines.append(f"📄 {filename} — Halaman {page}")

    return "\n".join(lines)
