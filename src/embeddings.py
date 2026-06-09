# src/embeddings.py

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List

load_dotenv()


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Membuat instance embedding model dari Google (Gemini).
    Model ini yang mengubah teks menjadi vector angka.

    Returns:
        GoogleGenerativeAIEmbeddings object
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY tidak ditemukan! "
            "Pastikan file .env sudah diisi dengan benar."
        )

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    return embedding_model


def create_vectorstore(
    chunks: List[Document],
    persist_directory: str = "vectorstore"
) -> Chroma:
    """
    Mengubah chunks menjadi vectors dan menyimpannya ke ChromaDB.
    Fungsi ini HANYA dipanggil SEKALI saat pertama kali setup.

    Args:
        chunks           : Hasil dari split_documents()
        persist_directory: Folder tempat ChromaDB menyimpan datanya

    Returns:
        Chroma vectorstore object
    """
    print("Memuat embedding model...")
    embedding_model = get_embedding_model()

    print(f"Membuat vectorstore dari {len(chunks)} chunks...")
    print("Proses ini membutuhkan waktu karena setiap chunk dikirim ke API Google...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print(f"Vectorstore berhasil dibuat!")
    print(f"Data tersimpan di folder: {persist_directory}/")
    print(f"Total vectors: {vectorstore._collection.count()}")

    return vectorstore


def load_vectorstore(
    persist_directory: str = "vectorstore"
) -> Chroma:
    """
    Memuat vectorstore yang SUDAH ADA dari disk.

    Args:
        persist_directory: Folder tempat ChromaDB tersimpan

    Returns:
        Chroma vectorstore object
    """
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"Vectorstore tidak ditemukan di '{persist_directory}'. "
            "Jalankan create_vectorstore() terlebih dahulu."
        )

    print(f"Memuat vectorstore dari {persist_directory}/...")

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    total = vectorstore._collection.count()
    print(f"Vectorstore berhasil dimuat! Total vectors: {total}")

    return vectorstore


def add_documents_to_vectorstore(
    vectorstore: Chroma,
    chunks: List[Document]
) -> Chroma:
    """
    Menambahkan dokumen baru ke vectorstore yang SUDAH ADA.
    Berbeda dengan create_vectorstore() yang membuat dari nol,
    fungsi ini hanya menambah tanpa menghapus yang lama.

    Args:
        vectorstore : Chroma vectorstore yang sudah ada
        chunks      : Chunks baru hasil split_documents()

    Returns:
        Chroma vectorstore yang sudah diperbarui
    """
    if not chunks:
        print("Tidak ada chunks untuk ditambahkan.")
        return vectorstore

    print(f"Menambahkan {len(chunks)} chunks baru ke vectorstore...")

    vectorstore.add_documents(chunks)

    total = vectorstore._collection.count()
    print(f"Berhasil! Total vectors sekarang: {total}")

    return vectorstore


def vectorstore_exists(persist_directory: str = "vectorstore") -> bool:
    """
    Mengecek apakah vectorstore sudah pernah dibuat.

    Args:
        persist_directory: Folder vectorstore

    Returns:
        True jika sudah ada, False jika belum
    """
    chroma_file = Path(persist_directory) / "chroma.sqlite3"
    return chroma_file.exists()
