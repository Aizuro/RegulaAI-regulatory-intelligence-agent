# Tahap 2: Baca PDF

# src/document_loader.py

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List


def load_single_pdf(file_path: str) -> List[Document]:
    """
    Memuat satu file PDF dan mengembalikan list of Document.
    Setiap halaman PDF menjadi satu Document terpisah.

    Args:
        file_path: Path ke file PDF

    Returns:
        List of Document objects
    """
    # Cek apakah file ada sebelum mencoba membuka
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    print(f"Memuat PDF: {file_path}")

    # PyPDFLoader membaca PDF halaman per halaman
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(
        f"Berhasil memuat {len(documents)} halaman dari {Path(file_path).name}")

    return documents


def load_multiple_pdfs(folder_path: str) -> List[Document]:
    """
    Memuat semua file PDF dari sebuah folder.

    Args:
        folder_path: Path ke folder yang berisi PDF

    Returns:
        List of Document objects dari semua PDF
    """
    all_documents = []
    folder = Path(folder_path)

    # Cek apakah folder ada
    if not folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")

    # Cari semua file .pdf di folder tersebut
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"Tidak ada file PDF di folder: {folder_path}")
        return []

    print(f"Ditemukan {len(pdf_files)} file PDF")

    # Loop setiap PDF dan muat satu per satu
    for pdf_file in pdf_files:
        documents = load_single_pdf(str(pdf_file))
        all_documents.extend(documents)

    print(
        f"\nTotal: {len(all_documents)} halaman dari {len(pdf_files)} dokumen")

    return all_documents


def inspect_documents(documents: List[Document], num_samples: int = 3) -> None:
    """
    Menampilkan sampel dokumen untuk memverifikasi hasil loading.
    Fungsi ini hanya untuk debugging/pengecekan, tidak dipakai di production.

    Args:
        documents: List of Document yang sudah dimuat
        num_samples: Berapa dokumen yang ditampilkan sebagai sampel
    """
    print("\n" + "="*50)
    print(f"INSPEKSI HASIL LOADING ({len(documents)} total halaman)")
    print("="*50)

    for i, doc in enumerate(documents[:num_samples]):
        print(f"\n--- Sampel {i+1} ---")
        print(f"Sumber  : {doc.metadata.get('source', 'N/A')}")
        print(f"Halaman : {doc.metadata.get('page', 'N/A')}")
        print(f"Panjang : {len(doc.page_content)} karakter")
        # Tampilkan 300 karakter pertama saja
        print(f"Isi     : {doc.page_content[:300]}...")
