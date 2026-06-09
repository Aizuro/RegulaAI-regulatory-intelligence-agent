# Tahap 3: Pecah Dokumen

# src/text_splitter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Memecah list of Document menjadi chunks yang lebih kecil.

    Args:
        documents   : Hasil dari document_loader
        chunk_size  : Jumlah maksimal karakter per chunk
        chunk_overlap: Jumlah karakter yang overlap antar chunk

    Returns:
        List of Document yang sudah dipecah menjadi chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Urutan pemisah: coba pisah di paragraph dulu,
        # kalau masih terlalu besar, pisah di baris baru,
        # kalau masih besar, pisah di spasi, terakhir per karakter
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    print(f"Dokumen asli  : {len(documents)} halaman")
    print(f"Setelah split : {len(chunks)} chunks")
    print(f"Chunk size    : {chunk_size} karakter")
    print(f"Chunk overlap : {chunk_overlap} karakter")

    return chunks


def inspect_chunks(chunks: List[Document], num_samples: int = 3) -> None:
    """
    Menampilkan sampel chunks untuk verifikasi hasil splitting.

    Args:
        chunks      : Hasil dari split_documents()
        num_samples : Berapa chunk yang ditampilkan sebagai sampel
    """
    print("\n" + "="*50)
    print(f"INSPEKSI CHUNKS ({len(chunks)} total chunks)")
    print("="*50)

    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Sumber  : {chunk.metadata.get('source', 'N/A')}")
        print(f"Halaman : {chunk.metadata.get('page', 'N/A')}")
        print(f"Panjang : {len(chunk.page_content)} karakter")
        print(f"Isi     :\n{chunk.page_content}")
        print("-" * 30)

    # Statistik ukuran chunk
    chunk_lengths = [len(c.page_content) for c in chunks]
    print(f"\nStatistik ukuran chunk:")
    print(f"  Terpendek : {min(chunk_lengths)} karakter")
    print(f"  Terpanjang: {max(chunk_lengths)} karakter")
    print(f"  Rata-rata : {sum(chunk_lengths) // len(chunk_lengths)} karakter")
