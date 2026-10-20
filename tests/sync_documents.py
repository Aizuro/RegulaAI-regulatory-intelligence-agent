# scripts/sync_documents.py
#
# Cara pakai:
#   python scripts/sync_documents.py
#   python scripts/sync_documents.py --folder data/documents --vectorstore vectorstore

from src.embeddings import (
    load_vectorstore,
    vectorstore_exists,
    get_embedding_model
)
from src.text_splitter import split_documents
from src.document_loader import load_single_pdf
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


TRACKING_FILE = "vectorstore/indexed_files.json"


# ============================================================
# FILE TRACKING
# ============================================================

def load_indexed_files() -> dict:
    if not Path(TRACKING_FILE).exists():
        return {}
    with open(TRACKING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_indexed_files(indexed: dict) -> None:
    Path(TRACKING_FILE).parent.mkdir(exist_ok=True)
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(indexed, f, indent=2, ensure_ascii=False)


def get_new_pdfs(folder_path: str, indexed_files: dict) -> list:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder tidak ditemukan: {folder_path}")
        return []
    all_pdfs = list(folder.glob("*.pdf"))
    new_pdfs = [pdf for pdf in all_pdfs if pdf.name not in indexed_files]
    return new_pdfs


def create_empty_vectorstore(vectorstore_dir: str):
    """Membuat vectorstore kosong tanpa dokumen apapun."""
    from langchain_chroma import Chroma
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=embedding_model
    )
    return vectorstore


# ============================================================
# RATE LIMIT HELPER
# ============================================================

def countdown(seconds: int, label: str = "Lanjut") -> None:
    """
    Countdown mundur yang akurat — 1 detik per iterasi.
    Lebih akurat daripada sleep(10) per langkah.
    """
    for remaining in range(seconds, 0, -1):
        print(f"  ⏳ {label} dalam {remaining} detik...   ", end="\r")
        time.sleep(1)
    print(f"  ✅ Selesai menunggu, lanjut!              ")


def add_with_rate_limit(
    vectorstore,
    chunks: list,
    batch_size: int = 50,
    delay_between_batches: int = 70,
    delay_between_files: int = 70
):
    """
    Menambahkan chunks ke vectorstore dengan dua lapis delay:
    1. Delay antar batch DALAM satu file
    2. (Dipanggil dari luar) delay antar FILE

    Args:
        vectorstore           : ChromaDB vectorstore
        chunks                : Semua chunks yang akan ditambahkan
        batch_size            : Jumlah chunks per batch (default: 50)
        delay_between_batches : Jeda antar batch dalam detik (default: 70)
        delay_between_files   : Tidak dipakai di sini, hanya dokumentasi
    """
    total = len(chunks)
    batches = [chunks[i:i+batch_size] for i in range(0, total, batch_size)]

    print(f"  Total chunks  : {total}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Jumlah batch  : {len(batches)}")

    for batch_num, batch in enumerate(batches, 1):
        print(
            f"  Mengirim batch {batch_num}/{len(batches)} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

        # Tunggu antar batch dalam satu file
        if batch_num < len(batches):
            print(
                f"  Rate limit protection: tunggu {delay_between_batches} detik...")
            countdown(delay_between_batches, label="Batch berikutnya")

    return vectorstore


# ============================================================
# FUNGSI UTAMA
# ============================================================

def sync_documents(
    documents_folder: str = "data/documents",
    vectorstore_dir: str = "vectorstore",
    batch_size: int = 50,
    delay_between_batches: int = 70,
    delay_between_files: int = 70
) -> None:
    from datetime import datetime

    print("=" * 50)
    print("SYNC DOKUMEN KE VECTORSTORE")
    print(f"Batch size          : {batch_size} chunks")
    print(f"Delay antar batch   : {delay_between_batches} detik")
    print(f"Delay antar file    : {delay_between_files} detik")
    print("=" * 50)

    # Step 1: Load tracking
    indexed_files = load_indexed_files()
    print(f"\nFile sudah terindeks: {len(indexed_files)}")
    if indexed_files:
        for fname in indexed_files:
            print(f"  ✅ {fname}")

    # Step 2: Cari PDF baru
    new_pdfs = get_new_pdfs(documents_folder, indexed_files)
    if not new_pdfs:
        print(f"\nTidak ada dokumen baru. Vectorstore sudah up-to-date!")
        return

    print(f"\nDitemukan {len(new_pdfs)} dokumen baru:")
    for pdf in new_pdfs:
        print(f"  📄 {pdf.name}")

    # Step 3: Load atau buat vectorstore kosong
    print(f"\nMemuat vectorstore...")
    if vectorstore_exists(vectorstore_dir):
        vectorstore = load_vectorstore(vectorstore_dir)
    else:
        print("Vectorstore belum ada, membuat vectorstore kosong...")
        vectorstore = create_empty_vectorstore(vectorstore_dir)
        print("Vectorstore kosong siap.")

    # Step 4: Proses setiap PDF dengan DUA LAPIS delay:
    # - Delay antar batch (di dalam add_with_rate_limit)
    # - Delay antar file (di sini, setelah tiap file selesai)
    total_chunks_added = 0

    for i, pdf_path in enumerate(new_pdfs, 1):
        print(f"\n[{i}/{len(new_pdfs)}] Memproses: {pdf_path.name}")

        try:
            documents = load_single_pdf(str(pdf_path))
            chunks = split_documents(documents)

            vectorstore = add_with_rate_limit(
                vectorstore=vectorstore,
                chunks=chunks,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches
            )

            indexed_files[pdf_path.name] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            save_indexed_files(indexed_files)

            total_chunks_added += len(chunks)
            print(f"  ✅ Berhasil! {len(chunks)} chunks ditambahkan.")

            # Delay antar FILE — supaya API "dingin" sebelum proses file berikutnya
            if i < len(new_pdfs):
                print(
                    f"\n  Jeda {delay_between_files} detik sebelum file berikutnya...")
                countdown(delay_between_files, label="File berikutnya")

        except Exception as e:
            print(f"  ❌ Gagal memproses {pdf_path.name}: {str(e)}")
            # Kalau gagal karena rate limit, tunggu lebih lama lalu lanjut
            if "429" in str(e):
                print(f"  Rate limit terdeteksi! Menunggu 120 detik...")
                countdown(120, label="Retry setelah rate limit")
            continue

    # Step 5: Ringkasan
    print("\n" + "=" * 50)
    print("SELESAI!")
    print(f"Dokumen baru diproses : {len(new_pdfs)}")
    print(f"Total chunks ditambah : {total_chunks_added}")
    print(f"Total file terindeks  : {len(indexed_files)}")
    if vectorstore:
        print(f"Total vectors di DB   : {vectorstore._collection.count()}")
    print("=" * 50)
    print("\nVectorstore siap dipakai. Sekarang jalankan Streamlit!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync dokumen PDF baru ke ChromaDB vectorstore"
    )
    parser.add_argument(
        "--folder", default="data/documents",
        help="Folder berisi file PDF (default: data/documents)"
    )
    parser.add_argument(
        "--vectorstore", default="vectorstore",
        help="Folder vectorstore ChromaDB (default: vectorstore)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Jumlah chunks per batch (default: 50)"
    )
    parser.add_argument(
        "--delay-batch", type=int, default=70,
        help="Delay antar batch dalam detik (default: 70)"
    )
    parser.add_argument(
        "--delay-file", type=int, default=70,
        help="Delay antar file dalam detik (default: 70)"
    )

    args = parser.parse_args()

    sync_documents(
        documents_folder=args.folder,
        vectorstore_dir=args.vectorstore,
        batch_size=args.batch_size,
        delay_between_batches=args.delay_batch,
        delay_between_files=args.delay_file
    )
