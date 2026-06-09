# Tahap 9: Logging

# src/logger.py

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List
from langchain.schema import Document


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Membuat dan mengkonfigurasi logger untuk aplikasi RAG.
    Dipanggil SEKALI saat aplikasi pertama start.

    Log akan disimpan di dua tempat sekaligus:
    - File  : logs/rag_YYYY-MM-DD.log (permanen, untuk analisis)
    - Console: terminal (untuk debugging saat development)

    Args:
        log_dir: folder tempat menyimpan file log

    Returns:
        Logger object siap pakai
    """
    # Buat folder logs kalau belum ada
    Path(log_dir).mkdir(exist_ok=True)

    # Nama logger unik untuk aplikasi kita
    logger = logging.getLogger("rag_hukum")
    logger.setLevel(logging.INFO)

    # Hindari duplikasi handler kalau setup_logger dipanggil lebih dari sekali
    # (ini bisa terjadi di Streamlit karena script dijalankan ulang)
    if logger.handlers:
        return logger

    # Format log: [waktu] LEVEL | pesan
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1: tulis ke file, satu file per hari
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        f"{log_dir}/rag_{today}.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Handler 2: tampilkan di console/terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_query(
    logger: logging.Logger,
    query: str,
    source_documents: List[Document],
    answer: str,
    response_time: float
) -> None:
    """
    Mencatat satu sesi tanya-jawab secara lengkap ke log.

    Args:
        logger          : Logger dari setup_logger()
        query           : Pertanyaan dari user
        source_documents: Dokumen yang diambil retriever
        answer          : Jawaban dari LLM
        response_time   : Waktu response dalam detik
    """
    # Kumpulkan info sumber dokumen
    sources_info = []
    for doc in source_documents:
        sources_info.append({
            "file": Path(doc.metadata.get("source", "unknown")).name,
            "page": doc.metadata.get("page", 0) + 1,
            "preview": doc.page_content[:100] + "..."  # 100 karakter pertama
        })

    # Buat satu log entry sebagai JSON agar mudah di-parse nanti
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response_time_seconds": round(response_time, 2),
        "num_sources": len(source_documents),
        "sources": sources_info,
        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
    }

    # Tulis ke log sebagai satu baris JSON
    logger.info(f"QUERY_LOG | {json.dumps(log_entry, ensure_ascii=False)}")


def log_document_added(
    logger: logging.Logger,
    filename: str,
    num_chunks: int
) -> None:
    """
    Mencatat ketika dokumen baru berhasil ditambahkan ke vectorstore.

    Args:
        logger    : Logger dari setup_logger()
        filename  : Nama file PDF yang ditambahkan
        num_chunks: Jumlah chunks yang dihasilkan
    """
    logger.info(
        f"DOCUMENT_ADDED | file={filename} | chunks={num_chunks}"
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: str = ""
) -> None:
    """
    Mencatat error yang terjadi.

    Args:
        logger : Logger dari setup_logger()
        error  : Exception yang terjadi
        context: Keterangan tambahan dimana error terjadi
    """
    logger.error(
        f"ERROR | context={context} | "
        f"type={type(error).__name__} | message={str(error)}"
    )
