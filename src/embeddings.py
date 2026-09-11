
import hashlib
import os
import re
import time
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DEFAULT_PERSIST_DIRECTORY = "vectorstore"
DEFAULT_COLLECTION_NAME = "legal_regulations"
EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_BATCH_SIZE = 50
DEFAULT_REQUESTS_PER_MINUTE = 100
DEFAULT_MAX_RETRIES = 5


def get_embedding_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan!")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=api_key
    )


def _document_id(doc: Document) -> str:
    md = doc.metadata or {}
    identity = "|".join([
        str(md.get("source", "")), str(md.get("document_type", "")),
        str(md.get("regulation_number", "")), str(md.get("year", "")),
        str(md.get("bab", "")), str(md.get("pasal", "")),
        str(md.get("ayat", "")), str(md.get("amendment_article", "")),
        str(md.get("target_pasal", "")), str(md.get("page", "")),
        str(md.get("page_end", "")), doc.page_content,
    ])
    return "legal-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _document_ids(documents: List[Document]) -> List[str]:
    ids = [_document_id(d) for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Ditemukan duplicate deterministic document ID.")
    return ids


def _is_daily_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in (
        "perday", "per day", "requestsperday", "quota_value: 1000"
    ))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in (
        "429", "too many requests", "rate limit", "quota exceeded"
    ))


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", str(exc), re.I)
    if match:
        return max(float(match.group(1)) + 2.0, 2.0)
    return min(2 ** attempt, 120) + 2.0


def _open_vectorstore(model, persist_directory: str, collection_name: str):
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=model,
        collection_name=collection_name,
    )


def _existing_ids(vectorstore: Chroma, ids: List[str]) -> Set[str]:
    if not ids:
        return set()
    result = vectorstore._collection.get(ids=ids, include=[])
    return set(result.get("ids") or [])


def _embed_one_batch(model, batch, batch_number, total_batches, max_retries):
    for attempt in range(1, max_retries + 1):
        try:
            vectors = model.embed_documents([d.page_content for d in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Gemini mengembalikan {len(vectors)} embeddings "
                    f"untuk {len(batch)} documents."
                )
            return vectors
        except Exception as exc:
            if _is_daily_quota_error(exc):
                raise RuntimeError(
                    "Gemini daily embedding quota habis. "
                    "Progress batch sebelumnya sudah tersimpan di Chroma. "
                    "Jalankan kembali indexer setelah quota harian reset."
                ) from exc
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = _retry_delay_seconds(exc, attempt)
            print(
                f"Rate limit pada batch {batch_number}/{total_batches}. "
                f"Retry {attempt}/{max_retries} dalam {delay:.1f} detik..."
            )
            time.sleep(delay)


def index_legal_documents_resumable(
    documents: List[Document],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """Embed and commit each batch immediately; never delete existing vectors."""
    if not documents:
        raise ValueError("Tidak ada legal Documents untuk di-index.")

    model = get_embedding_model()
    ids = _document_ids(documents)
    vs = _open_vectorstore(model, persist_directory, collection_name)

    existing = _existing_ids(vs, ids)
    pending = [(i, d) for i, d in zip(ids, documents) if i not in existing]

    print("=" * 72)
    print("RESUMABLE LEGAL EMBEDDING INDEX")
    print("=" * 72)
    print(f"Total documents : {len(documents)}")
    print(f"Sudah ter-index : {len(existing)}")
    print(f"Perlu di-embed  : {len(pending)}")
    print()

    if not pending:
        print("Semua documents sudah ter-index.")
        return vs

    quota_limit = max(1, requests_per_minute - 5)
    window_started = time.monotonic()
    texts_in_window = 0
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for start in range(0, len(pending), batch_size):
        pairs = pending[start:start + batch_size]
        ids_batch = [x[0] for x in pairs]
        batch = [x[1] for x in pairs]
        batch_no = start // batch_size + 1

        stored_now = _existing_ids(vs, ids_batch)
        pairs = [(i, d) for i, d in pairs if i not in stored_now]
        ids_batch = [x[0] for x in pairs]
        batch = [x[1] for x in pairs]
        if not batch:
            continue

        count = len(batch)
        if texts_in_window + count > quota_limit:
            elapsed = time.monotonic() - window_started
            wait = max(0.0, 60.0 - elapsed) + 1.0
            print(f"Menunggu short-window quota {wait:.1f} detik...")
            time.sleep(wait)
            window_started = time.monotonic()
            texts_in_window = 0

        print(
            f"Embedding batch {batch_no}/{total_batches} "
            f"({count} documents)..."
        )
        vectors = _embed_one_batch(
            model, batch, batch_no, total_batches, max_retries
        )

        # Durable checkpoint: successful batch is committed immediately.
        vs._collection.add(
            ids=ids_batch,
            embeddings=vectors,
            documents=[d.page_content for d in batch],
            metadatas=[d.metadata for d in batch],
        )
        texts_in_window += count
        print(f"  ✓ Tersimpan. Chroma: {vs._collection.count()} vectors.")

        if start + batch_size < len(pending):
            time.sleep(1)

    final_count = vs._collection.count()
    print(f"Target: {len(documents)} | Chroma: {final_count}")
    if final_count != len(documents):
        raise RuntimeError(
            f"Index belum lengkap: {final_count}/{len(documents)}. "
            "Jalankan kembali untuk melanjutkan."
        )
    print("STATUS: PASS")
    return vs


def create_legal_vectorstore(
    documents: List[Document],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Backward-compatible entry point; now resumable."""
    return index_legal_documents_resumable(
        documents, persist_directory, collection_name
    )


def load_legal_vectorstore(
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"Vectorstore tidak ditemukan di '{persist_directory}'."
        )
    model = get_embedding_model()
    vs = Chroma(
        persist_directory=persist_directory,
        embedding_function=model,
        collection_name=collection_name,
    )
    print(
        f"Loaded collection '{collection_name}' "
        f"({vs._collection.count()} vectors)."
    )
    return vs


def legal_vectorstore_exists(
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> bool:
    return (Path(persist_directory) / "chroma.sqlite3").exists()


def add_legal_documents_to_vectorstore(vectorstore, documents):
    if not documents:
        return vectorstore
    ids = _document_ids(documents)
    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"Total vectors sekarang: {vectorstore._collection.count()}")
    return vectorstore
