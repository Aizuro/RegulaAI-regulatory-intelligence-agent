"""Integration test for embeddings_v2.py.

Tests the actual public/module-level API without requiring the real
Google Gemini Embedding API. The module's get_embedding_model is patched
temporarily with a deterministic fake embedding implementation.
"""

import shutil
import tempfile

from langchain.schema import Document
from langchain_core.embeddings import Embeddings

import embeddings_v2
from embeddings_v2 import (
    _document_id,
    _document_ids,
    create_legal_vectorstore,
    load_legal_vectorstore,
    add_legal_documents_to_vectorstore,
    legal_vectorstore_exists,
)


class FakeEmbeddings(Embeddings):
    """Small deterministic embedding model for local integration testing."""

    def _embed(self, text):
        values = [0.0] * 8

        for i, byte in enumerate(text.encode("utf-8")):
            values[i % 8] += float(byte)

        norm = max(sum(abs(x) for x in values), 1.0)

        return [x / norm for x in values]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def main():
    temp_dir = tempfile.mkdtemp(
        prefix="legal_embeddings_test_"
    )

    docs = [
        Document(
            page_content=(
                "Pasal 17 ayat (2a) ketentuan pengaturan sistem."
            ),
            metadata={
                "source": "2024uu001.pdf",
                "document_type": "UU",
                "regulation_number": "1",
                "year": 2024,
                "pasal": "17",
                "ayat": "2a",
                "page": 6,
                "page_end": 7,
                "source_pages": "6,7",
                "is_cross_page": True,
                "block_type": "ayat",
            },
        ),
        Document(
            page_content=(
                "Pasal 18 ayat (1) ketentuan lainnya."
            ),
            metadata={
                "source": "2024uu001.pdf",
                "document_type": "UU",
                "regulation_number": "1",
                "year": 2024,
                "pasal": "18",
                "ayat": "1",
                "page": 8,
                "page_end": 8,
                "source_pages": "8",
                "is_cross_page": False,
                "block_type": "ayat",
            },
        ),
    ]

    # Patch module-level embedding factory so the test
    # does not call the real Gemini API.
    original_get_embedding_model = (
        embeddings_v2.get_embedding_model
    )

    embeddings_v2.get_embedding_model = (
        lambda: FakeEmbeddings()
    )

    try:

        # =========================================================
        # 1. Deterministic IDs
        # =========================================================

        id1 = _document_id(docs[0])
        id2 = _document_id(docs[0])

        assert id1 == id2, (
            "Document ID tidak deterministic."
        )

        assert isinstance(id1, str), (
            "Document ID harus berupa string."
        )

        assert len(id1) > 0, (
            "Document ID tidak boleh kosong."
        )

        # embeddings_v2.py menghasilkan:
        #
        # "legal-" + 32 hexadecimal characters
        #
        # sehingga total panjangnya adalah 38 karakter.
        assert id1.startswith("legal-"), (
            "Prefix ID harus 'legal-'."
        )

        assert len(id1) == 38, (
            "Total panjang ID seharusnya 38 karakter "
            "(legal- + 32 hex)."
        )

        ids = _document_ids(docs)

        assert len(ids) == len(docs)

        assert len(set(ids)) == len(ids), (
            "Document IDs harus unik."
        )

        # =========================================================
        # 2. Create Chroma vectorstore
        # =========================================================

        vectorstore = create_legal_vectorstore(
            documents=docs,
            persist_directory=temp_dir,
            collection_name="legal_regulations_test",
        )

        assert vectorstore is not None

        assert legal_vectorstore_exists(
            persist_directory=temp_dir,
        ), (
            "Vectorstore seharusnya terdeteksi."
        )

        assert vectorstore._collection.count() == 2, (
            "Jumlah vector setelah create harus 2."
        )

        # =========================================================
        # 3. Reload vectorstore
        # =========================================================

        loaded = load_legal_vectorstore(
            persist_directory=temp_dir,
            collection_name="legal_regulations_test",
        )

        assert loaded._collection.count() == 2, (
            "Jumlah dokumen setelah reload harus 2."
        )

        # =========================================================
        # 4. Retrieval smoke test
        # =========================================================

        result = loaded.similarity_search(
            "Pasal 17 ayat 2a",
            k=1,
        )

        assert result, (
            "Similarity search tidak mengembalikan hasil."
        )

        assert "pasal" in result[0].metadata, (
            "Hasil retrieval tidak memiliki metadata pasal."
        )

        # =========================================================
        # 5. Add another document
        # =========================================================

        new_doc = Document(
            page_content="Ketentuan Pasal 20.",
            metadata={
                "source": "2024uu001.pdf",
                "document_type": "UU",
                "regulation_number": "1",
                "year": 2024,
                "pasal": "20",
                "ayat": "1",
                "page": 10,
                "page_end": 10,
                "source_pages": "10",
                "is_cross_page": False,
                "block_type": "ayat",
            },
        )

        add_legal_documents_to_vectorstore(
            loaded,
            [new_doc],
        )

        assert loaded._collection.count() == 3, (
            "Jumlah vector setelah add harus 3."
        )

        # =========================================================
        # RESULT
        # =========================================================

        print("=" * 72)
        print("EMBEDDINGS V2 / CHROMA INTEGRATION TEST")
        print("=" * 72)

        print("Deterministic ID: PASS")
        print("ID format (legal- + 32 hex): PASS")
        print("Create vectorstore: PASS")
        print("Detect vectorstore: PASS")
        print("Reload vectorstore: PASS")
        print("Similarity search: PASS")
        print("Add documents: PASS")

        print("=" * 72)
        print("STATUS: PASS")
        print("=" * 72)

    finally:
        # Restore the original Gemini embedding factory.
        embeddings_v2.get_embedding_model = (
            original_get_embedding_model
        )

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
