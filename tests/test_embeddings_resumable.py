
import gc
import shutil
import tempfile

import embeddings_v2
from langchain.schema import Document


class FakeEmbeddings:
    def embed_documents(self, texts):
        out = []
        for text in texts:
            v = float(sum(map(ord, text)) % 1000) / 1000.0
            out.append([v + i * 0.01 for i in range(8)])
        return out

    def embed_query(self, text):
        v = float(sum(map(ord, text)) % 1000) / 1000.0
        return [v + i * 0.01 for i in range(8)]


def main():
    docs = [
        Document(
            page_content="Pasal 17 ayat (2a).",
            metadata={
                "source": "2024uu001.pdf",
                "document_type": "UU",
                "regulation_number": "1",
                "year": 2024,
                "pasal": "17",
                "ayat": "2a",
                "page": 6,
                "page_end": 7,
                "block_type": "ayat",
                "ingestion_version": "test",
            },
        ),
        Document(
            page_content="Pasal 18 ayat (1).",
            metadata={
                "source": "2024uu001.pdf",
                "document_type": "UU",
                "regulation_number": "1",
                "year": 2024,
                "pasal": "18",
                "ayat": "1",
                "page": 8,
                "page_end": 8,
                "block_type": "ayat",
                "ingestion_version": "test",
            },
        ),
    ]

    original = embeddings_v2.get_embedding_model
    embeddings_v2.get_embedding_model = lambda: FakeEmbeddings()

    tmp = tempfile.mkdtemp(prefix="legal_resume_test_")

    try:
        vs = embeddings_v2.index_legal_documents_resumable(
            docs, tmp, "test_collection", batch_size=1
        )
        assert vs._collection.count() == 2

        # Release Chroma references before Windows tries to delete
        # the temporary HNSW files.
        del vs
        gc.collect()

        vs2 = embeddings_v2.index_legal_documents_resumable(
            docs, tmp, "test_collection", batch_size=1
        )
        assert vs2._collection.count() == 2

        # Release the second Chroma instance as well.
        del vs2
        gc.collect()

    finally:
        embeddings_v2.get_embedding_model = original

        # Chroma/HNSW can briefly retain Windows file handles. Retry cleanup
        # after garbage collection instead of letting TemporaryDirectory
        # produce a traceback after a successful test.
        for _ in range(3):
            try:
                shutil.rmtree(tmp)
                break
            except (PermissionError, OSError):
                gc.collect()
        else:
            print(
                f"WARNING: Temporary test directory could not be removed: {tmp}"
            )

    print("RESUMABLE EMBEDDING TEST: PASS")


if __name__ == "__main__":
    main()
