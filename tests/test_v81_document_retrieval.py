"""V8.1 tests for the Document Retrieval Layer.

These tests mock Chroma/download boundaries. They do not require a live
embedding API, a network connection, or a populated production vectorstore.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from src.document_retrieval import (
    RegulationIdentity,
    retrieve_regulation,
    _download_pdf,
)


def _fake_local_documents():
    return [
        Document(
            page_content="Pasal 1 ketentuan...",
            metadata={
                "document_type": "PMK",
                "regulation_number": "25",
                "year": "2026",
                "page": 1,
            },
        ),
        Document(
            page_content="Pasal 2 ketentuan...",
            metadata={
                "document_type": "PMK",
                "regulation_number": "25",
                "year": "2026",
                "page": 2,
            },
        ),
    ]


def test_regulation_identity_builds_stable_key():
    identity = RegulationIdentity("PMK", "25", "2026")

    assert identity.regulation_key == "PMK-25-2026"


@patch("src.document_retrieval._load_existing_chunks")
def test_retrieval_prefers_local_corpus(mock_load):
    mock_load.return_value = _fake_local_documents()

    result = retrieve_regulation(
        {
            "type": "PMK",
            "number": "25",
            "year": "2026",
            "title": "Contoh",
            "url": "https://example.com/regulation.pdf",
        }
    )

    print("RESULT:", result)
    print("ERROR:", result.error)
    assert result.status == "ok"
    assert result.source == "local"
    assert result.retrieval_method == "local_chroma_metadata"
    assert len(result.documents) == 2


@patch("src.document_retrieval._fetch_and_ingest")
@patch("src.document_retrieval._load_existing_chunks")
def test_retrieval_fetches_external_pdf_when_local_missing(
    mock_load,
    mock_fetch,
):
    mock_load.return_value = []
    mock_fetch.return_value = [
        Document(
            page_content="Current regulation",
            metadata={"page": 1},
        )
    ]

    result = retrieve_regulation(
        {
            "type": "PMK",
            "number": "99",
            "year": "2026",
            "title": "Regulasi baru",
            "url": "https://example.com/regulation.pdf",
        }
    )

    assert result.status == "ok"
    assert result.source == "external"
    assert result.retrieval_method == "external_pdf_ingestion"
    mock_fetch.assert_called_once()


@patch("src.document_retrieval._load_existing_chunks")
def test_missing_local_and_no_url_returns_not_found(mock_load):
    mock_load.return_value = []

    result = retrieve_regulation(
        {
            "type": "PMK",
            "number": "100",
            "year": "2026",
            "title": "Tanpa URL",
        }
    )

    assert result.status == "not_found"
    assert result.documents == []
    assert "source URL" in result.error


@patch("src.document_retrieval._load_existing_chunks")
def test_fetch_can_be_disabled(mock_load):
    mock_load.return_value = []

    result = retrieve_regulation(
        {
            "type": "PMK",
            "number": "101",
            "year": "2026",
            "title": "Tanpa fetch",
            "url": "https://example.com/regulation.pdf",
        },
        fetch_if_missing=False,
    )

    assert result.status == "not_found"
    assert result.source == "none"


def test_download_rejects_non_https():
    destination = __import__("pathlib").Path("unused.pdf")

    try:
        _download_pdf("http://example.com/regulation.pdf", destination)
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("HTTP URL seharusnya ditolak")
