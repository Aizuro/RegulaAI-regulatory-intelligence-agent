from langchain_core.documents import Document

from src.document_comparison import (
    build_v82_comparison_input,
    compare_retrieved_documents,
)


def _doc(content, *, bab="", pasal="", ayat="", page=1, source="test.pdf"):
    return Document(
        page_content=content,
        metadata={
            "document_type": "UU",
            "regulation_number": "1",
            "year": "2024",
            "bab": bab,
            "pasal": pasal,
            "ayat": ayat,
            "page": page,
            "source": source,
        },
    )


def test_build_v82_input_preserves_regulation_identity_and_documents():
    docs_a = [_doc("A", pasal="1")]
    docs_b = [_doc("B", pasal="1")]

    result = build_v82_comparison_input(
        regulation_a={"document_type": "UU",
                      "regulation_number": "1", "year": "2024"},
        documents_a=docs_a,
        regulation_b={"document_type": "UU",
                      "regulation_number": "3", "year": "2024"},
        documents_b=docs_b,
    )

    assert result["comparison_ready"] is True
    assert result["regulation_a"]["regulation_number"] == "1"
    assert result["regulation_b"]["regulation_number"] == "3"
    assert result["documents_a"][0]["content"] == "A"
    assert result["documents_a"][0]["metadata"]["pasal"] == "1"


def test_comparison_reuses_v3_engine_and_aligns_common_provision():
    docs_a = [_doc("Ketentuan yang sama", pasal="1")]
    docs_b = [_doc("Ketentuan yang berubah", pasal="1", source="new.pdf")]

    report = compare_retrieved_documents(
        regulation_a={"document_type": "UU",
                      "regulation_number": "1", "year": "2024"},
        documents_a=docs_a,
        regulation_b={"document_type": "UU",
                      "regulation_number": "3", "year": "2024"},
        documents_b=docs_b,
    )

    assert report["type"] == "structural_comparison"
    assert report["comparison_ready"] is True
    assert report["counts"]["aligned_provisions"] == 1
    assert report["aligned_provisions"][0]["textually_different"] is True
    assert report["interpretation"] == "not_performed"


def test_comparison_reports_provisions_only_present_on_each_side():
    docs_a = [_doc("Only old", pasal="1")]
    docs_b = [_doc("Only new", pasal="2")]

    report = compare_retrieved_documents(
        regulation_a={"document_type": "UU",
                      "regulation_number": "1", "year": "2024"},
        documents_a=docs_a,
        regulation_b={"document_type": "UU",
                      "regulation_number": "3", "year": "2024"},
        documents_b=docs_b,
    )

    assert report["counts"]["aligned_provisions"] == 0
    assert report["counts"]["only_in_a"] == 1
    assert report["counts"]["only_in_b"] == 1


def test_empty_side_is_not_comparison_ready():
    report = compare_retrieved_documents(
        regulation_a={"document_type": "UU",
                      "regulation_number": "1", "year": "2024"},
        documents_a=[],
        regulation_b={"document_type": "UU",
                      "regulation_number": "3", "year": "2024"},
        documents_b=[_doc("B", pasal="1")],
    )

    assert report["comparison_ready"] is False
    assert report["status"] == "missing_source"
