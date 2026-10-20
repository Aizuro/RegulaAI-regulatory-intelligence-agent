"""Unit tests for Chroma metadata normalization without LangChain."""

# Import the module normally in the real project; this test is mainly a
# specification of expected primitive metadata behavior.
from src.legal_ingestion import _safe_metadata_value, _build_metadata


def main():
    assert _safe_metadata_value(None) is None
    assert _safe_metadata_value(["6", "7"]) == "6,7"
    assert _safe_metadata_value((1, 2)) == "1,2"
    assert _safe_metadata_value(3) == 3
    assert _safe_metadata_value(True) is True

    chunk = {
        "bab": "BAB III",
        "pasal": "17",
        "ayat": "2a",
        "block_type": "ayat",
        "page": 6,
        "page_end": 7,
        "source_pages": ["6", "7"],
        "is_cross_page": True,
        "merged_blocks": 2,
        "quality": {"cleaning_version": "v1.1"},
    }
    identity = {
        "document_type": "UU",
        "regulation_number": "1",
        "year": "2024",
        "title": "Test",
    }
    md = _build_metadata(chunk, identity, "2024uu001.pdf", {})

    assert md["source"] == "2024uu001.pdf"
    assert md["source_pages"] == "6,7"
    assert md["is_cross_page"] is True
    assert "quality" not in md
    assert all(value is not None for value in md.values())

    print("LEGAL INGESTION METADATA UNIT TEST: PASS")


if __name__ == "__main__":
    main()
