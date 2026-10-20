from pprint import pprint
from src.document_retrieval import retrieve_regulation

LOCAL_REGULATION = {
    "type": "UU",
    "number": "1",
    "year": "1959",
    "title": "BADAN PERUSAHAAN PRODUKSI BAHAN MAKANAN DAN PEMBUKAAN TANAH",
    "url": "https://example.com/not-used-for-local-test.pdf",
}

EXTERNAL_REGULATION = {
    "type": "PMK",
    "number": "64",
    "year": "2026",
    "title": "Petunjuk Teknis Akuntansi Penerimaan Negara Bukan Pajak dari Kegiatan Usaha Hulu Minyak dan Gas Bumi",
    "url": "https://jdih.kemenkeu.go.id/api/download/5b804770-d79a-4a36-8b4c-1ec4a77ec159/2026pmkeuangan064.pdf",
}


def summarize(label, result):
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)

    print("status:", result.status)
    print("source:", result.source)
    print("regulation_key:", result.regulation_key)
    print("retrieval_method:", result.retrieval_method)
    print("source_url:", result.source_url)
    print("document_count:", len(result.documents))
    print("error:", result.error)

    for i, doc in enumerate(result.documents[:3], 1):
        print(f"\n--- chunk {i} ---")
        pprint(doc.metadata)
        print(doc.page_content[:500].replace("\n", " "))


def run_local_check():
    result = retrieve_regulation(
        LOCAL_REGULATION,
        fetch_if_missing=False,
    )

    summarize("LOCAL CHROMA RETRIEVAL", result)

    assert result.status == "ok", result.error
    assert result.source == "local"
    assert result.documents

    required = {
        "document_type",
        "regulation_number",
        "year",
    }

    missing = required - set(result.documents[0].metadata)

    assert not missing, f"Missing metadata: {missing}"

    print("\nLOCAL CHECK: PASS")


def run_external_check():
    url = EXTERNAL_REGULATION["url"]

    if not url.startswith("https://") or "PASTE_REAL" in url:
        print(
            "\nEXTERNAL CHECK: SKIPPED "
            "- set a real JDIH PDF URL first."
        )
        return

    result = retrieve_regulation(
        EXTERNAL_REGULATION,
        fetch_if_missing=True,
    )

    summarize("EXTERNAL JDIH PDF RETRIEVAL", result)

    assert result.status == "ok", result.error
    assert result.source == "external"
    assert result.documents

    methods = {
        doc.metadata.get("retrieval_method")
        for doc in result.documents
    }

    assert "external_pdf" in methods, methods

    print("\nEXTERNAL CHECK: PASS")


if __name__ == "__main__":
    run_local_check()
    run_external_check()

    print("\nV8.1 INTEGRATION CHECK COMPLETE")
