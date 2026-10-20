from src.embeddings import load_legal_vectorstore
from src.legal_tools import search_regulations, get_regulation, compare_regulations

vs = load_legal_vectorstore()
assert vs._collection.count() == 776

print("=" * 78)
print("LEGAL TOOLS TEST")
print("=" * 78)

docs = search_regulations(
    vs,
    "perubahan Undang-Undang tentang Informasi dan Transaksi Elektronik",
    k=3,
)
assert docs
print("\nsearch_regulations: PASS")

docs = get_regulation(
    vs,
    document_type="UU",
    regulation_number="1",
    year="2024",
    pasal="17",
    ayat="2a",
    k=5,
)
assert docs
assert any(
    d["source"] == "2024uu001.pdf"
    and d["pasal"] == "17"
    and d["ayat"] == "2a"
    for d in docs
)
print("get_regulation: PASS")

comparison = compare_regulations(
    vs,
    {"document_type": "UU", "regulation_number": "1", "year": "2024"},
    {"document_type": "UU", "regulation_number": "19", "year": "2016"},
    k=20,
)

assert len(comparison["documents_a"]) == 20
assert len(comparison["documents_b"]) == 0
assert comparison["comparison_ready"] is False
assert comparison["status"] == "missing_source"
assert comparison["missing_regulations"] == [
    {"document_type": "UU", "regulation_number": "19", "year": "2016"}
]

print("compare_regulations: PASS")
print("  status:", comparison["status"])
print("  missing:", comparison["missing_regulations"])
print("\nSTATUS: PASS")
