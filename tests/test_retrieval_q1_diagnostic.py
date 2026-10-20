from embeddings_v2 import load_legal_vectorstore
from src.retriever import get_retriever

QUERY = "Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?"


def show(label, docs):
    print(f"\n{'=' * 78}")
    print(label)
    print("=" * 78)
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        print(
            f"{i}. {m.get('source')} | type={m.get('document_type')} "
            f"| reg={m.get('regulation_number')} | year={m.get('year')} "
            f"| pasal={m.get('pasal')} | ayat={m.get('ayat')} "
            f"| page={m.get('page')} | page_end={m.get('page_end')}"
        )
        print(f"   {doc.page_content[:220].replace(chr(10), ' ')}")


vectorstore = load_legal_vectorstore()
print(f"Collection: {vectorstore._collection.name}")
print(f"Vectors: {vectorstore._collection.count()}")

# 1) Production-style MMR, but with a larger k for diagnosis only.
mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7},
)
show("MMR k=10", mmr.invoke(QUERY))

# 2) Pure similarity, k=10.
similarity = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)
show("SIMILARITY k=10", similarity.invoke(QUERY))

# 3) Exact legal metadata filter.
flt = {
    "$and": [
        {"document_type": {"$eq": "UU"}},
        {"regulation_number": {"$eq": "1"}},
        {"year": {"$eq": "2024"}},
        {"pasal": {"$eq": "17"}},
        {"ayat": {"$eq": "2a"}},
    ]
}
exact = vectorstore.similarity_search(
    QUERY,
    k=10,
    filter=flt,
)
show("EXACT METADATA: UU 1/2024 Pasal 17 ayat 2a", exact)

print("\n" + "=" * 78)
print("DIAGNOSTIC COMPLETE")
print("=" * 78)
