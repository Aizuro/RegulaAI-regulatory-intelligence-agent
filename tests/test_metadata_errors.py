"""Focused diagnostic for metadata audit ERRORs.
Read-only: does not modify or re-index the Chroma collection.
"""
import re
from collections import Counter
from src.embeddings import load_legal_vectorstore

BAD_TOKENS = ("168", "2O", "2l", "3O", "3OA")


def norm(v):
    return str(v).strip() if v is not None else None


vs = load_legal_vectorstore()
collection = vs._collection
data = collection.get(include=["documents", "metadatas"])

print("=" * 80)
print("FOCUSED METADATA ERROR DIAGNOSTIC")
print("=" * 80)
print(f"Collection: {collection.name}")
print(f"Vectors  : {collection.count()}")

found = []
for idx, (content, m) in enumerate(zip(data["documents"], data["metadatas"]), start=1):
    pasal = norm(m.get("pasal"))
    ayat = norm(m.get("ayat"))
    hits = []
    for token in BAD_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", pasal or ""):
            hits.append(("pasal", token))
        if re.search(rf"\b{re.escape(token)}\b", ayat or ""):
            hits.append(("ayat", token))
    if hits:
        found.append((idx, m, content, hits))

print(f"BAD OCR metadata records: {len(found)}")
for idx, m, content, hits in found:
    print("\n" + "-" * 80)
    print(f"Record #{idx}")
    print("Hits      :", hits)
    print("Source    :", m.get("source"))
    print("Type      :", m.get("document_type"))
    print("Reg       :", m.get("regulation_number"))
    print("Year      :", m.get("year"))
    print("Pasal     :", repr(m.get("pasal")))
    print("Ayat      :", repr(m.get("ayat")))
    print("Page      :", m.get("page"), "-", m.get("page_end"))
    print("Target    :", repr(m.get("target_pasal")))
    print("Amendment :", repr(m.get("amendment_article")))
    print("Content   :")
    print((content or "")[:1200].replace("\n", " "))

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
if not found:
    print("No BAD OCR tokens exist in Chroma metadata. The previous 4 ERRORs were caused by the audit logic/environment, not stored metadata.")
else:
    print("The 4 ERROR candidates are real metadata values and must be traced back to ingestion/parser output before re-indexing.")
