"""
Metadata audit for the production legal Chroma collection.

Checks:
- all 776 vectors are readable
- required document identity metadata
- page metadata
- Pasal/Ayat consistency between content and metadata
- UUD special handling (no regulation_number required)
- suspicious OCR/legal identifiers that could indicate parser leakage
"""

import re
from collections import Counter, defaultdict

from src.embeddings import load_legal_vectorstore


EXPECTED_COUNT = 776


def norm(value):
    return str(value).strip() if value is not None else None


def first_pasal_in_content(content):
    m = re.search(r"\bPasal\s+(\d+[A-Za-z]?)\b", content, re.I)
    return m.group(1).upper() if m else None


def ayat_in_content(content):
    matches = re.findall(
        r"\(\s*(\d+[A-Za-z]?)\s*\)",
        content,
        re.I,
    )
    return [m.lower() for m in matches]


vs = load_legal_vectorstore()
collection = vs._collection

count = collection.count()
print("=" * 78)
print("LEGAL METADATA AUDIT")
print("=" * 78)
print(f"Collection : {collection.name}")
print(f"Vectors    : {count}")

assert count == EXPECTED_COUNT, (
    f"Expected {EXPECTED_COUNT} vectors, found {count}"
)

data = collection.get(include=["documents", "metadatas"])

docs = data["documents"]
metadatas = data["metadatas"]

issues = []
severity = Counter()
by_source = Counter()

for idx, (content, m) in enumerate(zip(docs, metadatas), start=1):
    source = m.get("source")
    dtype = norm(m.get("document_type"))
    reg = norm(m.get("regulation_number"))
    year = norm(m.get("year"))
    pasal = norm(m.get("pasal"))
    ayat = norm(m.get("ayat"))
    page = m.get("page")
    page_end = m.get("page_end")

    def add(code, detail, level="WARN"):
        issues.append({
            "idx": idx,
            "source": source,
            "level": level,
            "code": code,
            "detail": detail,
        })
        severity[f"{level}:{code}"] += 1
        by_source[source] += 1

    # Identity
    if not source:
        add("MISSING_SOURCE", "source kosong", "ERROR")

    if not dtype:
        add("MISSING_DOCUMENT_TYPE", "document_type kosong", "ERROR")

    if dtype != "UUD":
        if not reg:
            add("MISSING_REGULATION_NUMBER", "regulation_number kosong", "ERROR")
        if not year:
            add("MISSING_YEAR", "year kosong", "ERROR")

    # Page
    if not isinstance(page, int) or page < 1:
        add("INVALID_PAGE", f"page={page!r}", "ERROR")

    if page_end is not None:
        if not isinstance(page_end, int) or page_end < page:
            add("INVALID_PAGE_END",
                f"page={page!r}, page_end={page_end!r}", "ERROR")

    # Pasal consistency: only flag a strong mismatch when metadata content
    # clearly starts a Pasal different from metadata.
    content_pasal = first_pasal_in_content(content or "")
    if pasal and content_pasal and content_pasal != pasal:
        add(
            "PASAL_CONTENT_MISMATCH",
            f"metadata pasal={pasal}, first content Pasal={content_pasal}",
            "WARN",
        )

    # Ayat consistency: if metadata says an ayat, ensure that marker occurs.
    if ayat:
        ayat_markers = ayat_in_content(content or "")
        if ayat not in ayat_markers:
            add(
                "AYAT_MARKER_MISSING",
                f"metadata ayat={ayat}, content markers={ayat_markers[:10]}",
                "WARN",
            )

    # Known bad OCR identifiers from previous parser regression.
    bad_tokens = ("168", "2O", "2l", "3O", "3OA")
    for token in bad_tokens:
        if re.search(rf"\b{re.escape(token)}\b", pasal or ""):
            add("BAD_PASAL_OCR", f"pasal={pasal}", "ERROR")
        if re.search(rf"\b{re.escape(token)}\b", ayat or ""):
            add("BAD_AYAT_OCR", f"ayat={ayat}", "ERROR")

# Summaries
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)

print(f"Total records audited : {len(metadatas)}")
print(f"Total issues          : {len(issues)}")
print(
    f"Errors                : {sum(v for k, v in severity.items() if k.startswith('ERROR:'))}")
print(
    f"Warnings              : {sum(v for k, v in severity.items() if k.startswith('WARN:'))}")

if severity:
    print("\nIssue types:")
    for key, value in sorted(severity.items()):
        print(f"  {key}: {value}")

if issues:
    print("\nFirst 30 issues:")
    for item in issues[:30]:
        print(
            f"  [{item['level']}] #{item['idx']} "
            f"{item['source']} | {item['code']} | {item['detail']}"
        )

    print("\nSources with issues:")
    for source, value in by_source.most_common():
        print(f"  {source}: {value}")

errors = [x for x in issues if x["level"] == "ERROR"]

print("\n" + "=" * 78)
if errors:
    print("STATUS: FAIL")
    print("Metadata audit menemukan issue ERROR yang perlu diperbaiki.")
    raise SystemExit(1)

print("STATUS: PASS")
print("Tidak ada metadata ERROR pada 776 vectors.")
print("WARN, jika ada, bersifat kandidat audit manual dan tidak otomatis dianggap bug.")
