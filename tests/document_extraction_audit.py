from pathlib import Path
import re
import statistics

try:
    import fitz
except ImportError:
    raise SystemExit("PyMuPDF belum terinstall. Jalankan: pip install pymupdf")

DOCUMENT_DIR = Path("data/documents")
LOW_QUALITY_THRESHOLD = 0.70

SUSPICIOUS_PATTERNS = {
    "PRES!DEN": re.compile(r"PRES!DEN", re.I),
    "INDONE!3IA": re.compile(r"INDONE!3IA", re.I),
    "REPIJBUK": re.compile(r"REPIJBUK", re.I),
    "UNDANG.UNDANG": re.compile(r"UNDANG[.\-]UNDANG", re.I),
    "PasaJ": re.compile(r"\bPasaJ\b", re.I),
    "Pasal_168": re.compile(r"\bPasal\s+168\b", re.I),
    "exclamation": re.compile(r"[A-Za-z]+![A-Za-z0-9]+"),
}

def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def text_quality(text):
    text = normalize_text(text)
    if not text:
        return 0.0
    alnum = sum(ch.isalnum() for ch in text)
    ratio = alnum / len(text)
    length_score = min(len(text) / 1200.0, 1.0)
    ratio_score = min(ratio / 0.70, 1.0)
    return round(0.45 * length_score + 0.55 * ratio_score, 3)

def find_suspicious_patterns(text):
    return [(name, len(pattern.findall(text or "")))
            for name, pattern in SUSPICIOUS_PATTERNS.items()
            if pattern.findall(text or "")]

def audit_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    suspicious_total = {}
    for page_number, page in enumerate(doc, 1):
        text = page.get_text("text")
        compact = normalize_text(text)
        alnum = sum(ch.isalnum() for ch in compact)
        ratio = alnum / len(compact) if compact else 0
        quality = text_quality(compact)
        suspicious = find_suspicious_patterns(text)
        for name, count in suspicious:
            suspicious_total[name] = suspicious_total.get(name, 0) + count
        pages.append({
            "page": page_number,
            "chars": len(compact),
            "alnum_ratio": round(ratio, 3),
            "quality": quality,
            "suspicious": suspicious,
        })
    doc.close()
    qualities = [p["quality"] for p in pages]
    return {
        "file": pdf_path.name,
        "pages": len(pages),
        "avg_quality": round(statistics.mean(qualities), 3),
        "min_quality": min(qualities),
        "low_quality_pages": [p for p in pages if p["quality"] < LOW_QUALITY_THRESHOLD],
        "suspicious_total": suspicious_total,
        "page_results": pages,
    }

def print_report(r):
    print("\n" + "=" * 80)
    print(f"FILE: {r['file']}")
    print("=" * 80)
    print(f"Pages       : {r['pages']}")
    print(f"Avg quality : {r['avg_quality']}")
    print(f"Min quality : {r['min_quality']}")
    print(f"Low quality : {len(r['low_quality_pages'])} page(s) (< {LOW_QUALITY_THRESHOLD})")
    print("\nSuspicious OCR patterns:")
    if r["suspicious_total"]:
        for name, count in sorted(r["suspicious_total"].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {count}")
    else:
        print("  - None detected")
    if r["low_quality_pages"]:
        print("\nLow-quality pages:")
        for p in r["low_quality_pages"]:
            print(f"  - Page {p['page']}: quality={p['quality']}, chars={p['chars']}, alnum_ratio={p['alnum_ratio']}")
    suspicious_pages = [p for p in r["page_results"] if p["suspicious"]]
    if suspicious_pages:
        print("\nPages with suspicious OCR patterns:")
        for p in suspicious_pages:
            patterns = ", ".join(f"{n} ({c})" for n, c in p["suspicious"])
            print(f"  - Page {p['page']}: {patterns}")

def main():
    if not DOCUMENT_DIR.exists():
        raise SystemExit(f"Folder tidak ditemukan: {DOCUMENT_DIR.resolve()}")
    pdf_files = sorted(DOCUMENT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"Tidak ada PDF di: {DOCUMENT_DIR.resolve()}")

    print("=" * 80)
    print("DOCUMENT EXTRACTION AUDIT")
    print("=" * 80)
    print(f"Directory: {DOCUMENT_DIR}")
    print(f"PDF files : {len(pdf_files)}")

    results = []
    for pdf_path in pdf_files:
        print(f"\nAuditing: {pdf_path.name}")
        try:
            r = audit_pdf(pdf_path)
            results.append(r)
            print_report(r)
        except Exception as exc:
            print(f"ERROR: {exc}")

    print("\n" + "=" * 80)
    print("CORPUS SUMMARY")
    print("=" * 80)
    total_pages = sum(r["pages"] for r in results)
    all_q = [p["quality"] for r in results for p in r["page_results"]]
    low = sum(len(r["low_quality_pages"]) for r in results)
    print(f"PDF files       : {len(results)}")
    print(f"Total pages     : {total_pages}")
    print(f"Average quality : {statistics.mean(all_q):.3f}")
    print(f"Minimum quality : {min(all_q):.3f}")
    print(f"Low-quality     : {low} page(s) (< {LOW_QUALITY_THRESHOLD})")
    print("\nFiles ranked by average quality:")
    for r in sorted(results, key=lambda x: x["avg_quality"]):
        print(f"  {r['file']:<25} avg={r['avg_quality']:.3f} min={r['min_quality']:.3f} low_pages={len(r['low_quality_pages'])}")
    print("\nCatatan: quality adalah heuristik, bukan confidence OCR. Script hanya audit; tidak mengubah PDF.")

if __name__ == "__main__":
    main()
