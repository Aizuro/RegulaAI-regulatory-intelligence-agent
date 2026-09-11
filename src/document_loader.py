# src/document_loader.py
"""
Text-first PDF loader with OCR fallback.

Strategy:
1. Try extracting the PDF's existing text layer first.
2. Validate the extracted text per page.
3. OCR only pages whose text layer is missing/clearly unusable.

Important: an existing text layer may itself have been produced by OCR by
whoever created the PDF. We therefore label it as `text_layer`, not as
"error-free native text". The loader does not aggressively rewrite legal text.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


# OCR imports are optional at import time so the normal text-only path still
# works when Tesseract is not installed.
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


MIN_TEXT_CHARS = 80
MIN_ALNUM_CHARS = 30
MIN_ALNUM_RATIO = 0.20


def _text_quality(text: str) -> Tuple[float, bool]:
    """Return a lightweight quality score and whether the page is usable."""
    text = (text or "").strip()
    if not text:
        return 0.0, False

    compact = re.sub(r"\s+", "", text)
    alnum = sum(ch.isalnum() for ch in compact)
    length = len(compact)
    ratio = alnum / length if length else 0.0

    # Very short extraction is almost always insufficient for a legal page.
    usable = (
        len(text) >= MIN_TEXT_CHARS
        and alnum >= MIN_ALNUM_CHARS
        and ratio >= MIN_ALNUM_RATIO
    )

    # Score is diagnostic only; it is not a claim of OCR correctness.
    length_score = min(len(text) / 1200.0, 1.0)
    ratio_score = min(ratio / 0.75, 1.0)
    score = round(0.5 * length_score + 0.5 * ratio_score, 3)
    return score, usable


def _ocr_page(pdf_path: str, page_index: int, language: str = "ind") -> str:
    """OCR one PDF page. Requires PyMuPDF + pytesseract + Tesseract."""
    if fitz is None or pytesseract is None:
        raise RuntimeError(
            "OCR fallback membutuhkan PyMuPDF dan pytesseract. "
            "Install dependency tersebut dan pastikan Tesseract tersedia."
        )

    pdf = fitz.open(pdf_path)
    try:
        page = pdf.load_page(page_index)
        # 2x render gives Tesseract a substantially better source image than
        # a low-resolution screenshot while keeping memory reasonable.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = pix.tobytes("png")
        return pytesseract.image_to_string(image, lang=language).strip()
    finally:
        pdf.close()


def load_single_pdf(
    file_path: str,
    *,
    enable_ocr_fallback: bool = True,
    ocr_language: str = "ind",
) -> List[Document]:
    """
    Load one PDF page-by-page using text-first extraction.

    Metadata added to each Document:
      - extraction_method: `text_layer` or `ocr_fallback`
      - extraction_quality: lightweight diagnostic score
      - ocr_used: boolean

    Existing text is preserved. OCR is only used when the page's text layer
    is missing or clearly unusable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    print(f"Memuat PDF: {file_path}")

    # First pass: use the PDF's existing text layer.
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    output: List[Document] = []
    ocr_pages = []

    for page_index, doc in enumerate(documents):
        text = (doc.page_content or "").strip()
        score, usable = _text_quality(text)

        metadata = dict(doc.metadata or {})
        metadata.update({
            "extraction_method": "text_layer",
            "extraction_quality": score,
            "ocr_used": False,
        })

        if not usable and enable_ocr_fallback:
            try:
                ocr_text = _ocr_page(file_path, page_index, ocr_language)
                ocr_score, ocr_usable = _text_quality(ocr_text)

                if ocr_usable or not text:
                    text = ocr_text
                    score = ocr_score
                    metadata.update({
                        "extraction_method": "ocr_fallback",
                        "extraction_quality": score,
                        "ocr_used": True,
                    })
                    ocr_pages.append(page_index + 1)
            except Exception as exc:
                # Do not destroy a page's existing text just because OCR is
                # unavailable. Keep the text-layer result and report why OCR
                # was skipped.
                metadata["ocr_error"] = str(exc)

        output.append(Document(page_content=text, metadata=metadata))

    print(
        f"Berhasil memuat {len(output)} halaman dari {Path(file_path).name}"
    )
    if ocr_pages:
        print(f"OCR fallback digunakan pada halaman: {ocr_pages}")

    return output


def load_multiple_pdfs(
    directory,
    enable_ocr_fallback=True
):
    directory = Path(directory)

    pdf_files = sorted(directory.glob("*.pdf"))

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        documents = load_single_pdf(
            str(pdf_file),
            enable_ocr_fallback=enable_ocr_fallback
        )

        all_documents.extend(documents)

    print(f"\nTotal PDF   : {len(pdf_files)}")
    print(f"Total pages : {len(all_documents)}")

    return all_documents


def inspect_documents(documents: List[Document], num_samples: int = 3) -> None:
    """Print a few loaded pages for debugging."""
    print("\n" + "=" * 50)
    print(f"INSPEKSI HASIL LOADING ({len(documents)} total halaman)")
    print("=" * 50)

    for i, doc in enumerate(documents[:num_samples]):
        print(f"\n--- Sampel {i + 1} ---")
        print(f"Sumber     : {doc.metadata.get('source', 'N/A')}")
        print(f"Halaman    : {doc.metadata.get('page', 'N/A')}")
        print(f"Metode     : {doc.metadata.get('extraction_method', 'N/A')}")
        print(f"Kualitas   : {doc.metadata.get('extraction_quality', 'N/A')}")
        print(f"OCR used   : {doc.metadata.get('ocr_used', False)}")
        print(f"Panjang    : {len(doc.page_content)} karakter")
        print(f"Isi        : {doc.page_content[:300]}...")
