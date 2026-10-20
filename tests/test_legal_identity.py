from pathlib import Path

from src.document_loader import load_single_pdf
from src.legal_parser import extract_regulation_identity


DOCUMENT_DIR = Path("data/documents")


EXPECTED = {
    "1959uut001.pdf": {
        "document_type": "UU",
        "regulation_number": "1",
        "year": "1959",
    },
    "1959uut003.pdf": {
        "document_type": "UU",
        "regulation_number": "3",
        "year": "1959",
    },
    "1959uut004.pdf": {
        "document_type": "UU",
        "regulation_number": "4",
        "year": "1959",
    },
    "1959uut005.pdf": {
        "document_type": "UU",
        "regulation_number": "5",
        "year": "1959",
    },
    "2024uu001.pdf": {
        "document_type": "UU",
        "regulation_number": "1",
        "year": "2024",
    },
    "2024uu002.pdf": {
        "document_type": "UU",
        "regulation_number": "2",
        "year": "2024",
    },
    "2024uu003.pdf": {
        "document_type": "UU",
        "regulation_number": "3",
        "year": "2024",
    },
    "2024UU005.pdf": {
        "document_type": "UU",
        "regulation_number": "5",
        "year": "2024",
    },
    "2025perpres004.pdf": {
        "document_type": "Perpres",
        "regulation_number": "4",
        "year": "2025",
    },
    "UUD1945.pdf": {
        "document_type": "UUD",
        "year": "1945",
    },
}


def main():
    print("=" * 80)
    print("REGRESSION TEST LEGAL IDENTITY")
    print("=" * 80)

    passed = 0
    failed = 0

    for filename, expected in EXPECTED.items():
        pdf_path = DOCUMENT_DIR / filename

        print()
        print("-" * 80)
        print(filename)

        try:
            docs = load_single_pdf(str(pdf_path))

            if not docs:
                raise AssertionError("PDF tidak menghasilkan halaman.")

            identity = extract_regulation_identity(
                docs[0].page_content,
                filename=filename,
            )

            print(f"  document_type      : {identity.get('document_type')}")
            print(
                f"  regulation_number  : {identity.get('regulation_number')}")
            print(f"  year               : {identity.get('year')}")
            print(f"  title              : {identity.get('title')}")

            errors = []

            for key, expected_value in expected.items():
                actual_value = identity.get(key)

                if actual_value != expected_value:
                    errors.append(
                        f"{key}: expected={expected_value!r}, "
                        f"actual={actual_value!r}"
                    )

            if errors:
                failed += 1
                print("  STATUS             : FAIL")

                for error in errors:
                    print(f"    - {error}")
            else:
                passed += 1
                print("  STATUS             : PASS")

        except Exception as exc:
            failed += 1
            print("  STATUS             : ERROR")
            print(f"  {type(exc).__name__}: {exc}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"PASSED : {passed}")
    print(f"FAILED : {failed}")

    if failed == 0:
        print("STATUS : PASS")
    else:
        print("STATUS : FAIL")


if __name__ == "__main__":
    main()
