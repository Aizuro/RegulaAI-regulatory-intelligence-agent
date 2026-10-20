"""Unit/regression test for Legal Chunk Cleaner V1.1."""

from legal_chunk_cleaner_v1_1 import clean_chunk_content


def assert_removed(text, expected):
    result = clean_chunk_content(text)
    assert expected not in result, f"Expected removal failed: {expected!r}"


def assert_preserved(text, expected):
    result = clean_chunk_content(text)
    assert expected in result, f"Substantive text was removed: {expected!r}"


def main():
    print("=" * 80)
    print("LEGAL CHUNK CLEANER V1.1 UNIT TEST")
    print("=" * 80)

    # Layout artifacts must be removed.
    assert_removed(
        "Pasal 7\nSK No 205872 A\n-4-\nPresiden dan Wakil Presiden...",
        "SK No 205872 A",
    )
    assert_removed(
        "Pasal 7\nSK No 205872 A\n-4-\nPresiden dan Wakil Presiden...",
        "-4-",
    )
    assert_removed(
        "Pasal 7\nPRESIDEN\nREPUBLIK INDONESIA\nisi pasal",
        "PRESIDEN",
    )

    # Critical regression: legal prose beginning with "Presiden" MUST remain.
    assert_preserved(
        "Pasal 6\nPresiden dengan persetujuan Dewan Perwakilan Rakyat "
        "menyatakan perang, membuat perdamaian dan perjanjian dengan negara lain.",
        "Presiden dengan persetujuan Dewan Perwakilan Rakyat",
    )

    assert_preserved(
        "Pasal 7\nPresiden dan Wakil Presiden memegang jabatannya selama masa "
        "lima tahun, dan sesudahnya dapat dipilih kembali.",
        "Presiden dan Wakil Presiden memegang jabatannya selama masa lima tahun",
    )

    # Legal prose containing Republik Indonesia MUST remain.
    assert_preserved(
        "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945;",
        "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945;",
    )

    # Perpres reference must remain.
    assert_preserved(
        "Peraturan Presiden Nomor 5 Tahun 2015 tentang Penyelenggaraan "
        "Sistem Administrasi Manunggal Satu Atap Kendaraan Bermotor;",
        "Peraturan Presiden Nomor 5 Tahun 2015",
    )

    print("ALL TESTS: PASS")


if __name__ == "__main__":
    main()
