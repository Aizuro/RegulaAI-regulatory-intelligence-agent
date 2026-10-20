"""V7.4.5 tests for grounding constraints in regulatory AI analysis."""

from src.monitoring_analysis import _build_prompt


REGULATION = {
    "type": "PMK",
    "number": "64",
    "year": "2026",
    "title": "Peraturan Menteri Keuangan Nomor 64 Tahun 2026",
    "url": "https://jdih.kemenkeu.go.id/dok/pmk-64-tahun-2026",
    "source": "JDIH Kementerian Keuangan",
    "published_date": None,
}


def test_new_event_prompt_does_not_assume_previous_snapshot():
    prompt = _build_prompt(
        event="new_regulation",
        regulation=REGULATION,
        previous_regulation=None,
    )

    assert '"event": "new_regulation"' in prompt
    assert '"previous_regulation"' not in prompt
    assert "Jangan menyebut adanya snapshot sebelumnya" in prompt
    assert "Jangan menambahkan fakta" in prompt
    assert REGULATION["title"] in prompt


def test_new_event_prompt_treats_missing_publication_date_as_unknown():
    prompt = _build_prompt(
        event="new_regulation",
        regulation=REGULATION,
        previous_regulation=None,
    )

    assert '"published_date": null' in prompt
    assert "published_date" in prompt
    assert "jangan menyebut tanggal" in prompt.lower()


def test_updated_event_prompt_contains_previous_and_current_data():
    previous = {
        **REGULATION,
        "title": "Peraturan Menteri Keuangan Nomor 64 Tahun 2026",
    }

    current = {
        **REGULATION,
        "title": "Peraturan Menteri Keuangan Nomor 64 Tahun 2026 dan Perubahan",
    }

    prompt = _build_prompt(
        event="updated_regulation",
        regulation=current,
        previous_regulation=previous,
    )

    assert '"event": "updated_regulation"' in prompt
    assert '"previous_regulation"' in prompt
    assert previous["title"] in prompt
    assert current["title"] in prompt


def test_removed_event_does_not_assume_legal_status():
    prompt = _build_prompt(
        event="removed_regulation",
        regulation={
            "type": "PMK",
            "number": "64",
            "year": "2026",
            "title": "Peraturan Menteri Keuangan Nomor 64 Tahun 2026",
            "url": "https://jdih.kemenkeu.go.id/dok/pmk-64-tahun-2026",
            "source": "JDIH Kementerian Keuangan",
            "published_date": "2026-08-21",
        },
        previous_regulation=None,
    )

    prompt_lower = prompt.lower()

    assert "tidak lagi muncul" in prompt_lower
    assert "jangan menyimpulkan bahwa regulasi sudah tidak berlaku secara hukum" in prompt_lower
