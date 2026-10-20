"""V7.4.1 tests for previous/current regulation data in update events."""

from src.monitoring import detect_changes


def make_regulation(title: str) -> dict:
    return {
        "type": "PMK",
        "number": "64",
        "year": "2026",
        "title": title,
        "url": "https://jdih.kemenkeu.go.id/dok/pmk-64-tahun-2026",
        "source": "JDIH Kementerian Keuangan",
        "published_date": None,
    }


def test_updated_event_contains_previous_and_current_regulation():
    previous = [
        make_regulation(
            "Petunjuk Teknis Akuntansi Penerimaan Negara"
        )
    ]

    current = [
        make_regulation(
            "Petunjuk Teknis Akuntansi Penerimaan Negara dan Perubahan"
        )
    ]

    events = detect_changes(previous, current)

    assert len(events) == 1

    event = events[0]

    assert event["event"] == "updated_regulation"
    assert event["regulation"]["title"] == (
        "Petunjuk Teknis Akuntansi Penerimaan Negara dan Perubahan"
    )
    assert event["previous_regulation"]["title"] == (
        "Petunjuk Teknis Akuntansi Penerimaan Negara"
    )


def test_new_event_does_not_require_previous_regulation():
    previous = []

    current = [
        make_regulation(
            "Petunjuk Teknis Akuntansi Penerimaan Negara"
        )
    ]

    events = detect_changes(previous, current)

    assert len(events) == 1
    assert events[0]["event"] == "new_regulation"
    assert "previous_regulation" not in events[0]
