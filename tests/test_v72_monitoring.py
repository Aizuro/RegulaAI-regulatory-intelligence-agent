"""V7.2 monitoring change-detection and state tests."""

from src.monitoring import (
    build_regulation_fingerprint,
    build_regulation_key,
    detect_changes,
)
from src.monitoring_state import MonitoringStateStore


BASE = {
    "type": "Perpres",
    "number": "95",
    "year": "2018",
    "title": "Sistem Pemerintahan Berbasis Elektronik",
    "url": "https://example.com/perpres-95-2018",
    "source": "JDIH",
    "published_date": "2018-10-05",
}


def test_regulation_key_is_stable():
    assert build_regulation_key(" perpres ", "95", 2018) == "PERPRES-95-2018"


def test_fingerprint_is_deterministic():
    assert build_regulation_fingerprint(
        BASE) == build_regulation_fingerprint(dict(BASE))


def test_detects_new_regulation():
    events = detect_changes([], [BASE])
    assert [(event["regulation_key"], event["event"]) for event in events] == [
        ("PERPRES-95-2018", "new_regulation")
    ]


def test_detects_updated_regulation():
    updated = {
        **BASE, "title": "Sistem Pemerintahan Berbasis Elektronik - Perubahan"}
    events = detect_changes([BASE], [updated])
    assert events[0]["event"] == "updated_regulation"


def test_unchanged_regulation_produces_no_event():
    assert detect_changes([BASE], [dict(BASE)]) == []


def test_detects_removed_regulation():
    events = detect_changes([BASE], [])
    assert events[0]["event"] == "removed_regulation"
    assert events[0]["regulation_key"] == "PERPRES-95-2018"


def test_duplicate_keys_are_rejected():
    duplicate = {**BASE, "title": "Duplicate"}
    try:
        detect_changes([], [BASE, duplicate])
    except ValueError as exc:
        assert "Duplicate regulation_key" in str(exc)
    else:
        raise AssertionError(
            "Expected duplicate regulation_key to be rejected")


def test_state_persists_snapshot(tmp_path):
    database = tmp_path / "monitoring.sqlite3"
    store = MonitoringStateStore(database)
    store.save_snapshot([BASE])

    reloaded = MonitoringStateStore(database)
    assert reloaded.load_snapshot() == [BASE]


def test_state_replaces_previous_snapshot_atomically(tmp_path):
    database = tmp_path / "monitoring.sqlite3"
    store = MonitoringStateStore(database)
    store.save_snapshot([BASE])

    updated = {**BASE, "number": "96"}
    store.save_snapshot([updated])

    assert store.load_snapshot() == [updated]
