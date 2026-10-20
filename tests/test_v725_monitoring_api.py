"""V7.2.5 FastAPI integration tests for monitoring snapshot sync."""

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def regulation(number="95", title="Sistem Pemerintahan Berbasis Elektronik"):
    return {
        "type": "Perpres",
        "number": number,
        "year": "2018",
        "title": title,
        "url": f"https://example.com/perpres-{number}-2018",
        "source": "JDIH",
        "published_date": "2018-10-05",
    }


def test_monitor_sync_detects_new_and_persists_snapshot(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    response = client.post(
        "/monitor/sync",
        json={"regulations": [regulation()]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["previous_count"] == 0
    assert data["current_count"] == 1
    assert data["changed"] is True
    assert [(event["regulation_key"], event["event"]) for event in data["events"]] == [
        ("PERPRES-95-2018", "new_regulation")
    ]


def test_monitor_sync_detects_update_against_persisted_state(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    first = client.post("/monitor/sync", json={"regulations": [regulation()]})
    assert first.status_code == 200

    updated = regulation(title="Sistem Pemerintahan Berbasis Elektronik - Perubahan")
    response = client.post("/monitor/sync", json={"regulations": [updated]})

    assert response.status_code == 200
    data = response.json()
    assert data["previous_count"] == 1
    assert data["current_count"] == 1
    assert data["changed"] is True
    assert data["events"][0]["event"] == "updated_regulation"


def test_monitor_sync_returns_no_event_when_snapshot_is_unchanged(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    payload = {"regulations": [regulation()]}
    first = client.post("/monitor/sync", json=payload)
    assert first.status_code == 200

    response = client.post("/monitor/sync", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["previous_count"] == 1
    assert data["current_count"] == 1
    assert data["changed"] is False
    assert data["events"] == []


def test_monitor_sync_detects_removed_regulation(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    first = client.post("/monitor/sync", json={"regulations": [regulation()]})
    assert first.status_code == 200

    response = client.post("/monitor/sync", json={"regulations": []})

    assert response.status_code == 200
    data = response.json()
    assert data["previous_count"] == 1
    assert data["current_count"] == 0
    assert data["changed"] is True
    assert data["events"][0]["event"] == "removed_regulation"
    assert data["events"][0]["regulation_key"] == "PERPRES-95-2018"


def test_monitor_sync_rejects_duplicate_regulation_keys(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    duplicate = regulation(title="Duplicate")
    response = client.post(
        "/monitor/sync",
        json={"regulations": [regulation(), duplicate]},
    )

    assert response.status_code == 400
    assert "Duplicate regulation_key" in response.json()["detail"]
