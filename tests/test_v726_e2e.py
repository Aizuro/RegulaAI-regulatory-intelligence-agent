"""V7.2.6 end-to-end monitoring tests.

These tests validate the complete monitoring path represented by the n8n
V7.2.5 workflow: snapshot payload -> FastAPI /monitor/sync -> persistence ->
change events.

The n8n workflow itself is also checked as an exported workflow contract.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / "n8n" / "v7.2.5-monitoring-sync.json"


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


def workflow_payload():
    """Return the same snapshot shape produced by the n8n workflow."""
    return {"regulations": [regulation()]}


def test_n8n_workflow_export_has_expected_monitoring_contract():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    assert workflow["name"] == "Regula V7.2.5 - Monitoring Sync"

    node_names = {node["name"] for node in workflow["nodes"]}
    assert node_names == {
        "Manual Trigger",
        "Build Current Snapshot",
        "Call Regula /monitor/sync",
    }

    http_node = next(
        node
        for node in workflow["nodes"]
        if node["name"] == "Call Regula /monitor/sync"
    )

    assert http_node["parameters"]["method"] == "POST"
    assert http_node["parameters"]["url"] == (
        "http://host.docker.internal:8000/monitor/sync"
    )


def test_e2e_new_snapshot_flows_from_n8n_payload_to_api_and_persistence(
    tmp_path, monkeypatch
):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    response = client.post("/monitor/sync", json=workflow_payload())

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["previous_count"] == 0
    assert data["current_count"] == 1
    assert data["changed"] is True
    assert data["events"][0]["regulation_key"] == "PERPRES-95-2018"
    assert data["events"][0]["event"] == "new_regulation"


def test_e2e_second_identical_snapshot_is_unchanged(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    payload = workflow_payload()
    first = client.post("/monitor/sync", json=payload)
    assert first.status_code == 200

    second = client.post("/monitor/sync", json=payload)

    assert second.status_code == 200
    data = second.json()
    assert data["changed"] is False
    assert data["events"] == []


def test_e2e_changed_snapshot_produces_updated_event(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    first = client.post("/monitor/sync", json=workflow_payload())
    assert first.status_code == 200

    changed_payload = {
        "regulations": [
            regulation(
                title="Sistem Pemerintahan Berbasis Elektronik - Perubahan"
            )
        ]
    }
    response = client.post("/monitor/sync", json=changed_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["changed"] is True
    assert data["events"][0]["event"] == "updated_regulation"


def test_e2e_empty_snapshot_produces_removed_event(tmp_path, monkeypatch):
    database = tmp_path / "monitoring.sqlite3"
    monkeypatch.setenv("MONITORING_STATE_DB", str(database))

    first = client.post("/monitor/sync", json=workflow_payload())
    assert first.status_code == 200

    response = client.post("/monitor/sync", json={"regulations": []})

    assert response.status_code == 200
    data = response.json()
    assert data["changed"] is True
    assert data["events"][0]["event"] == "removed_regulation"
    assert data["events"][0]["regulation_key"] == "PERPRES-95-2018"
