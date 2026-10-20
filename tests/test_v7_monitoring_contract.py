"""V7.1-compatible monitoring API contract tests.

The analysis function is mocked so these tests do not call Groq or depend on
external LLM availability.
"""
from fastapi.testclient import TestClient

import src.api as api
from src.api import app
from src.monitoring_analysis import MonitoringAnalysis

client = TestClient(app)


def _mock_analysis(**kwargs):
    return MonitoringAnalysis(
        relevance="unknown",
        summary="Informasi yang tersedia hanya metadata regulasi; isi dan dampaknya belum dapat ditentukan.",
        key_points=[],
        impact="Dampak belum dapat ditentukan tanpa informasi isi regulasi.",
        should_notify=False,
    )


def test_health_reports_v71():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "7.1.0"


def test_monitor_analyze_accepts_new_regulation(monkeypatch):
    monkeypatch.setattr(api, "analyze_monitoring_event", _mock_analysis)
    payload = {
        "regulation": {
            "type": "Perpres",
            "number": "95",
            "year": "2018",
            "title": "Sistem Pemerintahan Berbasis Elektronik",
            "url": "https://jdih.kemenkeu.go.id/dok/perpres-95-tahun-2018/overview",
            "source": "JDIH",
            "published_date": "2018-10-05",
        },
        "event": "new_regulation",
    }
    response = client.post("/monitor/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["relevance"] == "unknown"
    assert data["summary"]
    assert data["key_points"] == []
    assert data["should_notify"] is False
    assert data["regulation_key"] == "PERPRES-95-2018"
    assert data["event"] == "new_regulation"


def test_monitor_analyze_accepts_updated_regulation(monkeypatch):
    monkeypatch.setattr(api, "analyze_monitoring_event", _mock_analysis)
    payload = {
        "regulation": {
            "type": "UU",
            "number": "1",
            "year": "2024",
            "title": "Contoh regulasi",
            "url": "https://example.com/regulation",
            "source": "JDIH",
        },
        "previous_regulation": {
            "type": "UU",
            "number": "1",
            "year": "2024",
            "title": "Contoh regulasi versi sebelumnya",
            "url": "https://example.com/regulation",
            "source": "JDIH",
        },
        "event": "updated_regulation",
    }
    response = client.post("/monitor/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["regulation_key"] == "UU-1-2024"
    assert data["event"] == "updated_regulation"


def test_monitor_analyze_rejects_invalid_event():
    payload = {
        "regulation": {
            "type": "Perpres",
            "number": "95",
            "year": "2018",
            "title": "Sistem Pemerintahan Berbasis Elektronik",
            "url": "https://example.com/regulation",
            "source": "JDIH",
        },
        "event": "something_else",
    }
    response = client.post("/monitor/analyze", json=payload)
    assert response.status_code == 422


def test_monitor_analyze_rejects_non_https_url():
    payload = {
        "regulation": {
            "type": "Perpres",
            "number": "95",
            "year": "2018",
            "title": "Sistem Pemerintahan Berbasis Elektronik",
            "url": "http://example.com/regulation",
            "source": "JDIH",
        },
        "event": "new_regulation",
    }
    response = client.post("/monitor/analyze", json=payload)
    assert response.status_code == 422
