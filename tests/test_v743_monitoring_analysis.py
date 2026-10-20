"""V7.4.3 tests for structured regulatory monitoring AI analysis."""

import pytest

from src.monitoring_analysis import MonitoringAnalysis, analyze_monitoring_event


REGULATION = {
    "type": "KMK",
    "number": "42/MK/EF.2",
    "year": "2026",
    "title": "Nilai Kurs Sebagai Dasar Pelunasan Bea Masuk",
    "url": "https://jdih.kemenkeu.go.id/dok/kmk-42mkef22026",
    "source": "JDIH Kementerian Keuangan",
    "published_date": None,
}

PREVIOUS = {
    **REGULATION,
    "title": "Nilai Kurs Sebagai Dasar Pelunasan Bea Masuk Versi Sebelumnya",
}


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return self.result


class FakeLLM:
    def __init__(self, result):
        self.structured = FakeStructuredModel(result)

    def with_structured_output(self, schema):
        assert schema is MonitoringAnalysis
        return self.structured


def test_new_event_returns_structured_analysis():
    fake = FakeLLM(
        {
            "relevance": "medium",
            "summary": "Regulasi baru terdeteksi dari sumber JDIH.",
            "key_points": ["Event adalah regulasi baru."],
            "impact": "Perlu ditinjau lebih lanjut sesuai kebutuhan organisasi.",
            "should_notify": True,
        }
    )

    result = analyze_monitoring_event(
        "new_regulation",
        REGULATION,
        llm=fake,
    )

    assert result.relevance == "medium"
    assert result.should_notify is True
    assert "Versi Sebelumnya" not in fake.structured.last_prompt
    assert REGULATION["title"] in fake.structured.last_prompt


def test_updated_event_prompt_contains_previous_and_current():
    fake = FakeLLM(
        {
            "relevance": "high",
            "summary": "Terjadi perubahan pada informasi regulasi.",
            "key_points": ["Judul regulasi berubah."],
            "impact": "Perubahan perlu ditinjau untuk menentukan dampaknya.",
            "should_notify": True,
        }
    )

    result = analyze_monitoring_event(
        "updated_regulation",
        REGULATION,
        previous_regulation=PREVIOUS,
        llm=fake,
    )

    assert result.relevance == "high"
    assert "previous_regulation" in fake.structured.last_prompt
    assert "Versi Sebelumnya" in fake.structured.last_prompt
    assert "Nilai Kurs Sebagai Dasar Pelunasan Bea Masuk\"" in fake.structured.last_prompt


def test_updated_event_requires_previous_regulation():
    with pytest.raises(ValueError, match="previous_regulation"):
        analyze_monitoring_event(
            "updated_regulation",
            REGULATION,
            llm=FakeLLM({}),
        )
