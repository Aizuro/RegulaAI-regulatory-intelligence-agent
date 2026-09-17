"""FastAPI service for Regulatory Intelligence Agent V7.1.

V7.1 preserves the V6.5 / V6.6 query contract and adds a monitoring
ingestion/analysis contract for future n8n automation.

Run from the project root:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator

from src.embeddings import load_legal_vectorstore
from src.agent_orchestrator import build_agent
from src.monitoring import build_monitoring_contract, detect_changes
from src.monitoring_state import MonitoringStateStore
from src.monitoring_analysis import analyze_monitoring_event


app = FastAPI(
    title="Regulatory Intelligence Agent",
    description="API for Indonesian regulatory intelligence and analysis.",
    version="7.1.0",
)

_vectorstore = None
_agent = None


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Pertanyaan pengguna.",
        examples=["Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?"],
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    plan: List[str] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)
    missing_regulations: List[Dict[str, Any]] = Field(default_factory=list)
    comparison: Dict[str, Any] | None = None
    sql_results: List[Dict[str, Any]] = Field(default_factory=list)
    external_sources: List[Dict[str, Any]] = Field(default_factory=list)
    external_citations: List[Dict[str, Any]] = Field(default_factory=list)


class MonitorRegulation(BaseModel):
    type: str = Field(..., min_length=1, examples=["Perpres"])
    number: str = Field(..., min_length=1, examples=["95"])
    year: str = Field(..., min_length=4, examples=["2018"])
    title: str = Field(
        ...,
        min_length=1,
        examples=["Sistem Pemerintahan Berbasis Elektronik"],
    )
    url: AnyHttpUrl = Field(
        ...,
        examples=["https://jdih.kemenkeu.go.id/dok/perpres-95-tahun-2018/overview"],
    )
    source: str = Field(..., min_length=1, examples=["JDIH"])
    published_date: str | None = Field(
        default=None,
        examples=["2018-10-05"],
    )

    @field_validator("url")
    @classmethod
    def validate_https_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https":
            raise ValueError("url harus menggunakan HTTPS.")
        return value


class MonitorAnalyzeRequest(BaseModel):
    regulation: MonitorRegulation
    previous_regulation: MonitorRegulation | None = None
    event: Literal[
        "new_regulation",
        "updated_regulation",
        "removed_regulation",
    ] = "new_regulation"


class MonitorAnalyzeResponse(BaseModel):
    status: Literal["ok"]
    relevance: Literal["unknown", "low", "medium", "high"]
    summary: str
    key_points: List[str]
    should_notify: bool
    regulation_key: str
    event: str
    impact: str


class MonitorSnapshotRequest(BaseModel):
    regulations: List[MonitorRegulation] = Field(default_factory=list)


class MonitorChangeEvent(BaseModel):
    regulation_key: str
    event: Literal[
        "new_regulation",
        "updated_regulation",
        "removed_regulation",
    ]
    regulation: Dict[str, Any]
    previous_regulation: Optional[Dict[str, Any]] = None
    fingerprint: str


class MonitorSnapshotResponse(BaseModel):
    status: Literal["ok"]
    previous_count: int
    current_count: int
    changed: bool
    events: List[MonitorChangeEvent] = Field(default_factory=list)


def get_monitoring_state_store() -> MonitoringStateStore:
    """Return the persistent monitoring store used by the API."""
    database_path = os.getenv(
        "MONITORING_STATE_DB",
        str(Path(__file__).resolve().with_name("monitoring_state.sqlite3")),
    )
    return MonitoringStateStore(database_path)


def serialize_source(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        return doc

    metadata = getattr(doc, "metadata", {}) or {}
    page_content = getattr(doc, "page_content", "")

    return {
        "content": str(page_content),
        "metadata": metadata,
    }


def get_agent():
    """Initialize the V6.5 agent lazily."""
    global _vectorstore, _agent

    if _agent is None:
        _vectorstore = load_legal_vectorstore()
        _agent = build_agent(_vectorstore)

    return _agent


QueryResponse.model_rebuild()


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "regulatory-intelligence-agent",
        "version": "7.1.0",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run a question through the Regulatory Intelligence Agent."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question tidak boleh kosong.",
        )

    try:
        result = get_agent().invoke(question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {type(exc).__name__}: {exc}",
        ) from exc

    return QueryResponse(
        answer=str(result.get("answer", "")),
        sources=[
            serialize_source(doc)
            for doc in result.get("sources", [])
        ],
        citations=result.get("citations", []),
        plan=result.get("plan", []),
        completed_steps=result.get("completed_steps", []),
        missing_regulations=result.get("missing_regulations", []),
        comparison=result.get("comparison"),
        sql_results=result.get("sql_results", []),
        external_sources=result.get("external_sources", []),
        external_citations=result.get("external_citations", []),
    )


@app.post(
    "/monitor/sync",
    response_model=MonitorSnapshotResponse,
)
def monitor_sync(
    request: MonitorSnapshotRequest,
) -> MonitorSnapshotResponse:
    """Analyze an incoming regulatory monitoring event using the LLM."""

    current = [
        regulation.model_dump(mode="json")
        for regulation in request.regulations
    ]
    store = get_monitoring_state_store()
    previous = store.load_snapshot()

    try:
        events = detect_changes(previous, current)
        store.save_snapshot(current)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid monitoring snapshot: {type(exc).__name__}: {exc}",
        ) from exc

    return MonitorSnapshotResponse(
        status="ok",
        previous_count=len(previous),
        current_count=len(current),
        changed=bool(events),
        events=events,
    )


@app.post(
    "/monitor/analyze",
    response_model=MonitorAnalyzeResponse,
)
def monitor_analyze(
    request: MonitorAnalyzeRequest,
) -> MonitorAnalyzeResponse:
    """Validate and normalize an incoming regulatory monitoring event.

    V7.1 intentionally does not call the LLM yet. The endpoint establishes
    the stable contract that n8n will call in later V7 stages.
    """
    try:
        contract = build_monitoring_contract(
            request.regulation.model_dump(mode="json"),
            request.event,
        )
        analysis = analyze_monitoring_event(
            event=request.event,
            regulation=request.regulation.model_dump(mode="json"),
            previous_regulation=(
                request.previous_regulation.model_dump(mode="json")
                if request.previous_regulation
                else None
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid monitoring event: {type(exc).__name__}: {exc}",
        ) from exc

    return MonitorAnalyzeResponse(
        status="ok",
        relevance=analysis.relevance,
        summary=analysis.summary,
        key_points=analysis.key_points,
        impact=analysis.impact,
        should_notify=analysis.should_notify,
        regulation_key=contract["regulation_key"],
        event=contract["event"],
    )
