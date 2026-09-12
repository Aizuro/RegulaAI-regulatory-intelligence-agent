"""FastAPI service for Regulatory Intelligence Agent V6.5.

V6.5 exposes verified external regulatory sources and citations while
preserving the V5 response contract.

Run from the project root:
    uvicorn src.api:app --reload
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.embeddings import load_legal_vectorstore
from src.agent_orchestrator import build_agent


app = FastAPI(
    title="Regulatory Intelligence Agent",
    description="API for Indonesian regulatory intelligence and analysis.",
    version="6.5.0",
)

logger = logging.getLogger(__name__)
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
        "version": "6.5.0",
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
        logger.exception("Agent execution failed")
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
