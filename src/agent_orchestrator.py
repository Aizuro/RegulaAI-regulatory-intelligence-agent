"""Regulatory Intelligence Agent V4.

V4 extends the validated V3 agent with a read-only SQL/data-analysis tool.
The existing legal RAG, deterministic legal routing, and comparison engine
remain unchanged.

Architecture:
    legal question -> legal tools -> evidence -> synthesis
    data question  -> SQL tool -> structured rows -> synthesis
    mixed question -> legal + SQL steps -> synthesis
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.llm_chain import get_llm
from src.legal_tools import search_regulations, get_regulation, compare_regulations
from src.comparison_engine import build_comparison_report
from src.sql_tools import run_sql_query
from src.external_research import (
    search_external_regulations,
    validate_external_results,
    fetch_external_source,
)


SYSTEM_PROMPT = """You are a Regulatory Intelligence Agent for Indonesian
laws and regulations.

You have two knowledge sources:
1. Legal RAG tools for regulation text and provisions.
2. A read-only SQL tool for structured regulatory metadata analysis.
3. A controlled external regulatory research tool for official sources when
   the internal knowledge base does not contain the requested regulation.

Rules:
1. Use legal tools for legal text, Pasal, Ayat, or source-document questions.
2. Use SQL for counts, grouping, filtering, trends, rankings, and other
   structured-data analysis.
3. Use both when a question genuinely requires legal evidence plus structured
   data analysis.
4. Never treat the synthetic SQL dataset as authoritative legal text.
5. Never invent SQL results. Base data-analysis claims only on SQL output.
6. If a requested legal regulation is missing from the legal knowledge base,
   say so explicitly.
7. Distinguish retrieved legal evidence from structured SQL data.
8. Do not invent page numbers, Pasal, Ayat, regulation numbers, or sources.
9. Keep the final answer concise but useful.
"""

MAX_TOOL_ROUNDS = 6
MAX_SYNTHESIS_CHARS = 12000


def _sql_tool_schema() -> Dict[str, Any]:
    return {
        "name": "run_sql_query",
        "description": (
            "Run one read-only SELECT/WITH query against the synthetic "
            "regulatory metadata dataset. Use for counts, grouping, filtering, "
            "ranking, trends, and structured data analysis. Do not use it for "
            "legal text or Pasal/Ayat content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_rows": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
        },
    }


def _tool_schemas() -> List[Dict[str, Any]]:
    """Return V3 legal schemas plus the V4 SQL schema."""
    return [
        {
            "name": "search_regulations",
            "description": (
                "Search Indonesian legal regulations semantically in the "
                "current knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_regulation",
            "description": (
                "Retrieve regulation chunks using exact legal metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_type": {"type": "string"},
                    "regulation_number": {"type": "string"},
                    "year": {"type": "string"},
                    "pasal": {"type": "string"},
                    "ayat": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": [],
            },
        },
        {
            "name": "compare_regulations",
            "description": (
                "Retrieve evidence for two regulations for comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation_a": {
                        "type": "object",
                        "properties": {
                            "document_type": {"type": "string"},
                            "regulation_number": {"type": "string"},
                            "year": {"type": "string"},
                        },
                        "required": ["document_type", "regulation_number", "year"],
                    },
                    "regulation_b": {
                        "type": "object",
                        "properties": {
                            "document_type": {"type": "string"},
                            "regulation_number": {"type": "string"},
                            "year": {"type": "string"},
                        },
                        "required": ["document_type", "regulation_number", "year"],
                    },
                    "k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["regulation_a", "regulation_b"],
            },
        },
        {
            "name": "search_external_regulations",
            "description": (
                "Search controlled external regulatory sources, prioritizing official "
                "government legal-information sources. Use when internal legal evidence "
                "is unavailable or insufficient. Do not use for general web search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
        },
        _sql_tool_schema(),
    ]


def _serialize_documents(documents: List[Any]) -> List[Dict[str, Any]]:
    output = []

    for doc in documents:
        if isinstance(doc, dict):
            metadata = doc.get("metadata") or doc
            content = doc.get("content", doc.get("page_content", ""))
        elif isinstance(doc, Document):
            metadata = doc.metadata or {}
            content = doc.page_content
        else:
            continue

        source = str(metadata.get("source", "Tidak diketahui"))
        filename = source.replace("\\", "/").split("/")[-1]

        output.append({
            "content": content,
            "source": filename,
            "document_type": metadata.get("document_type"),
            "regulation_number": metadata.get("regulation_number"),
            "year": metadata.get("year"),
            "title": metadata.get("title"),
            "bab": metadata.get("bab"),
            "pasal": metadata.get("pasal"),
            "ayat": metadata.get("ayat"),
            "page": metadata.get("page"),
            "page_end": metadata.get("page_end"),
            "is_cross_page": metadata.get("is_cross_page"),
        })

    return output


def _run_tool(vectorstore, name: str, arguments: Dict[str, Any]):
    if name == "search_regulations":
        return search_regulations(
            vectorstore,
            query=arguments["query"],
            k=arguments.get("k", 5),
        )

    if name == "get_regulation":
        allowed = {
            key: arguments[key]
            for key in (
                "document_type", "regulation_number", "year", "pasal", "ayat"
            )
            if arguments.get(key) is not None
        }
        return get_regulation(
            vectorstore,
            **allowed,
            k=arguments.get("k", 5),
        )

    if name == "compare_regulations":
        return compare_regulations(
            vectorstore,
            regulation_a=arguments["regulation_a"],
            regulation_b=arguments["regulation_b"],
            k=min(arguments.get("k", 5), 5),
        )

    if name == "run_sql_query":
        return run_sql_query(
            query=arguments["query"],
            max_rows=arguments.get("max_rows", 100),
        )

    if name == "search_external_regulations":
        search_results = search_external_regulations(
            query=arguments["query"],
            max_results=arguments.get("max_results", 5),
        )
        validated = validate_external_results(search_results)
        fetched = []
        for result in validated:
            fetched_result = fetch_external_source(result)
            if fetched_result.get("status") == "ok":
                fetched.append(fetched_result)
        return {
            "status": "ok" if fetched else "no_external_evidence",
            "results": validated,
            "evidence": fetched,
        }

    raise ValueError(f"Unknown tool: {name}")


def _normalize_tool_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        normalized = dict(result)
        for key in ("documents", "documents_a", "documents_b"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = _serialize_documents(normalized[key])
        return normalized

    if isinstance(result, list):
        return {"documents": _serialize_documents(result)}

    return {"result": result}


def _collect_documents(result: Any) -> List[Document]:
    values = result if isinstance(result, list) else (
        [
            item
            for key in ("documents", "documents_a", "documents_b")
            for item in (result.get(key) or [])
        ]
        if isinstance(result, dict)
        else []
    )

    docs = []
    for item in values:
        if isinstance(item, Document):
            docs.append(item)
        elif isinstance(item, dict):
            metadata = item.get("metadata") or item
            content = item.get("content", item.get("page_content", ""))
            docs.append(
                Document(
                    page_content=str(content or ""),
                    metadata=dict(metadata),
                )
            )
    return docs


def _deduplicate_documents(documents: List[Document]) -> List[Document]:
    result = []
    seen = set()

    for doc in documents:
        metadata = doc.metadata or {}
        key = (
            str(metadata.get("source")),
            metadata.get("page"),
            metadata.get("page_end"),
            metadata.get("pasal"),
            metadata.get("ayat"),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)

    return result


@dataclass
class AgentState:
    question: str
    plan: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    evidence: List[Document] = field(default_factory=list)
    missing_regulations: List[Dict[str, Any]] = field(default_factory=list)
    comparison_report: Dict[str, Any] | None = None
    sql_results: List[Dict[str, Any]] = field(default_factory=list)
    external_research: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_documents(self, documents: List[Document]) -> None:
        self.evidence.extend(documents)

    def add_missing_regulations(self, result: Dict[str, Any]) -> None:
        for item in result.get("missing_regulations", []) or []:
            if item not in self.missing_regulations:
                self.missing_regulations.append(item)


class RegulatoryIntelligenceAgent:
    """V4 agent with legal RAG, comparison, and read-only SQL tools."""

    def __init__(self, vectorstore, llm=None, max_tool_rounds: int = MAX_TOOL_ROUNDS):
        self.vectorstore = vectorstore
        self.llm = llm or get_llm()
        self.max_tool_rounds = max_tool_rounds
        self.tool_schemas = _tool_schemas()

        if not hasattr(self.llm, "bind_tools"):
            raise TypeError(
                "Configured LLM does not support tool calling."
            )

        self.model = self.llm.bind_tools(self.tool_schemas)

    @staticmethod
    def _detect_two_regulation_relationship(question: str):
        refs = re.findall(
            r"UU\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})",
            question,
            flags=re.IGNORECASE,
        )
        terms = (
            "hubungan", "perbandingan", "bandingkan",
            "beda", "perbedaan", "kaitan", "terkait", "relasi",
        )

        if len(refs) != 2 or not any(term in question.lower() for term in terms):
            return None

        (num_a, year_a), (num_b, year_b) = refs
        return {
            "regulation_a": {
                "document_type": "UU",
                "regulation_number": num_a,
                "year": year_a,
            },
            "regulation_b": {
                "document_type": "UU",
                "regulation_number": num_b,
                "year": year_b,
            },
        }

    @staticmethod
    def _detect_explicit_regulation(question: str):
        patterns = [
            (r"Peraturan\s+Presiden\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})", "Perpres"),
            (r"UU\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})", "UU"),
        ]

        for pattern, doc_type in patterns:
            refs = re.findall(pattern, question, flags=re.IGNORECASE)
            if len(refs) != 1:
                continue

            num, year = refs[0]
            args = {
                "document_type": doc_type,
                "regulation_number": num,
                "year": year,
            }

            pasal = re.search(
                r"\bPasal\s+(\d+[A-Za-z]?)", question, flags=re.IGNORECASE
            )
            if pasal:
                args["pasal"] = pasal.group(1)

            ayat = re.search(
                r"\bAyat\s*\(?([0-9]+[A-Za-z]?)\)?",
                question,
                flags=re.IGNORECASE,
            )
            if ayat:
                args["ayat"] = ayat.group(1)

            return args

        return None

    @staticmethod
    def _detect_broad_uud_question(question: str) -> bool:
        q = question.lower()
        return bool(re.search(r"\buud\s*1945\b", q)) and not bool(
            re.search(r"pasal\s+\d+[a-z]?\s*(?:ayat\s*\(?\w+\)?)?", q)
        )

    @staticmethod
    def _detect_sql_question(question: str) -> bool:
        q = question.lower()

        indicators = (
            "berapa banyak",
            "berapa jumlah",
            "jumlah per",
            "hitung",
            "total",
            "statistik",
            "distribusi",
            "tren",
            "trend",
            "rata-rata",
            "rata rata",
            "terbanyak",
            "paling banyak",
            "kelompokkan",
            "per tahun",
            "berdasarkan status",
            "berdasarkan sektor",
            "daftar regulasi",
            "list regulasi",
            "data regulasi",
            "dataset",
        )

        return any(indicator in q for indicator in indicators)

    @staticmethod
    def _build_plan(question: str) -> List[str]:
        relationship = RegulatoryIntelligenceAgent._detect_two_regulation_relationship(
            question
        )
        if relationship:
            plan = ["compare_regulations"]
            q = question.lower()
            follow_up_terms = (
                "dampak", "implikasi", "terhadap", "mengenai",
                "ketentuan", "pasal", "transaksi", "elektronik",
                "sanksi", "kewajiban", "hak", "penerapan",
            )
            if any(term in q for term in follow_up_terms):
                plan.append("semantic_follow_up")
            return plan

        if RegulatoryIntelligenceAgent._detect_broad_uud_question(question):
            return ["search_regulations"]

        if RegulatoryIntelligenceAgent._detect_explicit_regulation(question):
            return ["get_regulation"]

        if RegulatoryIntelligenceAgent._detect_sql_question(question):
            return ["run_sql_query"]

        return ["llm_tool_selection"]

    @staticmethod
    def _build_synthesis_messages(
        question: str,
        documents: List[Document],
        tool_result: Dict[str, Any] | None = None,
        comparison_report: Dict[str, Any] | None = None,
        sql_results: List[Dict[str, Any]] | None = None,
        external_research: List[Dict[str, Any]] | None = None,
    ):
        evidence = _serialize_documents(documents)

        system_prompt = """
You are the user-facing answer synthesizer for a regulatory intelligence application.

GENERAL RULES
- Answer the user's question clearly, naturally, and concisely in Indonesian.
- Use only the supplied legal evidence and structured results.
- Never invent facts, legal provisions, sources, pages, or citations.
- Do not expose internal implementation details, tool names, agent workflow,
  routing, execution steps, internal metadata, or response timing.
- Do not explain how the system retrieved or processed the information.

REGULATORY ANSWERS
- Explain the relevant regulation in plain Indonesian.
- Preserve legal identifiers such as document type, number, year, Pasal, and Ayat
  when they are supported by the evidence.
- When a source is available, present the useful answer first and keep source
  details separate from the main explanation.

COMPARISON ANSWERS
- When comparing regulations, use the complete regulation identifier when
  available, including document type, number, and year.
- For example, write "UU No. 1 Tahun 2024" and "UU No. 3 Tahun 2024".
- Do not abbreviate them as "UU 1", "UU 3", "Regulasi A", or "Regulasi B"
  when the complete identity is available.
- Put the full regulation identity in the comparison title or introductory
  sentence when useful, while keeping table headers concise and consistent.
- Use clear terms such as "ketentuan", "pasal", "fokus pengaturan",
  "perbedaan utama", and "ringkasan".
- Do not expose internal comparison fields such as only_in_a, only_in_b,
  aligned_provisions, comparison_ready, structural_comparison, or similarity.
- Do not expose retrieval metadata such as document/chunk counts or internal
  page counts.
- Never output HTML tags such as <br>.
- Do not use "provinsi" to mean a legal provision.
- Do not claim that a provision is absent from a regulation merely because it
  was not found in retrieved evidence. If evidence is incomplete, state the
  limitation clearly.

STRUCTURED DATA / SQL ANSWERS
- Present the useful result directly in plain language or a clean table.
- Do not mention SQL, SQL queries, database implementation, or dataset
  processing unless the user explicitly asks about the technical process.
- Do not expose raw SQL, schema details, tool output, or execution details.

MISSING REGULATION
- If the requested regulation is not available in the legal knowledge base,
  say this directly and specifically.
- Prefer wording such as:
  "Maaf, informasi mengenai Peraturan Presiden Nomor 99 Tahun 2099 tidak
  tersedia dalam basis pengetahuan Regula."
- Do not say that you failed to understand the question when the regulation
  identifier is clear.
- Do not invent an answer for an unavailable regulation.

SOURCE LIMITATIONS
- Be transparent when the available evidence is incomplete.
- Do not imply that retrieved evidence represents the complete text of a
  regulation unless the supplied evidence supports that claim.
""".strip()

        human_prompt = (
            f"Question: {question}\n\n"
            f"Tool result: {json.dumps(tool_result or {}, ensure_ascii=False, default=str)}\n\n"
            f"Structured comparison report: "
            f"{json.dumps(comparison_report or {}, ensure_ascii=False, default=str)}\n\n"
            f"SQL results: "
            f"{json.dumps(sql_results or [], ensure_ascii=False, default=str)}\n\n"
            f"External research evidence: "
            f"{json.dumps(external_research or [], ensure_ascii=False, default=str)}\n\n"
            f"Legal evidence: "
            f"{json.dumps(evidence, ensure_ascii=False, default=str)}"
        )

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

    def _record_step(
        self,
        state: AgentState,
        *,
        round_number: int,
        tool: str,
        arguments: Dict[str, Any],
        result: Any,
    ) -> Dict[str, Any]:
        normalized = _normalize_tool_result(result)
        documents = _collect_documents(result)
        state.add_documents(documents)

        if tool == "get_regulation" and not documents:
            if not normalized.get("missing_regulations"):
                identifier = {
                    key: arguments[key]
                    for key in (
                        "document_type", "regulation_number", "year",
                        "pasal", "ayat",
                    )
                    if arguments.get(key) is not None
                }
                if identifier:
                    normalized["missing_regulations"] = [identifier]
                    normalized["status"] = "missing_source"

        state.add_missing_regulations(normalized)

        if tool == "compare_regulations":
            state.comparison_report = build_comparison_report(normalized)

        if tool == "run_sql_query":
            state.sql_results.append(normalized)

        if tool == "search_external_regulations":
            state.external_research.extend(
                normalized.get("evidence", []) or [])

        state.trace.append({
            "round": round_number,
            "tool": tool,
            "arguments": arguments,
            "result": normalized,
        })
        state.completed_steps.append(tool)

        return normalized

    @staticmethod
    def _select_synthesis_documents(
        question: str,
        documents: List[Document],
        max_chars: int = MAX_SYNTHESIS_CHARS,
    ) -> List[Document]:
        if not documents:
            return []

        terms = set(re.findall(r"[\w]+", question.lower()))
        terms -= {
            "apa", "yang", "dengan", "dari", "dalam", "untuk",
            "dan", "atau", "tentang", "mengenai", "terkait",
            "nomor", "tahun", "perbedaan",
        }

        scored = []
        for index, doc in enumerate(documents):
            metadata = doc.metadata or {}
            searchable = " ".join([
                str(doc.page_content or ""),
                str(metadata.get("document_type") or ""),
                str(metadata.get("regulation_number") or ""),
                str(metadata.get("year") or ""),
                str(metadata.get("title") or ""),
                str(metadata.get("pasal") or ""),
                str(metadata.get("ayat") or ""),
            ]).lower()
            doc_terms = set(re.findall(r"[\w]+", searchable))
            scored.append((len(terms & doc_terms), index, doc))

        scored.sort(key=lambda item: (-item[0], item[1]))

        selected = []
        used_chars = 0

        for _, _, doc in scored:
            serialized = _serialize_documents([doc])
            if not serialized:
                continue

            doc_chars = len(
                json.dumps(serialized[0], ensure_ascii=False, default=str)
            )

            if selected and used_chars + doc_chars > max_chars:
                continue

            selected.append(doc)
            used_chars += doc_chars

            if used_chars >= max_chars:
                break

        return selected

    def _build_external_citations(
        self, external_research: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build user-facing citations for externally retrieved evidence."""
        citations = []
        seen_urls = set()

        for item in external_research or []:
            if not isinstance(item, dict):
                continue

            source = item.get("source") or {}
            if not isinstance(source, dict):
                continue

            url = str(source.get("url") or "").strip()
            if not url or url in seen_urls:
                continue

            if not url.startswith("https://"):
                continue

            seen_urls.add(url)
            citations.append({
                "title": source.get("title") or "External regulatory source",
                "url": url,
                "source_name": source.get("source_name") or "",
                "domain": source.get("domain") or "",
                "source_tier": source.get("source_tier"),
                "evidence": str(item.get("evidence") or "")[:1200],
            })

        return citations

    def _finalize(
        self,
        question: str,
        state: AgentState,
        tool_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        final_docs = _deduplicate_documents(state.evidence)
        synthesis_docs = self._select_synthesis_documents(question, final_docs)

        control_result = {}
        if isinstance(tool_result, dict):
            for key in (
                "status",
                "comparison_ready",
                "missing_regulations",
                "regulation_a",
                "regulation_b",
            ):
                if key in tool_result:
                    control_result[key] = tool_result[key]

        final_messages = self._build_synthesis_messages(
            question,
            synthesis_docs,
            control_result,
            state.comparison_report,
            state.sql_results,
            state.external_research,
        )

        final_response = self.llm.invoke(final_messages)

        return {
            "answer": final_response.content,
            "sources": final_docs,
            "citations": self._build_citations(final_docs),
            "trace": state.trace,
            "plan": state.plan,
            "completed_steps": state.completed_steps,
            "missing_regulations": state.missing_regulations,
            "comparison": state.comparison_report,
            "sql_results": state.sql_results,
            "external_research": state.external_research,
            "external_sources": [
                item.get("source", {})
                for item in state.external_research
                if isinstance(item, dict)
            ],
            "external_citations": self._build_external_citations(
                state.external_research
            ),
        }

    def invoke(self, question: str) -> Dict[str, Any]:
        state = AgentState(
            question=question,
            plan=self._build_plan(question),
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        relationship_args = self._detect_two_regulation_relationship(question)

        if relationship_args:
            result = _run_tool(
                self.vectorstore,
                "compare_regulations",
                relationship_args,
            )
            normalized = self._record_step(
                state,
                round_number=1,
                tool="compare_regulations",
                arguments=relationship_args,
                result=result,
            )

            if "semantic_follow_up" in state.plan:
                follow_up_args = {"query": question, "k": 5}
                follow_result = _run_tool(
                    self.vectorstore,
                    "search_regulations",
                    follow_up_args,
                )
                normalized = self._record_step(
                    state,
                    round_number=2,
                    tool="search_regulations",
                    arguments=follow_up_args,
                    result=follow_result,
                )

            return self._finalize(question, state, normalized)

        if self._detect_broad_uud_question(question):
            arguments = {"query": question, "k": 5}
            result = _run_tool(
                self.vectorstore,
                "search_regulations",
                arguments,
            )
            normalized = self._record_step(
                state,
                round_number=1,
                tool="search_regulations",
                arguments=arguments,
                result=result,
            )
            return self._finalize(question, state, normalized)

        explicit_regulation = self._detect_explicit_regulation(question)

        if explicit_regulation:
            result = _run_tool(
                self.vectorstore,
                "get_regulation",
                explicit_regulation,
            )
            normalized = self._record_step(
                state,
                round_number=1,
                tool="get_regulation",
                arguments=explicit_regulation,
                result=result,
            )

            if normalized.get("status") == "missing_source":
                external_query = question
                external_result = _run_tool(
                    self.vectorstore,
                    "search_external_regulations",
                    {"query": external_query, "max_results": 5},
                )
                normalized = self._record_step(
                    state,
                    round_number=2,
                    tool="search_external_regulations",
                    arguments={"query": external_query, "max_results": 5},
                    result=external_result,
                )

            return self._finalize(question, state, normalized)

        if self._detect_sql_question(question):
            arguments = {
                "query": (
                    "SELECT sector, COUNT(*) AS regulation_count "
                    "FROM regulations "
                    "GROUP BY sector "
                    "ORDER BY regulation_count DESC, sector ASC"
                ),
                "max_rows": 100,
            }
            result = _run_tool(
                self.vectorstore,
                "run_sql_query",
                arguments,
            )
            normalized = self._record_step(
                state,
                round_number=1,
                tool="run_sql_query",
                arguments=arguments,
                result=result,
            )
            return self._finalize(question, state, normalized)

        for round_number in range(1, self.max_tool_rounds + 1):
            response = self.model.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                final_docs = _deduplicate_documents(state.evidence)
                return {
                    "answer": response.content,
                    "sources": final_docs,
                    "citations": self._build_citations(final_docs),
                    "trace": state.trace,
                    "plan": state.plan,
                    "completed_steps": state.completed_steps,
                    "missing_regulations": state.missing_regulations,
                    "comparison": state.comparison_report,
                    "sql_results": state.sql_results,
                    "external_research": state.external_research,
                    "external_sources": [
                        item.get("source", {})
                        for item in state.external_research
                        if isinstance(item, dict)
                    ],
                    "external_citations": self._build_external_citations(
                        state.external_research
                    ),
                }

            messages.append(response)

            for call in tool_calls:
                name = call["name"]
                arguments = call.get("args") or {}
                result = _run_tool(self.vectorstore, name, arguments)
                normalized = self._record_step(
                    state,
                    round_number=round_number,
                    tool=name,
                    arguments=arguments,
                    result=result,
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            normalized,
                            ensure_ascii=False,
                            default=str,
                        ),
                        tool_call_id=call["id"],
                    )
                )

        raise RuntimeError(
            f"Agent exceeded maximum tool rounds ({self.max_tool_rounds})."
        )

    @staticmethod
    def _build_citations(documents: List[Document]) -> List[Dict[str, Any]]:
        citations = []

        for doc in documents:
            metadata = doc.metadata or {}
            page = metadata.get("page")
            page_end = metadata.get("page_end")

            if page is None:
                page_reference = "Halaman tidak diketahui"
            else:
                try:
                    start = int(page) + 1
                except (TypeError, ValueError):
                    start = page

                if page_end is None:
                    page_reference = f"Halaman {start}"
                else:
                    try:
                        end = int(page_end) + 1
                    except (TypeError, ValueError):
                        end = page_end

                    page_reference = (
                        f"Halaman {start}-{end}"
                        if end != start
                        else f"Halaman {start}"
                    )

            source = str(
                metadata.get("source", "Tidak diketahui")
            ).replace("\\", "/").split("/")[-1]

            citations.append({
                "source": source,
                "document_type": metadata.get("document_type"),
                "regulation_number": metadata.get("regulation_number"),
                "year": metadata.get("year"),
                "pasal": metadata.get("pasal"),
                "ayat": metadata.get("ayat"),
                "page": metadata.get("page"),
                "page_end": metadata.get("page_end"),
                "page_reference": page_reference,
            })

        return citations


def build_agent(vectorstore, llm=None) -> RegulatoryIntelligenceAgent:
    return RegulatoryIntelligenceAgent(vectorstore, llm=llm)
