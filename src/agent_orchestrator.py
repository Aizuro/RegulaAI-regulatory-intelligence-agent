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

from llm_chain import get_llm
from legal_tools import search_regulations, get_regulation, compare_regulations
from comparison_engine import build_comparison_report
from sql_tools import run_sql_query
from v84_impact_tool import analyze_regulation_impact


SYSTEM_PROMPT = """You are a Regulatory Intelligence Agent for Indonesian
laws and regulations.

You have two knowledge sources:
1. Legal RAG tools for regulation text and provisions.
2. A read-only SQL tool for structured regulatory metadata analysis.

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
    """Return legal, SQL, and V8.4 impact-analysis tool schemas."""
    return [
        {
            "name": "analyze_regulation_impact",
            "description": (
                "Retrieve two regulations, compare their structural evidence, and "
                "produce a grounded impact analysis. Use when the user explicitly "
                "asks about the impact or implications of differences between two "
                "regulations."
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
                },
                "required": ["regulation_a", "regulation_b"],
            },
        },
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

    if name == "analyze_regulation_impact":
        return analyze_regulation_impact(
            regulation_a=arguments["regulation_a"],
            regulation_b=arguments["regulation_b"],
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
            q = question.lower()
            impact_terms = (
                "dampak", "implikasi", "pengaruh", "konsekuensi",
                "berpengaruh", "efek",
            )
            if any(term in q for term in impact_terms):
                return ["analyze_regulation_impact"]
            return ["compare_regulations"]

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
    ):
        evidence = _serialize_documents(documents)
        return [
            SystemMessage(content=(
                "You are the user-facing answer synthesizer for a regulatory "
                "intelligence application. Answer the user's question clearly and "
                "concisely using only the supplied evidence and structured results. "
                "Never invent facts. Do not expose internal implementation details, "
                "tool names, agent workflow, routing, execution steps, raw SQL, SQL "
                "queries, database/schema details, internal metadata, or response "
                "timing. Do not say that the answer was produced from an SQL query "
                "or describe how the system obtained the data. For structured-data "
                "questions, present the useful result directly as prose or a clean "
                "table when appropriate. The SQL dataset is synthetic metadata, "
                "not authoritative legal text. If a source is unavailable, say so "
                "briefly and clearly. When answering comparison questions, write for "
                "a non-technical user. Use natural Indonesian labels such as "
                "'Regulasi A' and 'Regulasi B'. Present only information that helps "
                "the user understand the substantive comparison. Do not expose "
                "retrieval metadata such as jumlah dokumen/chunks, page counts used "
                "internally, tool output, or comparison-engine statistics. Do not "
                "expose internal field names such as only_in_a, only_in_b, "
                "aligned_provisions, comparison_ready, structural_comparison, or "
                "similarity. Translate those concepts into plain language when they "
                "are useful, for example 'ketentuan yang ditemukan pada Regulasi A' "
                "or 'ketentuan yang ditemukan pada kedua sumber'. Do not present "
                "technical counts unless the user explicitly asks for them. Never "
                "output HTML tags such as <br>. Prefer concise Markdown lists or "
                "tables with readable Indonesian headings such as 'Fokus pengaturan', "
                "'Ketentuan yang relevan', 'Perbedaan utama', and 'Ringkasan'. Avoid "
                "the word 'provisi' when 'ketentuan' or 'pasal' is clearer. Never "
                "invent a similarity, thematic relationship, or legal conclusion "
                "that is not supported by the supplied evidence. Do not claim that "
                "a provision is absent from a regulation merely because it was not "
                "found in retrieved evidence; when evidence is incomplete, describe "
                "the limitation explicitly. In regulatory comparisons, do not use "
                "'provinsi' to mean a legal provision; use 'ketentuan' or 'pasal'."
            )),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Tool result: {json.dumps(tool_result or {}, ensure_ascii=False, default=str)}\n\n"
                f"Structured comparison report: "
                f"{json.dumps(comparison_report or {}, ensure_ascii=False, default=str)}\n\n"
                f"SQL results: "
                f"{json.dumps(sql_results or [], ensure_ascii=False, default=str)}\n\n"
                f"Legal evidence: "
                f"{json.dumps(evidence, ensure_ascii=False, default=str)}"
            )),
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
                "impact",
            ):
                if key in tool_result:
                    control_result[key] = tool_result[key]

        final_messages = self._build_synthesis_messages(
            question,
            synthesis_docs,
            control_result,
            state.comparison_report,
            state.sql_results,
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

        if relationship_args and state.plan == ["analyze_regulation_impact"]:
            result = _run_tool(
                self.vectorstore,
                "analyze_regulation_impact",
                relationship_args,
            )
            normalized = self._record_step(
                state,
                round_number=1,
                tool="analyze_regulation_impact",
                arguments=relationship_args,
                result=result,
            )
            state.comparison_report = normalized.get("comparison")
            return self._finalize(question, state, normalized)

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
