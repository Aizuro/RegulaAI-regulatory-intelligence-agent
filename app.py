"""
Regula — Regulatory Intelligence Assistant

Streamlit frontend for the Agentic AI backend.
The frontend communicates with FastAPI /query instead of directly
initializing or mutating the RAG/vectorstore.
"""

import re
import time
from typing import Any, Dict, List

import requests
import streamlit as st
import os


# ============================================================
# KONFIGURASI
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Regula — Regulatory Intelligence Assistant",
    page_icon="⚖️",
    layout="centered",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        max-width: 780px;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.25rem 0 !important;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 12px !important;
    }

    .welcome-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 4rem;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        text-align: center;
        font-size: 1rem;
        color: gray;
        margin-bottom: 2rem;
    }

    .welcome-description {
        text-align: center;
        font-size: 0.9rem;
        color: gray;
        margin-bottom: 2rem;
    }

    .stButton button {
        border-radius: 20px !important;
        border: 1px solid rgba(128,128,128,0.3) !important;
        background: transparent !important;
        font-size: 0.85rem !important;
        padding: 0.3rem 0.9rem !important;
        color: inherit !important;
        transition: background 0.2s !important;
    }

    .stButton button:hover {
        background: rgba(128,128,128,0.1) !important;
    }

    [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }

    .status-ok {
        font-size: 0.85rem;
        color: #16803c;
    }

    .status-error {
        font-size: 0.85rem;
        color: #b42318;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_status" not in st.session_state:
    st.session_state.api_status = None


# ============================================================
# API CLIENT
# ============================================================

def check_api_health() -> bool:
    """Check whether the FastAPI backend is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.ok
    except requests.RequestException:
        return False


def query_agent(question: str) -> Dict[str, Any]:
    """Send a question to the Agentic AI backend."""
    response = requests.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=120,
    )

    if response.ok:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    raise RuntimeError(f"API error ({response.status_code}): {detail}")


# ============================================================
# RENDER HELPERS
# ============================================================

def clean_answer_for_display(answer: str) -> str:
    """Remove formatting artifacts that should never reach the user."""
    if not answer:
        return ""

    cleaned = answer
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander("📚 Sources"):
        for i, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            content = source.get("content", "")

            document_type = metadata.get("document_type", "")
            number = metadata.get("regulation_number", "")
            year = metadata.get("year", "")
            pasal = metadata.get("pasal", "")
            ayat = metadata.get("ayat", "")
            page = metadata.get("page")

            title_parts = []
            if document_type:
                title_parts.append(document_type)
            if number:
                title_parts.append(f"Nomor {number}")
            if year:
                title_parts.append(f"Tahun {year}")

            title = " ".join(title_parts) or metadata.get(
                "source", f"Source {i}")

            if pasal:
                title += f" — Pasal {pasal}"
            if ayat:
                title += f" Ayat {ayat}"
            if page is not None:
                title += f" · halaman {page}"

            st.markdown(f"**{i}. {title}**")

            source_name = metadata.get("source")
            if source_name:
                st.caption(f"📄 {source_name}")

            if content:
                st.caption(content[:1200])

            if i < len(sources):
                st.divider()


def render_external_citations(
    citations: List[Dict[str, Any]]
) -> None:
    """Render verified external regulatory sources as clickable citations."""
    if not citations:
        return

    with st.expander("🌐 External Sources"):
        for i, citation in enumerate(citations, start=1):
            title = citation.get("title") or f"External source {i}"
            url = str(citation.get("url") or "").strip()
            source_name = citation.get(
                "source_name") or citation.get("domain") or ""
            tier = citation.get("source_tier")
            evidence = citation.get("evidence") or ""

            st.markdown(f"**{i}. {title}**")

            meta = []
            if source_name:
                meta.append(str(source_name))
            if tier is not None:
                meta.append(f"Tier {tier}")
            if meta:
                st.caption(" · ".join(meta))

            if url.startswith("https://"):
                st.markdown(f"[Open source]({url})")

            if evidence:
                st.caption(evidence[:1200])

            if i < len(citations):
                st.divider()


def render_missing_regulations(
    missing_regulations: List[Dict[str, Any]]
) -> None:
    if not missing_regulations:
        return

    st.warning(
        "Sumber regulasi yang diminta tidak tersedia dalam basis pengetahuan.")

    with st.expander("🔍 Regulasi yang diminta"):
        for item in missing_regulations:
            if isinstance(item, dict):
                document_type = item.get("document_type", "")
                number = item.get("regulation_number", "")
                year = item.get("year", "")

                parts = [
                    str(x)
                    for x in [document_type, number, year]
                    if x not in ("", None)
                ]

                if parts:
                    st.write(" ".join(parts))
                else:
                    st.json(item)
            else:
                st.write(str(item))


def render_agent_result(result: Dict[str, Any]) -> None:
    """Render all structured fields returned by FastAPI."""
    st.markdown(clean_answer_for_display(result.get("answer", "")))

    render_missing_regulations(result.get("missing_regulations") or [])
    render_sources(result.get("sources") or [])
    render_external_citations(result.get("external_citations") or [])


# ============================================================
# CHAT
# ============================================================

def send_question(question: str) -> None:
    question = question.strip()

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing regulation..."):
            start_time = time.time()

            try:
                result = query_agent(question)
                elapsed = time.time() - start_time

                render_agent_result(result)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "result": result,
                        "response_time": elapsed,
                    }
                )

            except requests.RequestException as exc:
                message = (
                    "Backend API tidak dapat dihubungi. "
                    "Pastikan FastAPI berjalan di "
                    f"`{API_URL}`."
                )
                st.error(message)
                st.caption(str(exc))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message,
                    }
                )

            except Exception as exc:
                message = f"Terjadi error saat memproses pertanyaan: {exc}"
                st.error(message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message,
                    }
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚖️ Regula")
    st.caption("Regulatory Intelligence Assistant")
    st.divider()

    st.markdown("**Backend**")

    api_is_healthy = check_api_health()

    if api_is_healthy:
        st.markdown(
            '<p class="status-ok">● API connected</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="status-error">● API unavailable</p>',
            unsafe_allow_html=True,
        )
        st.caption("Start FastAPI with:")
        st.code("uvicorn src.api:app --reload")

    st.divider()

    st.markdown("**Capabilities**")
    st.caption("🔎 Regulatory research")
    st.caption("⚖️ Regulation comparison")
    st.caption("📊 Regulatory data analysis")
    st.caption("📚 Source-grounded answers")

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
        type="secondary",
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# MAIN — WELCOME / CHAT HISTORY
# ============================================================

if not st.session_state.messages:
    st.markdown(
        '<p class="welcome-title">⚖️ Regula</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="welcome-subtitle">'
        "Regulatory Intelligence Assistant"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="welcome-description">'
        "Research, compare, and analyze Indonesian regulations with AI."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
        "Try asking:"
        "</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    example_questions = [
        "Apa ketentuan mengenai transaksi elektronik?",
        "Bandingkan dua regulasi",
        "Berapa jumlah regulasi berdasarkan sektor?",
    ]

    for col, question in zip([col1, col2, col3], example_questions):
        with col:
            if st.button(question, use_container_width=True):
                send_question(question)
                st.rerun()

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(clean_answer_for_display(message["content"]))

            if message["role"] == "assistant":
                result = message.get("result")

                if result:
                    render_missing_regulations(
                        result.get("missing_regulations") or []
                    )
                    render_sources(result.get("sources") or [])
                    render_external_citations(
                        result.get("external_citations") or [])


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask about Indonesian regulations...",
    disabled=not api_is_healthy,
):
    send_question(prompt)
    st.rerun()
