# app.py

import streamlit as st
import tempfile
import os
import time
from pathlib import Path

from src.rag_chain import build_rag_chain, ask, setup_vectorstore, format_sources_for_display
from src.document_loader import load_single_pdf
from src.text_splitter import split_documents
from src.embeddings import (
    load_vectorstore,
    create_vectorstore,
    add_documents_to_vectorstore,  # import dari embeddings, bukan tulis ulang di sini
    vectorstore_exists
)
from src.logger import setup_logger, log_query, log_document_added, log_error
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="LexAI — Asisten Hukum Indonesia",
    page_icon="⚖️",
    layout="centered"  # centered agar mirip ChatGPT, tidak full width
)

# CSS custom untuk mempercantik tampilan
# Kita inject CSS untuk hal-hal yang tidak bisa dilakukan Streamlit secara native
st.markdown("""
<style>
    /* Sembunyikan header & footer bawaan Streamlit */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Lebar konten utama — mirip ChatGPT yang tidak terlalu lebar */
    .block-container {
        max-width: 780px;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Styling chat message agar lebih bersih */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.25rem 0 !important;
    }

    /* Input chat — tambah sedikit radius */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
    }

    /* Hilangkan border merah/fokus default yang mengganggu */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
    }

    /* Welcome screen — nama besar di tengah */
    .welcome-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 4rem;
        margin-bottom: 0.5rem;
        color: inherit;
    }

    .welcome-subtitle {
        text-align: center;
        font-size: 1rem;
        color: gray;
        margin-bottom: 3rem;
    }

    /* Chip contoh pertanyaan */
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

    /* Sumber dokumen expander */
    [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "logger" not in st.session_state:
    st.session_state.logger = setup_logger()


# ============================================================
# FUNGSI HELPER
# ============================================================

def initialize_rag():
    """Load vectorstore yang sudah ada dan bangun RAG chain."""
    with st.spinner("Memuat sistem RAG..."):
        vectorstore = setup_vectorstore()
        rag_chain, retriever = build_rag_chain(vectorstore)
        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever


def process_uploaded_pdf(uploaded_file) -> bool:
    """
    Menerima file PDF dari widget upload Streamlit,
    lalu menambahkan isinya ke vectorstore yang sudah ada.
    Menggunakan add_documents_to_vectorstore() dari embeddings.py.

    Args:
        uploaded_file: objek file dari st.file_uploader()

    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Streamlit memberikan file sebagai bytes di memory.
        # PyPDFLoader butuh path file di disk, jadi kita simpan sementara.
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner(f"Memproses {uploaded_file.name}..."):
            documents = load_single_pdf(tmp_path)
            chunks = split_documents(documents)

            # Kalau vectorstore belum ada sama sekali, buat baru
            if st.session_state.vectorstore is None:
                vectorstore = create_vectorstore(chunks)
            else:
                # Kalau sudah ada, tambahkan saja dokumen barunya
                # Fungsi ini sekarang diimport dari embeddings.py
                vectorstore = add_documents_to_vectorstore(
                    st.session_state.vectorstore,
                    chunks
                )

            # Rebuild chain dengan vectorstore terbaru
            rag_chain, retriever = build_rag_chain(vectorstore)
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever

        os.unlink(tmp_path)

        log_document_added(
            st.session_state.logger,
            filename=uploaded_file.name,
            num_chunks=len(chunks)
        )

        return True

    except Exception as e:
        st.error(f"Gagal memproses PDF: {str(e)}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False


def send_question(question: str):
    """
    Proses pertanyaan user, tampilkan di chat, dan generate jawaban.
    Dipisah jadi fungsi tersendiri agar bisa dipanggil dari
    input chat maupun tombol contoh pertanyaan.
    """
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate dan tampilkan jawaban
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            start_time = time.time()

            result = ask(
                st.session_state.rag_chain,
                st.session_state.retriever,
                question
            )

            response_time = time.time() - start_time

            log_query(
                logger=st.session_state.logger,
                query=question,
                source_documents=result["sources"],
                answer=result["answer"],
                response_time=response_time
            )

        st.markdown(result["answer"])

        sources_text = format_sources_for_display(result["sources"])
        with st.expander("📚 Lihat sumber dokumen"):
            st.text(sources_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources_text
    })


# ============================================================
# SIDEBAR — Upload & Manajemen Dokumen
# ============================================================

with st.sidebar:
    st.markdown("### ⚖️ LexAI")
    st.caption("Asisten Hukum Indonesia")
    st.divider()

    # Upload PDF
    st.markdown("**Tambah Dokumen**")
    uploaded_file = st.file_uploader(
        "Upload PDF regulasi",
        type="pdf",
        label_visibility="collapsed",
        help="Upload dokumen PDF peraturan atau regulasi Indonesia dari jdih.go.id"
    )

    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            success = process_uploaded_pdf(uploaded_file)
            if success:
                st.session_state.processed_files.append(uploaded_file.name)
                st.success(f"✅ Berhasil ditambahkan!")
                st.rerun()

    # Daftar dokumen aktif
    if st.session_state.processed_files:
        st.divider()
        st.markdown("**Dokumen Aktif**")
        for fname in st.session_state.processed_files:
            st.caption(f"📄 {fname}")

    # Tombol reset
    st.divider()
    if st.button("🗑️ Reset Semua", use_container_width=True, type="secondary"):
        import shutil
        if Path("vectorstore").exists():
            shutil.rmtree("vectorstore")
        st.session_state.clear()
        st.rerun()


# ============================================================
# AREA UTAMA — Welcome Screen atau Chat
# ============================================================

# Kalau belum ada percakapan → tampilkan welcome screen
if not st.session_state.messages:
    st.markdown('<p class="welcome-title">⚖️ LexAI</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="welcome-subtitle">Asisten tanya jawab dokumen hukum & regulasi Indonesia</p>',
        unsafe_allow_html=True
    )

    # Contoh pertanyaan — hanya muncul kalau RAG sudah siap
    if st.session_state.rag_chain is not None:
        st.markdown(
            "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
            "Contoh pertanyaan:</p>",
            unsafe_allow_html=True
        )

        # Tiga tombol contoh pertanyaan ditampilkan horizontal
        col1, col2, col3 = st.columns(3)
        example_questions = [
            "Apa sanksi tidak membayar pajak?",
            "Jelaskan hak wajib pajak",
            "Apa itu surat ketetapan pajak?"
        ]

        for col, question in zip([col1, col2, col3], example_questions):
            with col:
                if st.button(question, use_container_width=True):
                    send_question(question)
                    st.rerun()

    elif not vectorstore_exists():
        # Belum ada dokumen sama sekali
        st.markdown(
            "<p style='text-align:center; color:gray; margin-top:1rem;'>"
            "Upload dokumen PDF di sidebar untuk memulai ⬅️</p>",
            unsafe_allow_html=True
        )

# Kalau sudah ada percakapan → tampilkan riwayat chat
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Lihat sumber dokumen"):
                    st.text(message["sources"])


# ============================================================
# INPUT CHAT — selalu di bawah (fixed by Streamlit native)
# ============================================================

if prompt := st.chat_input(
    "Tanyakan sesuatu tentang dokumen hukum...",
    disabled=(st.session_state.rag_chain is None)
):
    if st.session_state.rag_chain is None:
        st.warning("⚠️ Upload dokumen PDF terlebih dahulu.")
    else:
        send_question(prompt)
        st.rerun()


# ============================================================
# INISIALISASI AWAL — load vectorstore kalau sudah ada di disk
# ============================================================

if st.session_state.rag_chain is None and vectorstore_exists():
    initialize_rag()
    st.rerun()
