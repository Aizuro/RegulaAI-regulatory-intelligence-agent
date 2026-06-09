---
title: RAG Dokumen Hukum Indonesia
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
---

# ⚖️ RAG Chatbot Dokumen Hukum Indonesia

Chatbot tanya jawab dokumen regulasi dan peraturan perundang-undangan 
Indonesia menggunakan RAG (Retrieval Augmented Generation).

## Teknologi
- **LLM**: Llama 3.3 70B via Groq API
- **Embedding**: Google Gemini Embedding
- **Vector DB**: ChromaDB
- **Framework**: LangChain + Streamlit

## Cara Pakai
1. Upload dokumen PDF regulasi Indonesia (dari jdih.go.id)
2. Tunggu dokumen selesai diproses
3. Mulai ajukan pertanyaan tentang isi dokumen

## Catatan
Jawaban dihasilkan berdasarkan dokumen yang diupload.
Selalu verifikasi ke dokumen asli untuk kepastian hukum.