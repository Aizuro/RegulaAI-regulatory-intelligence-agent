# ⚖️ Regulatory Intelligence Agent

**Regulatory Intelligence Agent** adalah aplikasi AI untuk membantu pencarian, pemahaman, perbandingan, dan analisis regulasi Indonesia menggunakan pendekatan **Agentic AI + Retrieval-Augmented Generation (RAG)**.

Project ini dikembangkan dari RAG chatbot dokumen hukum menjadi sistem agent yang dapat memilih dan menjalankan beberapa kemampuan sesuai kebutuhan pertanyaan pengguna:

- 🔎 **Regulatory Retrieval** — mencari ketentuan dari knowledge base regulasi
- 🤖 **Agentic Tool Calling** — menentukan tool yang relevan untuk menyelesaikan pertanyaan
- ⚖️ **Regulatory Comparison** — membandingkan ketentuan dari dua regulasi
- 📊 **SQL / Data Analysis** — melakukan analisis terhadap dataset regulasi terstruktur
- 🌐 **External Regulatory Research** — mencari sumber regulasi dari sumber resmi pemerintah ketika informasi internal tidak tersedia
- 📚 **Source-Grounded Answering** — menyertakan sumber dan evidence untuk membantu verifikasi
- 🚀 **REST API** — menyediakan backend melalui FastAPI
- 💬 **Streamlit UI** — antarmuka pengguna untuk berinteraksi dengan agent

> **Disclaimer:** Project ini merupakan sistem bantuan informasi, bukan pengganti nasihat atau interpretasi hukum profesional. Selalu verifikasi informasi ke dokumen dan sumber regulasi resmi.

---

## 🏗️ Architecture

```text
                         User
                          │
                          ▼
                  ┌───────────────┐
                  │  Streamlit UI │
                  │    Regula     │
                  └───────┬───────┘
                          │
                     POST /query
                          │
                          ▼
                  ┌───────────────┐
                  │    FastAPI    │
                  └───────┬───────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Regulatory Intelligence│
              │        Agent          │
              └───────────┬───────────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
     ┌─────────┐    ┌────────────┐   ┌─────────┐
     │   RAG   │    │ Comparison │   │   SQL   │
     │  Chroma │    │   Engine   │   │  Tools  │
     └─────────┘    └────────────┘   └─────────┘
          │
          │
          └───────────────┐
                          ▼
                 External Research
                          │
                  Search → Validate
                          │
                        Fetch
                          │
                          ▼
                    Final Synthesis
                          │
                          ▼
                    Grounded Answer
```

### Deployment Architecture

```text
┌─────────────────────────┐
│ Streamlit Community     │
│ Cloud                   │
│                         │
│ Regula UI               │
└────────────┬────────────┘
             │ HTTPS
             │ POST /query
             ▼
┌─────────────────────────┐
│ Render                  │
│                         │
│ FastAPI + Agent         │
│ + Chroma Knowledge Base │
└────────────┬────────────┘
             │
             ▼
        Groq API
```

---

## ✨ Key Features

### 1. Agentic Regulatory Retrieval

Agent tidak hanya melakukan similarity search secara langsung. Pertanyaan dianalisis terlebih dahulu untuk menentukan jalur penyelesaian yang sesuai.

Contoh:

```text
"Apa ketentuan dalam UU Nomor 1 Tahun 2024?"
                    │
                    ▼
             Detect regulation
                    │
                    ▼
             get_regulation
                    │
                    ▼
             Retrieve evidence
                    │
                    ▼
              Final answer
```

Untuk pertanyaan yang membutuhkan dua regulasi:

```text
"Bandingkan UU Nomor 1 Tahun 2024
 dengan UU Nomor 3 Tahun 2024"
                    │
                    ▼
         compare_regulations
                    │
                    ▼
          Comparison Report
                    │
                    ▼
              Final answer
```

---

### 2. Metadata-Aware RAG

Knowledge base menggunakan **ChromaDB** sebagai vector store.

Metadata regulasi digunakan untuk membantu retrieval yang lebih terarah, misalnya:

```text
jenis
nomor
tahun
bab
pasal
ayat
```

Dengan demikian retrieval tidak hanya bergantung pada semantic similarity, tetapi juga dapat mempertimbangkan identifier regulasi yang eksplisit.

---

### 3. Regulatory Comparison

Agent dapat membandingkan dua regulasi berdasarkan evidence yang berhasil di-retrieve.

Output comparison disusun secara terstruktur, termasuk:

- jumlah ketentuan yang ter-align
- ketentuan yang hanya ditemukan pada regulasi A
- ketentuan yang hanya ditemukan pada regulasi B
- ketentuan yang memiliki perubahan/perbedaan
- ringkasan perbedaan

> Comparison saat ini merupakan **structural comparison over retrieved evidence**, bukan full-document legal diff.

---

### 4. SQL / Data Analysis

Agent juga memiliki tool read-only untuk melakukan analisis terhadap dataset regulasi terstruktur.

Contoh pertanyaan:

```text
"Berapa jumlah regulasi berdasarkan sektor?"
```

atau:

```text
"Bagaimana distribusi status regulasi dalam dataset?"
```

Hasil analisis dapat dikembalikan dalam bentuk data terstruktur untuk ditampilkan di UI.

---

### 5. External Regulatory Research

Jika regulasi yang diminta tidak ditemukan dalam knowledge base internal, agent dapat melakukan fallback ke sumber eksternal yang telah divalidasi.

Saat ini external research menggunakan **allowlist sumber resmi**, antara lain:

- JDIH BPK
- JDIHN

Pipeline:

```text
Search
  ↓
Validate Source
  ↓
Fetch Source
  ↓
Extract Evidence
  ↓
Grounded Synthesis
```

External source tidak langsung dipercaya hanya karena ditemukan melalui search. Source divalidasi berdasarkan HTTPS, domain allowlist, source tier, title, dan snippet.

---

### 6. Source-Grounded Responses

Response agent dapat membawa:

- internal sources
- external sources
- citations
- evidence
- missing regulations
- comparison report
- SQL results

Tujuannya adalah membuat jawaban lebih mudah diverifikasi dan mengurangi risiko hallucination.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM | `openai/gpt-oss-20b` via Groq |
| Embedding | Google Gemini Embedding |
| Vector Store | ChromaDB |
| RAG / Agent Framework | LangChain |
| Backend API | FastAPI |
| Frontend | Streamlit |
| External Research | Official regulatory sources |
| Data Analysis | SQL |
| Deployment | Streamlit Community Cloud + Render |

### Why ChromaDB?

ChromaDB dipilih karena sesuai dengan skala dan kebutuhan knowledge base saat ini:

- lightweight
- mudah diintegrasikan dengan LangChain
- cocok untuk corpus kecil hingga menengah
- mendukung metadata filtering
- tidak membutuhkan infrastructure tambahan yang kompleks

Vector store dapat dimigrasikan ke solusi seperti **Qdrant** atau **pgvector** jika kebutuhan project berkembang menjadi multi-user atau production-scale retrieval.

---

## 📁 Project Structure

```text
agentic-ai/
│
├── src/
│   ├── __init__.py
│   ├── agent_orchestrator.py
│   ├── api.py
│   ├── comparison_engine.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── legal_chunker.py
│   ├── legal_indexer.py
│   ├── legal_ingestion.py
│   ├── legal_parser.py
│   ├── legal_tools.py
│   ├── llm_chain.py
│   ├── logger.py
│   ├── rag_chain.py
│   ├── retriever.py
│   ├── sql_tools.py
│   └── text_splitter.py
│
├── testing/
│   ├── test_agent.py
│   ├── evaluate_agent.py
│   └── ...
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd agentic-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Optional:

```env
GROQ_MODEL=openai/gpt-oss-20b
API_URL=http://127.0.0.1:8000
```

> Jangan commit file `.env` ke repository.

---

## ▶️ Running Locally

Project menggunakan arsitektur frontend/backend terpisah.

### Start FastAPI

Dari project root:

```bash
uvicorn src.api:app --reload
```

API akan tersedia di:

```text
http://127.0.0.1:8000
```

Health check:

```text
GET /health
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Start Streamlit

Pada terminal lain:

```bash
streamlit run app.py
```

Secara default UI akan mengakses:

```text
http://127.0.0.1:8000
```

Jika backend berada di URL berbeda, set:

```env
API_URL=https://your-api-url
```

---

## 🔌 API

### `GET /health`

Memeriksa status backend.

Example response:

```json
{
  "status": "ok",
  "service": "regulatory-intelligence-agent",
  "version": "6.5.0"
}
```

### `POST /query`

Request:

```json
{
  "question": "Apa ketentuan dalam UU Nomor 1 Tahun 2024?"
}
```

Response mencakup:

```json
{
  "answer": "...",
  "sources": [],
  "citations": [],
  "plan": [],
  "completed_steps": [],
  "missing_regulations": [],
  "comparison": null,
  "sql_results": [],
  "external_sources": [],
  "external_citations": []
}
```

---

## 🧪 Evaluation

Project dikembangkan secara incremental dengan evaluation pada setiap major milestone.

Status terakhir sebelum deployment:

```text
V2 Agent Evaluation          5/5 PASS
V3 Comparison                5/5 PASS
V4 SQL/Data Analysis         5/5 PASS
V6 Agent Evaluation          7/7 PASS
V6 Pre-deployment E2E       10/10 PASS
```

Pre-deployment test mencakup:

- health endpoint
- internal regulation retrieval
- regulatory comparison
- SQL analysis
- missing regulation handling
- external research contract
- external citation validation
- API response contract
- empty question handling
- missing question handling

---

## 🛡️ Design Principles

### Read-only knowledge base

Agent tidak melakukan indexing atau mutasi Chroma ketika menjalankan query production.

### Source validation

External research tidak menggunakan unrestricted web search. Sumber eksternal dibatasi melalui source allowlist dan validation.

### Missing-source preservation

Jika regulasi tidak tersedia, sistem mempertahankan informasi tersebut daripada membuat ketentuan hukum yang tidak memiliki evidence.

### Bounded context

Evidence yang dikirim ke LLM dibatasi untuk mengurangi request yang terlalu besar dan menjaga reliability.

### Separation of concerns

```text
UI
 ↓
API
 ↓
Agent
 ↓
Tools
 ↓
Data / External Sources
```

Frontend tidak mengakses atau memodifikasi vector store secara langsung.

---

## ⚠️ Current Limitations

Project masih memiliki beberapa keterbatasan yang diketahui:

1. **External research**
   - Saat ini sumber eksternal dibatasi pada domain resmi yang telah di-allowlist.
   - Beberapa query eksternal yang menghasilkan banyak evidence dapat melebihi token/request limit model yang digunakan.

2. **Regulatory comparison**
   - Comparison dilakukan berdasarkan retrieved evidence, bukan full-document diff.

3. **Knowledge base**
   - Cakupan regulasi bergantung pada dokumen yang tersedia di knowledge base.

4. **Legal accuracy**
   - Output AI bukan merupakan interpretasi hukum final.
   - Dokumen asli tetap menjadi sumber utama untuk verifikasi.

---

## 🗺️ Roadmap

```text
V1  RAG Foundation                 ✅
 ↓
V2  Agent + Tool Calling           ✅
 ↓
V3  Regulatory Comparison          ✅
 ↓
V4  SQL / Data Analysis            ✅
 ↓
V5  FastAPI                        ✅
 ↓
Regula Streamlit UI                ✅
 ↓
V6  External Regulatory Research   ✅
 ↓
V7  n8n Monitoring & Automation    🔜
 ↓
V8  Evaluation & Observability     🔜
 ↓
V9  Deployment & Production        🔜
```

---

## 🎯 Project Goals

Project ini ditujukan sebagai portfolio **AI Engineer / Applied AI Engineer** yang menunjukkan kemampuan dalam:

- RAG system design
- Agentic AI
- LLM tool calling
- metadata-aware retrieval
- document comparison
- structured data analysis
- API development
- external source validation
- source-grounded generation
- evaluation
- observability
- AI application deployment

---

## 📌 Status

**Current status: V6 — Deployment Ready**

Core agentic workflow dan pre-deployment API tests telah tervalidasi sebelum deployment.

Live demo dan production deployment akan ditambahkan setelah deployment selesai.

---

## 📄 Disclaimer

Regulatory Intelligence Agent dibuat untuk tujuan pembelajaran, eksperimen, dan demonstrasi teknologi AI.

Informasi yang diberikan sistem tidak dimaksudkan sebagai nasihat hukum. Untuk kebutuhan hukum yang sebenarnya, selalu periksa regulasi terbaru dan dokumen resmi dari institusi yang berwenang.
