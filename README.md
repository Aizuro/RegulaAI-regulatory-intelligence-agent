# ⚖️ Regulatory Intelligence Agent (Regula)

**Regula** adalah aplikasi AI untuk membantu pencarian, pemahaman,
perbandingan, dan analisis regulasi Indonesia. Proyek ini menggabungkan
Retrieval-Augmented Generation (RAG), Agentic AI, pemrosesan evidence,
dan monitoring perubahan sumber regulasi.

> **Disclaimer:** Regula merupakan proyek bantuan informasi dan
> demonstrasi teknologi AI, bukan pengganti nasihat atau interpretasi
> hukum profesional. Verifikasi hasil terhadap dokumen resmi dan minta
> peninjauan ahli untuk keputusan hukum.

## ✨ Kapabilitas Utama

-   **Regulatory retrieval (RAG):** mengambil evidence dari knowledge
    base regulasi berbasis ChromaDB.
-   **Agentic tool calling:** memilih tool yang relevan berdasarkan
    pertanyaan.
-   **Regulatory comparison:** membandingkan ketentuan berdasarkan
    evidence yang berhasil diambil.
-   **SQL / data analysis:** analisis read-only terhadap dataset
    regulasi terstruktur.
-   **External regulatory research:** pencarian lanjutan melalui sumber
    resmi yang diizinkan.
-   **Grounded impact analysis:** analisis potensi dampak berdasarkan
    hasil comparison, disertai validasi evidence.
-   **Regulation change monitoring:** deteksi perubahan sumber,
    persistence state, analisis, dan notifikasi melalui workflow
    otomatis.
-   **REST API dan UI:** FastAPI untuk backend dan Streamlit untuk
    antarmuka.

## 🏗️ Arsitektur

### Alur pertanyaan regulasi

``` text
User Question
     |
     v
Streamlit UI
     | HTTPS
     v
FastAPI
     |
     v
Regulatory Agent
     |
     v
Intent / Tool Routing
     |
     +-------------------+-------------------+
     |                   |                   |
     v                   v                   v
 Internal RAG       Comparison Engine     SQL Tools
 (ChromaDB)               |
     |                    |
     +--------------------+
              |
              v
 External Research (when applicable)
              |
              v
 Grounded Answer + Sources / Evidence
```

Agent tidak selalu menjalankan semua jalur. Tool yang dipilih bergantung
pada pertanyaan, hasil retrieval, dan evidence yang tersedia.

### Alur impact analysis V8

``` text
User Question
     |
     v
Agent / Deterministic Routing
     |
     v
analyze_regulation_impact
     |
     v
V8.1 Document Retrieval
     |
     v
V8.2 Structural Comparison
     |
     v
V8.3 Grounded Impact Analysis
     |
     v
Evidence Guardrail
     |
     v
Agent Final Answer
```

V8 menggunakan comparison engine yang sudah ada melalui adapter.
Evidence guardrail memeriksa agar evidence pada analisis dampak merujuk
pada ketentuan yang ada di comparison report. Ini mengurangi klaim tanpa
dukungan, tetapi bukan jaminan bahwa hallucination dapat dihilangkan
sepenuhnya.

### Alur monitoring regulasi

``` text
Regulatory Source / JDIH
          |
          v
         n8n
          |
          v
    FastAPI / Python
          |
          v
 Change Detection + Persistence
          |
          v
      AI Analysis
          |
          v
        Telegram
```

Monitoring merupakan jalur terpisah dari tanya-jawab RAG. Perubahan
sumber atau snapshot tidak otomatis berarti regulasi dicabut atau
berubah secara hukum.

## 🧩 Detail Fitur

### 1. Metadata-aware RAG

ChromaDB digunakan sebagai vector store. Metadata seperti jenis, nomor,
tahun, bab, pasal, dan ayat dapat membantu retrieval lebih terarah
ketika identifier regulasi diketahui. Cakupan jawaban bergantung pada
dokumen dan kualitas parsing/metadata.

### 2. Regulatory comparison

Comparison menyusun perbedaan berdasarkan evidence yang berhasil
di-retrieve, seperti ketentuan yang ter-align, hanya ditemukan di
dokumen A/B, atau memiliki teks/struktur berbeda.

**Batasan:** ini adalah *structural comparison over retrieved evidence*,
bukan full-document legal diff. Perbedaan teks atau struktur tidak
dengan sendirinya membuktikan perubahan hukum substantif.

### 3. SQL / data analysis

Tool analisis data bersifat read-only untuk pertanyaan terhadap dataset
regulasi terstruktur, jika kolom dan data yang diperlukan tersedia.

### 4. External regulatory research

External research digunakan ketika knowledge base internal tidak
memadai. Dokumentasi proyek mencantumkan sumber resmi seperti JDIH BPK
dan JDIHN, dengan validasi sumber sebelum evidence digunakan. Hasil
bergantung pada akses situs dan konten yang dapat diambil.

### 5. Grounded impact analysis

Analisis dampak V8 didasarkan pada hasil retrieval dan comparison. Jika
evidence tidak cukup, sistem dapat mengembalikan
`impact_level = unknown` dan `requires_further_research = true`,
alih-alih membuat kesimpulan tanpa dasar.

Prinsip semantik: - Ketentuan yang hanya ditemukan pada satu dokumen
tidak otomatis berarti dicabut atau baru berlaku. - Perbedaan teks tidak
otomatis berarti perubahan substantif. - Hilangnya data dari snapshot
tidak otomatis berarti pencabutan hukum. - Metadata yang hilang tidak
boleh digunakan untuk mengarang status atau tanggal publikasi.

### 6. Monitoring dan change detection

Workflow monitoring menggunakan n8n untuk orkestrasi, Python/FastAPI
untuk logika domain, persistence untuk state/snapshot, dan Telegram
untuk notifikasi. Pengujian monitoring mencakup konsistensi
key/fingerprint, deteksi perubahan, penolakan key duplikat, dan
persistensi snapshot.

## 🧰 Tech Stack

  Area                  Teknologi
  --------------------- ----------------------------------------------------
  Bahasa                Python
  LLM                   Groq API dengan model dari konfigurasi environment
  Embedding             Google Gemini Embedding
  Vector store          ChromaDB
  RAG dan agent         LangChain
  Backend API           FastAPI
  Frontend              Streamlit
  Workflow automation   n8n
  Structured data       SQL / database relasional sesuai konfigurasi
  Testing               pytest
  Notifications         Telegram

Simpan kredensial pada environment atau platform secrets. Jangan menaruh
API key di source code atau README.

## ☁️ Deployment

UI menggunakan **Streamlit Community Cloud**. Berdasarkan arsitektur
yang didokumentasikan, UI berkomunikasi melalui HTTPS dengan backend
FastAPI yang di-host terpisah, misalnya di Render.

``` text
Streamlit Community Cloud
        Regula UI
            |
            | HTTPS / POST /query
            v
      FastAPI Backend
      + Agent / Tools
      + Chroma Knowledge Base
            |
            v
          LLM API
```

Pastikan URL backend yang dikonfigurasi di Streamlit Secrets mengarah ke
backend aktif. Nama secret, endpoint, dan command deployment perlu
disesuaikan dengan konfigurasi repository yang benar-benar digunakan.

## 🚀 Menjalankan Secara Lokal

Instruksi berikut merupakan pola umum berdasarkan dokumentasi proyek.
Sesuaikan dengan struktur dan konfigurasi repository aktual.

### 1. Clone repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

### 2. Buat virtual environment

``` bash
python -m venv .venv
```

Aktifkan di Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Konfigurasi environment

Contoh variabel yang pernah digunakan dalam dokumentasi proyek:

``` env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
API_URL=http://127.0.0.1:8000
```

Nama variabel di atas harus diverifikasi terhadap kode versi yang
digunakan; tidak semua komponen harus memakai seluruh variabel tersebut.

### 5. Jalankan backend dan UI

Contoh command backend yang tercantum dalam dokumentasi proyek:

``` bash
uvicorn src.api:app --reload
```

UI Streamlit:

``` bash
streamlit run app.py
```

Jalankan di terminal terpisah. Jika entry point atau konfigurasi URL
berbeda pada branch/deployment aktif, ikuti konfigurasi repository
tersebut.

## 🔌 API

Dokumentasi lama mencantumkan endpoint berikut:

-   `GET /health` --- health check backend.
-   `POST /query` --- menerima pertanyaan dan mengembalikan jawaban
    agent.

Contoh request konseptual:

``` json
{
  "question": "Apa ketentuan dalam regulasi yang tersedia di knowledge base?"
}
```

Schema response mengikuti versi API yang sedang dijalankan. Periksa
OpenAPI di `/docs` saat backend aktif; jangan mengandalkan contoh
response lama sebagai kontrak API terkini.

## 🧪 Testing dan Validasi

Proyek menggunakan unit test, contract test, integration test, serta
test monitoring dan persistence.

Hasil regression suite yang tercatat dari salah satu eksekusi
pengembangan:

``` text
Collected: 136 tests
Passed:    126
Skipped:    10
Failed:      0
Warnings:    4
```

Sepuluh test yang dilewati merupakan test pre-deployment V6 bersifat
opt-in yang menggunakan `RUN_V6_PREDEPLOYMENT=1`. Ini adalah hasil dari
eksekusi tertentu, bukan jaminan bahwa semua branch atau deployment
mendapat hasil identik. Jalankan kembali test pada commit yang akan
dirilis.

Area pengujian yang terdokumentasi: - kontrak dan routing tool impact
analysis; - validasi evidence pada hasil analisis; - parsing JSON dan
validasi struktur output; - integrasi retrieval → comparison → impact
analysis; - monitoring key dan fingerprint; - deteksi sumber baru,
berubah, atau hilang; - penolakan key duplikat; dan - persistence
snapshot/state.

Contoh menjalankan pytest:

``` bash
python -m pytest -v
```

Gunakan virtual environment dan konfigurasi layanan yang sesuai.
Sebagian test dapat membutuhkan Redis, corpus lokal, environment
variable, atau flag opt-in. Untuk command spesifik, ikuti `TESTING.md`,
`pytest.ini`, dan instruksi test di repository.

## ⚠️ Batasan yang Diketahui

1.  **Cakupan knowledge base:** retrieval terbatas pada dokumen yang
    tersedia dan berhasil diproses.
2.  **Kualitas parsing:** ekstraksi PDF/OCR dan metadata Bab/Pasal/Ayat
    dapat memengaruhi alignment.
3.  **Structural comparison:** bukan pengganti diff hukum menyeluruh
    atau kajian legal.
4.  **Impact analysis:** output LLM bersifat probabilistik; guardrail
    mengurangi klaim tanpa dukungan tetapi tidak menjamin nol
    hallucination.
5.  **External research:** dibatasi oleh sumber yang diizinkan,
    ketersediaan situs, dan kemampuan mengambil konten.
6.  **Monitoring:** perubahan sumber bukan bukti otomatis tentang
    pencabutan atau keberlakuan hukum.
7.  **Production readiness:** test yang lulus tidak dengan sendirinya
    membuktikan kesiapan untuk penggunaan hukum kritis atau beban
    produksi besar.

## 🎯 Tujuan Portofolio

Regula mendemonstrasikan pengalaman membangun aplikasi Applied AI yang
mencakup: - RAG dan metadata-aware retrieval; - agentic orchestration
dan tool calling; - comparison berbasis evidence; - SQL / structured
data analysis; - API development dengan FastAPI; - UI dengan
Streamlit; - source validation dan evidence-grounded generation; -
workflow automation dengan n8n; - change detection dan persistence; -
unit, contract, integration, dan regression testing; serta - deployment
frontend/backend yang terpisah.

## 📌 Status Proyek

Dokumentasi V8.5 mencatat V8.1--V8.4 selesai untuk retrieval, structural
comparison, grounded impact analysis, evidence guardrail, routing/tool
integration, dan real integration test. V8.5 berfokus pada finalisasi
dokumentasi dan pengujian.

Status test dan deployment harus diverifikasi pada commit dan lingkungan
yang benar-benar dipublikasikan. Regula sebaiknya dipresentasikan
sebagai **AI application / portfolio MVP**, bukan sistem pengambilan
keputusan hukum production-grade.

## 📄 Disclaimer

Regulatory Intelligence Agent dibuat untuk tujuan pembelajaran,
eksperimen, dan demonstrasi teknologi AI. Informasi yang dihasilkan
bukan nasihat hukum. Untuk kebutuhan hukum nyata, periksa dokumen
terbaru dari institusi resmi dan mintalah penilaian profesional yang
kompeten.
