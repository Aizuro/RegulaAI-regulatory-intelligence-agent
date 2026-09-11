# src/llm.py

import os
from dotenv import load_dotenv
# ganti dari langchain_google_genai
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List

load_dotenv()


def get_llm() -> ChatGroq:
    """
    Membuat instance Groq LLM.
    Groq menggunakan model open-source (Llama) yang dijalankan
    di chip LPU mereka — sangat cepat dan gratis tanpa kartu kredit.

    Returns:
        ChatGroq object
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY tidak ditemukan! "
            "Daftar di console.groq.com dan tambahkan key ke file .env"
        )

    # Allow the model to be changed from .env without editing source code.
    # GPT-OSS 20B is currently listed by Groq as a production model.
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.1,                  # Tetap rendah untuk dokumen hukum
        max_tokens=2048,
        max_retries=2,                    # Retry otomatis kalau ada error sementara
    )

    return llm


# Template instruksi — tidak berubah dari sebelumnya
RAG_PROMPT_TEMPLATE = """Kamu adalah asisten hukum yang membantu menganalisis \
dokumen regulasi dan peraturan perundang-undangan Indonesia.

INSTRUKSI:
1. Jawab pertanyaan HANYA berdasarkan konteks dokumen yang diberikan di bawah.
2. Jika informasi tidak ada dalam konteks, katakan dengan jelas: \
"Informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
3. Jangan mengarang atau menambahkan informasi dari luar dokumen.
4. Gunakan Bahasa Indonesia yang formal dan mudah dipahami.
5. Jika relevan, sebutkan pasal atau bagian dokumen sebagai referensi.

KONTEKS DARI DOKUMEN:
{context}

PERTANYAAN:
{question}

JAWABAN:"""


def get_prompt_template() -> PromptTemplate:
    """
    Membuat prompt template untuk RAG chain.

    Returns:
        PromptTemplate object dengan variabel {context} dan {question}
    """
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return prompt


def test_llm_connection() -> None:
    """
    Tes koneksi ke Groq API dengan pertanyaan sederhana.
    Hanya untuk verifikasi saat setup awal.
    """
    print("Menguji koneksi ke Groq API...")

    llm = get_llm()
    response = llm.invoke("Jawab dengan satu kalimat: Apa itu hukum perdata?")

    print("Koneksi berhasil!")
    print(f"Model : {os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')}")
    print(f"Respons: {response.content}")
