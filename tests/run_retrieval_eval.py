"""Evaluasi retrieval Regula (tanpa LLM, hanya embedding query).

Metrik per mode retriever:
  - Recall@1 dan Recall@K : ada chunk relevan di top-1 / top-K
  - MRR                   : 1 / peringkat chunk relevan pertama (0 jika tidak ada)
  - DocPrecision@K        : persen chunk top-K yang berasal dari dokumen yang diharapkan
  - ChunkPrecision@K      : persen chunk top-K yang benar-benar memuat teks bukti

Chunk dianggap RELEVAN jika:
  (a) nama file sumbernya ada di expected_sources, DAN
  (b) teksnya (huruf kecil, spasi dinormalisasi) memuat salah satu evidence_any.
Definisi ini sengaja tidak bergantung pada metadata Bab/Pasal, karena metadata itu
sendiri bagian dari sistem yang sedang diuji.

Mode:
  legal      : retriever produksi (get_retriever: filter metadata + MMR)
  mmr        : MMR polos (baseline)
  similarity : similarity search polos (baseline)

Jalankan dari ROOT project (folder yang berisi src/ dan vectorstore/):
  python eval/run_retrieval_eval.py --split test --k 4 --mode all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def source_name(doc: Any) -> str:
    md = getattr(doc, "metadata", None) or {}
    return str(md.get("source", "")).replace("\\", "/").split("/")[-1].lower()


def is_relevant(doc: Any, item: Dict[str, Any]) -> bool:
    expected = {s.lower() for s in item["expected_sources"]}
    if source_name(doc) not in expected:
        return False
    text = norm(getattr(doc, "page_content", ""))
    return any(norm(e) in text for e in item["evidence_any"])


def score_item(docs: List[Any], item: Dict[str, Any], k: int) -> Dict[str, Any]:
    top = docs[:k]
    expected = {s.lower() for s in item["expected_sources"]}
    rel_flags = [is_relevant(d, item) for d in top]
    first = next((i + 1 for i, f in enumerate(rel_flags) if f), None)
    n = max(len(top), 1)
    return {
        "recall_at_1": bool(rel_flags[:1] and rel_flags[0]),
        "recall_at_k": any(rel_flags),
        "mrr": (1.0 / first) if first else 0.0,
        "doc_precision_at_k": sum(source_name(d) in expected for d in top) / n,
        "chunk_precision_at_k": sum(rel_flags) / n,
        "first_relevant_rank": first,
        "retrieved": [
            {
                "source": source_name(d),
                "page": (d.metadata or {}).get("page"),
                "pasal": (d.metadata or {}).get("pasal"),
                "ayat": (d.metadata or {}).get("ayat"),
                "relevant": f,
            }
            for d, f in zip(top, rel_flags)
        ],
    }


def summarize(scores: List[Dict[str, Any]], k: int) -> Dict[str, float]:
    n = len(scores)
    if n == 0:
        return {}
    avg = lambda key: round(sum(float(s[key]) for s in scores) / n, 4)
    return {
        "n": n,
        "recall_at_1": avg("recall_at_1"),
        f"recall_at_{k}": avg("recall_at_k"),
        "mrr": avg("mrr"),
        f"doc_precision_at_{k}": avg("doc_precision_at_k"),
        f"chunk_precision_at_{k}": avg("chunk_precision_at_k"),
    }


def build_retrievers(k: int) -> Dict[str, Callable[[str], List[Any]]]:
    """Import berat (langchain/chroma) ditunda agar fungsi metrik bisa diuji offline."""
    from src.embeddings import load_legal_vectorstore
    from src.retriever import get_retriever

    vs = load_legal_vectorstore()
    legal = get_retriever(vs, k=k, search_type="mmr")
    return {
        "legal": lambda q: legal.invoke(q),
        "mmr": lambda q: vs.max_marginal_relevance_search(
            q, k=k, fetch_k=k * 3, lambda_mult=0.7
        ),
        "similarity": lambda q: vs.similarity_search(q, k=k),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "gold_retrieval.json"))
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mode", choices=["legal", "mmr", "similarity", "all"], default="all")
    ap.add_argument("--sleep", type=float, default=0.5, help="jeda antar query (detik)")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))["items"]
    if args.split != "all":
        gold = [g for g in gold if g["split"] == args.split]
    unverified = [g["id"] for g in gold if not g.get("verified_by_human")]
    if unverified:
        print(f"PERINGATAN: {len(unverified)} item belum diverifikasi manual "
              f"(verified_by_human=false). Angka belum boleh masuk CV.\n")

    retrievers = build_retrievers(args.k)
    modes = list(retrievers) if args.mode == "all" else [args.mode]

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": args.split, "k": args.k, "n_questions": len(gold), "modes": {},
    }
    for mode in modes:
        scores, per_item = [], []
        for item in gold:
            try:
                docs = retrievers[mode](item["question"])
                sc = score_item(docs, item, args.k)
            except Exception as exc:  # satu error tidak boleh menghentikan semuanya
                sc = {"recall_at_1": False, "recall_at_k": False, "mrr": 0.0,
                      "doc_precision_at_k": 0.0, "chunk_precision_at_k": 0.0,
                      "first_relevant_rank": None, "retrieved": [],
                      "error": f"{type(exc).__name__}: {exc}"}
            scores.append(sc)
            per_item.append({"id": item["id"], "question": item["question"], **sc})
            time.sleep(args.sleep)
        summ = summarize(scores, args.k)
        report["modes"][mode] = {"summary": summ, "items": per_item}

        print("=" * 70)
        print(f"MODE: {mode}   split={args.split}   k={args.k}")
        print("=" * 70)
        for key, val in summ.items():
            print(f"  {key:24s}: {val}")
        misses = [p for p in per_item if not p["recall_at_k"]]
        if misses:
            print(f"  Gagal (Recall@{args.k}=0): " + ", ".join(m["id"] for m in misses))
        print()

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"retrieval_{args.split}_k{args.k}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hasil lengkap per pertanyaan: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
