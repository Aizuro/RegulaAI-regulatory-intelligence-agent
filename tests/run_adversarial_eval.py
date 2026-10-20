"""Evaluasi adversarial/negatif untuk agent Regula (end-to-end).

Memanggil agent langsung (src.agent_orchestrator.build_agent), BUKAN lewat /query,
sehingga cache Redis tidak ikut campur dan Redis tidak perlu hidup.

Aturan per kasus (semua opsional, lihat gold_adversarial.json):
  expect_any             : jawaban memuat minimal satu frasa (penolakan / 'tidak ditemukan')
  must_not_contain       : jawaban TIDAK boleh memuat frasa ini (kebocoran / menuruti injeksi)
  expect_no_local_sources: tidak ada chunk dari corpus lokal yang dikutip sebagai sumber

Pemeriksaan rule-based tidak sempurna. Baca jawaban mentah di file hasil untuk
kasus yang gagal, lalu putuskan apakah itu kegagalan sistem atau aturan yang kurang pas.

Jalankan dari ROOT project:
  python eval/run_adversarial_eval.py --sleep 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def check_case(case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    answer = norm(result.get("answer"))
    checks: Dict[str, bool] = {}

    if case.get("expect_any"):
        checks["expect_any"] = any(norm(p) in answer for p in case["expect_any"])
    if case.get("must_not_contain"):
        checks["must_not_contain"] = not any(
            norm(p) in answer for p in case["must_not_contain"]
        )
    if case.get("expect_no_local_sources"):
        checks["no_local_sources"] = not (result.get("sources") or [])
    if case.get("expect_tools"):
        used = {str(t.get("tool")) for t in result.get("trace", [])}
        checks["expect_tools"] = set(case["expect_tools"]).issubset(used)
    if not checks:
        checks["answered"] = bool(answer)

    return {"checks": checks, "passed": all(checks.values())}


def invoke_with_retry(agent, query: str, attempts: int = 4):
    """Ulangi panggilan saat kena rate limit (HTTP 429) dengan jeda bertahap."""
    for i in range(attempts):
        try:
            return agent.invoke(query)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if ("429" in msg or "rate limit" in msg) and i < attempts - 1:
                wait = 20 * (i + 1)
                print(f"   [rate limit] menunggu {wait}s lalu coba lagi ({i + 1}/{attempts - 1})...")
                time.sleep(wait)
                continue
            raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "gold_adversarial.json"))
    ap.add_argument("--sleep", type=float, default=15.0, help="jeda antar kasus (rate limit)")
    ap.add_argument("--only", default="", help="ID kasus dipisah koma, mis. A08,A03")
    args = ap.parse_args()

    cases = json.loads(Path(args.gold).read_text(encoding="utf-8"))["items"]
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"].upper() in wanted]

    from src.embeddings import load_legal_vectorstore
    from src.agent_orchestrator import build_agent

    agent = build_agent(load_legal_vectorstore())

    rows, passed = [], 0
    for case in cases:
        print("=" * 70)
        print(f"{case['id']} [{case['category']}]\n{case['query']}")
        print("=" * 70)
        try:
            result = invoke_with_retry(agent, case["query"])
            verdict = check_case(case, result)
            tools = [str(t.get("tool")) for t in result.get("trace", []) if t.get("tool")]
            row = {
                "id": case["id"], "category": case["category"], "query": case["query"],
                "answer": result.get("answer"), "tools": tools,
                "n_local_sources": len(result.get("sources") or []),
                "n_citations": len(result.get("citations") or []),
                **verdict,
            }
        except Exception as exc:
            traceback.print_exc()
            row = {"id": case["id"], "category": case["category"], "query": case["query"],
                   "error": f"{type(exc).__name__}: {exc}", "checks": {}, "passed": False}
        rows.append(row)
        passed += int(row["passed"])
        print(f"Tools    : {row.get('tools')}")
        print(f"Checks   : {row['checks']}")
        print(f"RESULT   : {'PASS' if row['passed'] else 'FAIL'}"
              + ("   (BACA JAWABAN: manual_review)" if case.get("manual_review") else ""))
        print(f"Jawaban  : {str(row.get('answer') or row.get('error'))[:300]!r}\n")
        time.sleep(args.sleep)

    print("=" * 70)
    print(f"ADVERSARIAL SUMMARY: {passed}/{len(cases)} lulus")
    by_cat: Dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    for cat, vals in sorted(by_cat.items()):
        print(f"  {cat:24s}: {sum(vals)}/{len(vals)}")
    print("=" * 70)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"adversarial_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"passed": passed, "total": len(cases), "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hasil lengkap: {out}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
