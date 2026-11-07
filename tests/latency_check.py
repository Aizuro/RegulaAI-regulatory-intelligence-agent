import time
import statistics
import requests

from src.redis_client import redis_client
from src.cache import build_cache_key, build_query_cache_key

URL = "https://regulaai-regulatory-intelligence-agent.fastapicloud.dev/query"
Q = {"question": "Apa ketentuan Pasal 17 ayat (2a) UU Nomor 1 Tahun 2024?"}


def timed(fn):
    t = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t) * 1000


def summary(label, values):
    values = sorted(values)
    p95 = values[int(len(values) * 0.95) - 1]
    print(f"{label:<28} median {statistics.median(values):6.0f} ms | "
          f"min {values[0]:.0f} | p95 {p95:.0f} | max {values[-1]:.0f}")


session = requests.Session()
key = build_cache_key("query", build_query_cache_key(Q["question"]))

# 1. RTT mentah ke Redis (jaringan saja)
summary("PING ke Redis", [timed(redis_client.ping)[1] for _ in range(30)])

# 2. Cache miss: hapus kunci, lalu panggil API (dua kali supaya terlihat efek warm-up)
for label in ("MISS pertama (termasuk init)", "MISS hangat"):
    redis_client.delete(key)
    _, ms = timed(lambda: session.post(URL, json=Q, timeout=120))
    print(f"{label:<28} {ms:.0f} ms")

# 3. Cache hit berulang
hits = [timed(lambda: session.post(URL, json=Q, timeout=120))[1]
        for _ in range(30)]
summary("HIT", hits)
