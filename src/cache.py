from typing import Any
import json
import hashlib

from src.redis_client import redis_client


DEFAULT_TTL = 3600


def build_cache_key(namespace: str, key: str) -> str:
    return f"regula:cache:{namespace}:{key}"


def get_cache(namespace: str, key: str) -> Any | None:
    cache_key = build_cache_key(namespace, key)

    value = redis_client.get(cache_key)

    if value is None:
        return None

    return json.loads(value)


def set_cache(
    namespace: str,
    key: str,
    value: Any,
    ttl: int = DEFAULT_TTL,
) -> None:
    cache_key = build_cache_key(namespace, key)

    redis_client.set(
        cache_key,
        json.dumps(value),
        ex=ttl,
    )


def delete_cache(namespace: str, key: str) -> None:
    cache_key = build_cache_key(namespace, key)

    redis_client.delete(cache_key)


def build_query_cache_key(question: str) -> str:
    normalized_question = " ".join(question.lower().split())

    return hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()
