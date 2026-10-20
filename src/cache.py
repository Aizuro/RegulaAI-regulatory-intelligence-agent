from typing import Any
import json
import hashlib
import logging

from src.redis_client import redis_client


logger = logging.getLogger(__name__)

DEFAULT_TTL = 3600


def build_cache_key(namespace: str, key: str) -> str:
    return f"regula:cache:{namespace}:{key}"


def get_cache(namespace: str, key: str) -> Any | None:
    cache_key = build_cache_key(namespace, key)

    try:
        value = redis_client.get(cache_key)
        if value is None:
            return None
        return json.loads(value)
    except Exception as e:
        logger.warning("get_cache gagal, lanjut tanpa cache: %s", e)
        return None


def set_cache(
    namespace: str,
    key: str,
    value: Any,
    ttl: int = DEFAULT_TTL,
) -> None:
    cache_key = build_cache_key(namespace, key)

    try:
        redis_client.set(cache_key, json.dumps(value), ex=ttl)
    except Exception as e:
        logger.warning("set_cache gagal: %s", e)


def delete_cache(namespace: str, key: str) -> None:
    cache_key = build_cache_key(namespace, key)

    try:
        redis_client.delete(cache_key)
    except Exception as e:
        logger.warning("delete_cache gagal: %s", e)


def build_query_cache_key(question: str) -> str:
    normalized_question = " ".join(question.lower().split())

    return hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()
