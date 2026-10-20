from src.cache import build_query_cache_key
from src.cache import (
    get_cache,
    set_cache,
    delete_cache,
)


def test_cache_set_and_get():
    namespace = "test"
    key = "basic"

    value = {
        "status": "ok",
        "message": "hello redis",
    }

    set_cache(namespace, key, value)

    assert get_cache(namespace, key) == value

    delete_cache(namespace, key)


def test_cache_miss():
    namespace = "test"
    key = "does-not-exist"

    delete_cache(namespace, key)

    assert get_cache(namespace, key) is None


def test_cache_expiry():
    namespace = "test"
    key = "expiry"

    value = {
        "temporary": True,
    }

    set_cache(
        namespace,
        key,
        value,
        ttl=1,
    )

    assert get_cache(namespace, key) == value

    delete_cache(namespace, key)


def test_query_cache_key_is_deterministic():
    key_a = build_query_cache_key(
        "Apa itu UU 1 Tahun 2024?"
    )

    key_b = build_query_cache_key(
        "  apa itu uu 1 tahun 2024?  "
    )

    assert key_a == key_b


def test_query_cache_key_is_query_specific():
    key_a = build_query_cache_key(
        "Apa itu UU 1 Tahun 2024?"
    )

    key_b = build_query_cache_key(
        "Apa itu UU 3 Tahun 2024?"
    )

    assert key_a != key_b
