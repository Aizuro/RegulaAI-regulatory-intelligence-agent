from src.cached_query import execute_with_cache
from src.cache import delete_cache


def test_first_request_is_cache_miss():
    delete_cache("test", "query-1")
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return {
            "answer": "hasil regulasi",
        }

    result, cache_hit = execute_with_cache(
        namespace="test",
        key="query-1",
        loader=loader,
    )

    assert result == {
        "answer": "hasil regulasi",
    }

    assert cache_hit is False
    assert calls["count"] == 1
    delete_cache("test", "query-1")


def test_second_request_is_cache_hit():
    delete_cache("test", "query-2")
    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return {
            "answer": "hasil regulasi",
        }

    first_result, first_hit = execute_with_cache(
        namespace="test",
        key="query-2",
        loader=loader,
    )

    second_result, second_hit = execute_with_cache(
        namespace="test",
        key="query-2",
        loader=loader,
    )

    assert first_result == second_result

    assert first_hit is False
    assert second_hit is True

    assert calls["count"] == 1
    delete_cache("test", "query-2")
