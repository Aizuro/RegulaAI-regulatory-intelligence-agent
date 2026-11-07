import redis
from src import cache


def test_get_cache_returns_none_when_redis_down(monkeypatch):
    def boom(*args, **kwargs):
        raise redis.ConnectionError("down")

    monkeypatch.setattr(cache.redis_client, "get", boom)
    assert cache.get_cache("query", "abc") is None
