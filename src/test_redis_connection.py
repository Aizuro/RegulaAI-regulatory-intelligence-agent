from src.redis_client import redis_client, check_redis_connection


def test_redis_connection():
    assert check_redis_connection() is True


def test_redis_set_get():
    key = "regula:test:redis"
    value = "hello-regula"

    redis_client.set(key, value)

    assert redis_client.get(key) == value

    redis_client.delete(key)


def test_redis_ttl():
    key = "regula:test:ttl"

    redis_client.set(key, "temporary", ex=10)

    ttl = redis_client.ttl(key)

    assert 0 < ttl <= 10

    redis_client.delete(key)
