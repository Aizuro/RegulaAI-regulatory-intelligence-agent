import redis


redis_client = redis.Redis(
    REDIS_URL,
    socket_connect_timeout=1,
    socket_timeout=1,
)


def check_redis_connection() -> bool:
    return bool(redis_client.ping())
