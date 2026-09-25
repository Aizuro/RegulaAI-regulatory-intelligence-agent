import redis


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)


def check_redis_connection() -> bool:
    return bool(redis_client.ping())
