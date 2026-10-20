import os

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    socket_connect_timeout=1,
    socket_timeout=1,
)


def check_redis_connection() -> bool:
    try:
        return bool(redis_client.ping())
    except Exception:
        return False
