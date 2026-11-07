import os

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")

_COMMON = {
    "decode_responses": True,
    "socket_connect_timeout": 2,
    "socket_timeout": 2,
    "health_check_interval": 30,
}

if REDIS_URL:
    redis_client = redis.Redis.from_url(REDIS_URL, **_COMMON)
else:
    redis_client = redis.Redis(host="localhost", port=6379, **_COMMON)


def check_redis_connection() -> bool:
    try:
        return bool(redis_client.ping())
    except Exception:
        return False
