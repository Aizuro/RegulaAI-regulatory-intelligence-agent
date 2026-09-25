import time
from typing import Any, Callable

from src.cache import get_cache, set_cache


def execute_with_cache(
    namespace: str,
    key: str,
    loader: Callable[[], Any],
    ttl: int = 3600,
) -> tuple[Any, bool]:
    cached = get_cache(namespace, key)

    if cached is not None:
        return cached, True

    value = loader()

    set_cache(
        namespace,
        key,
        value,
        ttl=ttl,
    )

    return value, False
