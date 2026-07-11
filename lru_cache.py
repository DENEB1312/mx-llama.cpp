from collections import OrderedDict
from typing import Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache:
    """Least Recently Used (LRU) cache with fixed capacity.

    Evicts the least recently accessed entry when the cache is full.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._cache: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        """Return the value for *key*, or None if not present.

        Marks the key as recently used on a hit.
        """
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: K, value: V) -> None:
        """Insert or update *key* with *value*.

        Evicts the least recently used item if at capacity.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)


if __name__ == "__main__":
    cache = LRUCache[str, int](2)
    cache.put("a", 1)
    cache.put("b", 2)
    print(cache.get("a"))  # 1  — "a" is now most recent
    cache.put("c", 3)      # evicts "b"
    print(cache.get("b"))  # None
    print(cache.get("c"))  # 3
