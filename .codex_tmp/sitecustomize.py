import hashlib
import os
import random

_counter = 0


def _deterministic_urandom(n: int) -> bytes:
    global _counter
    if n <= 0:
        return b""
    out = bytearray()
    while len(out) < n:
        _counter += 1
        out.extend(hashlib.sha256(str(_counter).encode("utf-8")).digest())
    return bytes(out[:n])


os.urandom = _deterministic_urandom
# random.py caches os.urandom as random._urandom on import.
random._urandom = _deterministic_urandom
