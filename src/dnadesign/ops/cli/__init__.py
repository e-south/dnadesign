"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/__init__.py

Public CLI entrypoint module for the installed OPS console script.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Sequence


class _AppProxy:
    def _load(self):
        from .app import app as real_app

        return real_app

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


app = _AppProxy()


def main(argv: Sequence[str] | None = None) -> int:
    from .app import main as _main

    stderr_fd = os.dup(2)
    try:
        return _main(argv, stderr_fd=stderr_fd)
    finally:
        os.close(stderr_fd)


__all__ = ["app", "main"]
