"""Executable module for ``python -m ...decision.opal.densegen_axis_probe``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
