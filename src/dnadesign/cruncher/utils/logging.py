"""
--------------------------------------------------------------------------------
<dnadesign project>
utils/logging.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback


def configure_logging(level: str = "INFO") -> None:
    install_rich_traceback(show_locals=False)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)],
    )
