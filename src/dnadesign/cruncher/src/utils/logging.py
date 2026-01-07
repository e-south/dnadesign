"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/utils/logging.py

Author(s): Eric J. South
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
    for noisy_logger in ("arviz", "arviz_base", "arviz_stats", "arviz_plots"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
