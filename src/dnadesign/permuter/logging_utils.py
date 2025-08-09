"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/logging_utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Compact, readable log format (no module/lineno noise)
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def init_logger(
    level: str | int = "INFO", logfile: str | None = None
) -> logging.Logger:
    """
    Initialize the root logger once with a compact format.

    Args:
      level: "DEBUG", "INFO", "WARNING", "ERROR"
      logfile: optional path to also tee logs to a file
    """
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=lvl,
        format=_LOG_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("permuter")
