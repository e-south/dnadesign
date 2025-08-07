"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/logging_utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
import sys
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s:" "%(funcName)s:%(lineno)d  -  %(message)s"
)


def init_logger(
    level: str | int = "INFO", logfile: str | None = None
) -> logging.Logger:
    """Initialises root logger in a single place."""
    level = getattr(logging, str(level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)
    return logging.getLogger("permuter")
