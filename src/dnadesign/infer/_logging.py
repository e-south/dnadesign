"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/logging.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys

try:
    from rich.logging import RichHandler
except Exception:
    RichHandler = None  # type: ignore

_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_console_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure root logger for CLI. Library code should still use get_logger()."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level.upper())

    if json_logs:

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                import json as _json

                payload = {
                    "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                }
                return _json.dumps(payload)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level.upper())
        sh.setFormatter(JsonFormatter())
        root.addHandler(sh)
        return

    if RichHandler is not None:
        handler = RichHandler(
            show_time=True, show_level=True, markup=True, rich_tracebacks=False
        )
        handler.setLevel(level.upper())
        root.addHandler(handler)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level.upper())
        sh.setFormatter(logging.Formatter(_DEF_FMT))
        root.addHandler(sh)


def get_logger(name: str = "dnadesign.infer", level: str = "INFO") -> logging.Logger:
    """Library-friendly logger (idempotent handler attach)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level.upper())
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level.upper())
        ch.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(ch)
    return logger
