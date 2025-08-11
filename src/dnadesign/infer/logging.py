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

_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "dnadesign.infer", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level.upper())
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level.upper())
        ch.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(ch)
    return logger
