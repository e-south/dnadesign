"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/package/test_logging_policy.py

Contract tests for infer logging policy defaults.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging

from dnadesign.infer.src import _logging as logging_policy


def test_setup_console_logging_suppresses_third_party_info_noise_at_info_level() -> None:
    root = logging.getLogger()
    original_root_level = root.level
    original_root_handlers = list(root.handlers)
    logger_names = ("httpx", "huggingface_hub", "StripedHyena", "vortex.model.utils")
    original_logger_levels = {name: logging.getLogger(name).level for name in logger_names}

    try:
        logging_policy.setup_console_logging(level="INFO", json_logs=False)

        for name in logger_names:
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in original_root_handlers:
            root.addHandler(handler)
        root.setLevel(original_root_level)
        for name, level in original_logger_levels.items():
            logging.getLogger(name).setLevel(level)


def test_setup_console_logging_rehomes_infer_library_loggers_to_root_handlers() -> None:
    root = logging.getLogger()
    original_root_level = root.level
    original_root_handlers = list(root.handlers)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logger = logging.getLogger("dnadesign.infer.src.audit")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger = logging_policy.get_logger("dnadesign.infer.src.audit")
    original_logger_level = logger.level
    original_logger_handlers = list(logger.handlers)
    original_logger_propagate = logger.propagate

    try:
        assert logger.handlers

        logging_policy.setup_console_logging(level="INFO", json_logs=False)

        assert logger.handlers == []
        assert logger.propagate is True
        assert logger.level == logging.NOTSET
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in original_root_handlers:
            root.addHandler(handler)
        root.setLevel(original_root_level)

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in original_logger_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_logger_level)
        logger.propagate = original_logger_propagate
