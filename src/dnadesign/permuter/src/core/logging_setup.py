"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/logging_setup.py

Rich logging configuration for the Permuter CLI.
• Non-protocol logs → stderr (level controlled by -v)
• Protocol logs (permuter.protocol.*) → stdout, pretty, INFO by default

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(verbose: int = 0) -> None:
    """
    Install two Rich handlers:
      1) default handler to stderr for everything EXCEPT 'permuter.protocol.*'
      2) dedicated handler to stdout for 'permuter.protocol.*'

    Levels:
      • default handler: WARNING (no -v), INFO (-v), DEBUG (-vv)
      • protocol handler: INFO by default; DEBUG with -vv
    """
    root = logging.getLogger()
    # Reset any prior basicConfig/handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)  # let handlers decide what to emit

    class _ExcludeProtocols(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not record.name.startswith("permuter.protocol")

    # ---- default (non‑protocol) channel → stderr ----
    default_level = (
        logging.WARNING
        if verbose == 0
        else (logging.INFO if verbose == 1 else logging.DEBUG)
    )
    default_handler = RichHandler(
        console=Console(file=sys.stderr),
        show_time=False,
        show_level=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    default_handler.setLevel(default_level)
    default_handler.addFilter(_ExcludeProtocols())
    root.addHandler(default_handler)

    # ---- protocol channel → stdout (pretty, INFO by default) ----
    class _DedupFilter(logging.Filter):
        """Drop immediately repeated identical log messages on the same handler."""

        def __init__(self):
            super().__init__()
            self._last: str = ""

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if msg == self._last:
                return False
            self._last = msg
            return True

    proto_handler = RichHandler(
        console=Console(file=sys.stdout),
        show_time=False,
        show_level=False,
        show_path=False,
        rich_tracebacks=False,
        markup=True,
    )
    proto_handler.addFilter(_DedupFilter())
    proto_logger = logging.getLogger("permuter.protocol")
    proto_logger.handlers.clear()
    proto_logger.setLevel(logging.DEBUG)  # accept all; handler filters
    proto_logger.propagate = False  # avoid duplicate emission via root
    proto_logger.addHandler(proto_handler)

    # Quiet noisy third‑party libs unless verbose
    for name in ("matplotlib", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(
            logging.WARNING if verbose == 0 else logging.INFO
        )
