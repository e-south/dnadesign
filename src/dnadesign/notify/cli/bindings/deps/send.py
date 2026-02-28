"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/deps/send.py

Send-domain dependency exports and helper adapters for notify CLI bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....delivery.http import post_json
from ....delivery.payload import build_payload
from ....errors import NotifyError
from ...handlers import run_send_command
from .. import helpers


def _load_meta(meta_path: Path | None) -> dict[str, Any]:
    return helpers.load_meta(
        meta_path,
        notify_error_cls=NotifyError,
    )
