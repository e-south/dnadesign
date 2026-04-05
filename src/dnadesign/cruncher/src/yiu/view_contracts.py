"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_contracts.py

Compatibility facade for YIU view-contract builders shared by publish and
integrity.
Prefer the specialized `view_payload_contracts.py` and
`view_sequence_contracts.py` modules for new internal imports.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.yiu.view_common import YIU_EMPTY_ROW_LABELS
from dnadesign.cruncher.yiu.view_payload_contracts import build_payload_view_contract
from dnadesign.cruncher.yiu.view_sequence_contracts import (
    build_assembled_payload_view_contract,
    build_split_payload_view_rows,
)
from dnadesign.cruncher.yiu.view_styles import build_payload_view_title, build_yiu_style_overrides

__all__ = [
    "YIU_EMPTY_ROW_LABELS",
    "build_assembled_payload_view_contract",
    "build_payload_view_contract",
    "build_payload_view_title",
    "build_split_payload_view_rows",
    "build_yiu_style_overrides",
]
