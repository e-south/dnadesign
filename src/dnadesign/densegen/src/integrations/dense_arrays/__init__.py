"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/integrations/dense_arrays/__init__.py

Expose DenseGen integration with public dense-arrays playback contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .playback import (
    playback_plan_from_densegen_record,
    realized_array_from_densegen_record,
)
from .publisher import publish_densegen_playback_endpoint

__all__ = [
    "playback_plan_from_densegen_record",
    "realized_array_from_densegen_record",
    "publish_densegen_playback_endpoint",
]
