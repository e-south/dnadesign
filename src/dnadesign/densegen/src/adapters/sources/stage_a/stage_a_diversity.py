"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a/stage_a_diversity.py

Re-exports Stage-A module 'stage_a_diversity' from core.stage_a.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from ....core.stage_a import stage_a_diversity as _impl

for _name, _value in _impl.__dict__.items():
    if _name.startswith("__") and _name != "__all__":
        continue
    globals()[_name] = _value

__all__ = getattr(_impl, "__all__", [])
