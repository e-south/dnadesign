"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/normalize.py

Compatibility wrapper for YIU v4 normalization.
Prefer `dnadesign.cruncher.yiu.normalizer`.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.yiu.normalizer import aligned_complement_3to5, normalize_payload

__all__ = ["aligned_complement_3to5", "normalize_payload"]
