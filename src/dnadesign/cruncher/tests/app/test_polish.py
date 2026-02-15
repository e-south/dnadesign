"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_polish.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dnadesign.cruncher.app.sample_workflow as sample_workflow


def test_polish_helper_removed() -> None:
    assert not hasattr(sample_workflow, "_polish_sequence")
