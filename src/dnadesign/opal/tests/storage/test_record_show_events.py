"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/storage/test_record_show_events.py

Regression tests for record show events OPAL storage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from dnadesign.opal.src.reporting.record_show import build_record_report


def test_record_show_requires_ledger_reader():
    rec = pd.DataFrame({"id": ["x"], "sequence": ["AC"], "bio_type": ["dna"], "alphabet": ["dna_4"]})
    with pytest.raises(ValueError):
        build_record_report(rec, "demo", id_="x", ledger_reader=None, selection_view_id="primary")
