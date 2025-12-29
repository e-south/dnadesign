"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_sigma70_plugin_strict.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.contracts import PluginError
from dnadesign.baserender.src.model import SeqRecord
from dnadesign.baserender.src.plugins.builtin.sigma70 import Sigma70Plugin


def _record_with_seq(seq: str) -> SeqRecord:
    return SeqRecord(id="rec", alphabet="DNA", sequence=seq, annotations=(), guides=()).validate()


def test_sigma70_invalid_label_mode_raises():
    with pytest.raises(PluginError):
        Sigma70Plugin(label_mode="bogus")


def test_sigma70_multiple_matches_default_raises():
    motif = "TTGACA" + ("A" * 17) + "TATAAT"
    seq = motif + "GGGGGG" + motif
    rec = _record_with_seq(seq)
    plugin = Sigma70Plugin()
    with pytest.raises(PluginError):
        plugin.apply(rec)


def test_sigma70_multiple_matches_first_allowed_when_configured():
    motif = "TTGACA" + ("A" * 17) + "TATAAT"
    seq = motif + "GGGGGG" + motif
    rec = _record_with_seq(seq)
    plugin = Sigma70Plugin(on_multiple_matches="first")
    out = plugin.apply(rec)
    assert any(a.tag == "sigma" for a in out.annotations)
