"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/tests/test_meme_parsers.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np

from dnadesign.cruncher.motif.parsers.meme import parse_meme
from dnadesign.cruncher.motif.registry import Registry


def test_registry_load_cached(tmp_path, meme_file):
    # copy sample file into a fake motif root dir
    motif_root = tmp_path / "motifs"
    motif_root.mkdir()
    target = motif_root / meme_file.name
    target.write_bytes(meme_file.read_bytes())

    reg = Registry(motif_root, {".txt": "MEME"})
    tf_name = meme_file.stem  # oxyR / lexA

    pwm1 = reg.load(tf_name)
    pwm2 = reg.load(tf_name.upper())  # case-insensitive
    assert pwm1 is pwm2  # cache hit


def test_parse_real_motif(meme_file):
    """Parse a real MEME file (oxyR / lexA)."""
    pwm = parse_meme(meme_file)

    # basic invariants
    assert pwm.length >= 4  # realistic motif
    np.testing.assert_allclose(pwm.matrix.sum(axis=1), 1, rtol=1e-6)
    # optional: ensure log-odds exists if file contains that block
    if pwm.log_odds_matrix is not None:
        assert pwm.log_odds_matrix.shape == (pwm.length, 4)
