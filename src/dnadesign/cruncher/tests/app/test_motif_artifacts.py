"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_motif_artifacts.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.app.motif_artifacts import build_densegen_artifact


def test_build_densegen_artifact_prefers_log_odds_matrix() -> None:
    payload = {
        "descriptor": {"motif_id": "m1", "alphabet": "ACGT"},
        "matrix_semantics": "probabilities",
        "matrix": [
            [0.0, 1.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        "log_odds_matrix": [
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "background": [0.25, 0.25, 0.25, 0.25],
    }

    artifact = build_densegen_artifact(
        payload,
        producer="cruncher",
        background_policy="record",
        pseudocount=None,
    )

    assert artifact["log_odds"][0]["A"] == 0.1
    assert artifact["log_odds"][0]["C"] == 0.2
