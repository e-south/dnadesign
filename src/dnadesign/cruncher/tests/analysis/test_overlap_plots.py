"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_overlap_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dnadesign.baserender import cruncher_showcase_style_overrides
from dnadesign.cruncher.analysis.plots.elites_showcase import (
    _overlay_text,
    build_elites_showcase_records,
    plot_elites_showcase,
)
from dnadesign.cruncher.core.pwm import PWM


def test_elites_showcase_plot_smoke(tmp_path) -> None:
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC", "norm_tfA": 0.91, "norm_tfB": 0.72},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA", "norm_tfA": 0.88, "norm_tfB": 0.69},
            {"id": "elite-3", "rank": 3, "sequence": "AACCGAAT", "norm_tfA": 0.77, "norm_tfB": 0.64},
        ]
    )
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-1", "tf": "tfB", "best_start": 4, "pwm_width": 4, "best_strand": "-"},
            {"elite_id": "elite-2", "tf": "tfA", "best_start": 1, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfB", "best_start": 3, "pwm_width": 4, "best_strand": "-"},
            {"elite_id": "elite-3", "tf": "tfA", "best_start": 2, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-3", "tf": "tfB", "best_start": 4, "pwm_width": 4, "best_strand": "+"},
        ]
    )
    pwms = {
        "tfA": PWM(
            name="tfA",
            matrix=np.array(
                [
                    [0.8, 0.1, 0.05, 0.05],
                    [0.1, 0.8, 0.05, 0.05],
                    [0.05, 0.05, 0.8, 0.1],
                    [0.05, 0.05, 0.1, 0.8],
                ],
                dtype=float,
            ),
        ),
        "tfB": PWM(
            name="tfB",
            matrix=np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.7, 0.1, 0.1, 0.1],
                ],
                dtype=float,
            ),
        ),
    }

    panel_path = tmp_path / "elites_showcase.png"
    plot_elites_showcase(
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=["tfA", "tfB"],
        pwms=pwms,
        out_path=panel_path,
        max_panels=12,
        dpi=150,
        png_compress_level=9,
    )
    assert panel_path.exists()


def test_elites_showcase_reverse_hits_match_antisense_positioning() -> None:
    elites_df = pd.DataFrame([{"id": "elite-1", "rank": 1, "sequence": "ATACAGTT", "norm_tfA": 0.91}])
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 6, "best_strand": "-"},
        ]
    )
    matrix = np.array(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.05, 0.60, 0.30, 0.05],
            [0.20, 0.10, 0.10, 0.60],
            [0.05, 0.20, 0.70, 0.05],
            [0.65, 0.10, 0.20, 0.05],
            [0.10, 0.75, 0.10, 0.05],
        ],
        dtype=float,
    )
    pwms = {"tfA": PWM(name="tfA", matrix=matrix)}

    records = build_elites_showcase_records(
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=["tfA"],
        pwms=pwms,
        max_panels=12,
    )
    assert len(records) == 1
    record = records[0]
    assert len(record.features) == 1
    feature = record.features[0]
    assert feature.span.strand == "rev"
    assert feature.label == "CTGTAT"
    assert len(record.effects) == 1
    effect = record.effects[0]
    assert effect.kind == "motif_logo"
    assert effect.target == {"feature_id": feature.id}
    assert np.allclose(np.asarray(effect.params["matrix"], dtype=float), np.asarray(matrix, dtype=float))


def test_elites_showcase_style_uses_match_window_coloring() -> None:
    overrides = cruncher_showcase_style_overrides()
    motif_logo = dict(overrides.get("motif_logo", {}))
    letter_coloring = dict(motif_logo.get("letter_coloring", {}))
    assert str(letter_coloring.get("mode")) == "match_window_seq"


def test_elites_showcase_fails_fast_when_panel_limit_exceeded(tmp_path) -> None:
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC", "norm_tfA": 0.91},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA", "norm_tfA": 0.88},
        ]
    )
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
        ]
    )
    pwms = {"tfA": PWM(name="tfA", matrix=np.array([[0.25, 0.25, 0.25, 0.25]] * 4, dtype=float))}

    panel_path = tmp_path / "elites_showcase.png"
    try:
        plot_elites_showcase(
            elites_df=elites_df,
            hits_df=hits_df,
            tf_names=["tfA"],
            pwms=pwms,
            out_path=panel_path,
            max_panels=1,
            dpi=150,
            png_compress_level=9,
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "analysis.elites_showcase.max_panels" in str(exc)
    assert raised


def test_elites_showcase_overlay_title_is_succinct_and_ranked() -> None:
    row = pd.Series(
        {
            "id": "elite_showcase_1234567890",
            "rank": 12,
            "sequence": "ACGTACGTACGT",
            "norm_tfA": 0.91234,
            "norm_tfB": 0.66789,
        }
    )
    text = _overlay_text(row, tf_names=["tfA", "tfB"], max_chars=80)
    expected_hash = hashlib.sha256("elite_showcase_1234567890|ACGTACGTACGT".encode("utf-8")).hexdigest()[:12]
    assert len(text) <= 80
    assert text.startswith(f"Elite #12 [{expected_hash}]\n")
    assert text.count("\n") == 1
    score_line = text.splitlines()[1]
    assert "tfA=0.91" in score_line
    assert "tfB=0.67" in score_line


def test_elites_showcase_overlay_title_prefers_explicit_hash_id() -> None:
    row = pd.Series({"id": "elite_showcase_1", "rank": 1, "hash_id": "abc123def456", "norm_tfA": 0.7})
    text = _overlay_text(row, tf_names=["tfA"], max_chars=80)
    assert text.startswith("Elite #1 [abc123def456]")


def test_elites_showcase_overlay_title_prefers_normalized_score_columns() -> None:
    row = pd.Series(
        {
            "id": "elite_showcase_1",
            "rank": 1,
            "norm_tfA": 0.31,
            "score_tfA": 7.77,
        }
    )
    text = _overlay_text(row, tf_names=["tfA"], max_chars=80)
    assert "tfA=0.31" in text


def test_elites_showcase_overlay_title_fails_when_norm_score_out_of_bounds() -> None:
    row = pd.Series(
        {
            "id": "elite_showcase_1",
            "rank": 1,
            "norm_tfA": 1.2,
        }
    )
    with pytest.raises(ValueError, match="must be normalized in \\[0,1\\]"):
        _overlay_text(row, tf_names=["tfA"], max_chars=80)


def test_elites_showcase_overlay_title_handles_missing_rank() -> None:
    row = pd.Series({"id": "elite_showcase_1"})
    text = _overlay_text(row, max_chars=40)
    assert len(text) <= 40
    assert text == "Elite"


def test_elites_showcase_uses_single_row_layout(monkeypatch, tmp_path) -> None:
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC", "norm_tfA": 0.91},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA", "norm_tfA": 0.88},
            {"id": "elite-3", "rank": 3, "sequence": "AACCGAAT", "norm_tfA": 0.77},
        ]
    )
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-3", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
        ]
    )
    pwms = {"tfA": PWM(name="tfA", matrix=np.array([[0.25, 0.25, 0.25, 0.25]] * 4, dtype=float))}

    seen: dict[str, int] = {}

    def _fake_grid(records, *, ncols, style_overrides):
        seen["ncols"] = int(ncols)
        seen["nrecords"] = len(list(records))
        seen["style_overrides"] = 1 if isinstance(style_overrides, dict) else 0
        return plt.figure(figsize=(2, 2), dpi=100)

    monkeypatch.setattr(
        "dnadesign.cruncher.analysis.plots.elites_showcase.render_record_grid_figure",
        _fake_grid,
    )

    panel_path = tmp_path / "elites_showcase.png"
    plot_elites_showcase(
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=["tfA"],
        pwms=pwms,
        out_path=panel_path,
        max_panels=12,
        dpi=120,
        png_compress_level=9,
    )

    assert panel_path.exists()
    assert seen["nrecords"] == 3
    assert seen["ncols"] == 3
    assert seen["style_overrides"] == 1


def test_elites_showcase_fails_when_tf_norm_columns_missing(tmp_path) -> None:
    elites_df = pd.DataFrame([{"id": "elite-1", "rank": 1, "sequence": "AAAACCCC"}])
    hits_df = pd.DataFrame([{"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"}])
    pwms = {"tfA": PWM(name="tfA", matrix=np.array([[0.25, 0.25, 0.25, 0.25]] * 4, dtype=float))}

    panel_path = tmp_path / "elites_showcase.png"
    with pytest.raises(ValueError, match="elites_df missing required columns"):
        plot_elites_showcase(
            elites_df=elites_df,
            hits_df=hits_df,
            tf_names=["tfA"],
            pwms=pwms,
            out_path=panel_path,
            max_panels=12,
            dpi=120,
            png_compress_level=9,
        )
