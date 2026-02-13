"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_provenance_transform.py

Tests for motif provenance transform that sources PWM matrices from Cruncher config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.baserender.src.core import PluginError, Record, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.pipeline.attach_motifs_from_config import AttachMotifsFromConfigTransform


def _demo_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "workspaces" / "demo_cruncher_render" / "inputs" / "config_used.yaml"
    )


def test_attach_motifs_from_config_rewrites_effect_matrix() -> None:
    config = yaml.safe_load(_demo_config_path().read_text())
    expected_lexa = config["cruncher"]["pwms_info"]["lexA"]["pwm_matrix"]
    transform = AttachMotifsFromConfigTransform(config_path=str(_demo_config_path()))
    record = Record(
        id="elite",
        alphabet="DNA",
        sequence="CTGCATATATTTACAG",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=1, end=16, strand="fwd"),
                label="TGCATATATTTACAG",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={"matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(15)]},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    rewritten = transform.apply(record)
    matrix = rewritten.effects[0].params["matrix"]
    assert matrix == expected_lexa


def test_attach_motifs_from_config_errors_on_missing_tf_matrix() -> None:
    transform = AttachMotifsFromConfigTransform(config_path=str(_demo_config_path()))
    record = Record(
        id="elite",
        alphabet="DNA",
        sequence="CTGCATATATTTACAG",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=1, end=16, strand="fwd"),
                label="TGCATATATTTACAG",
                tags=("tf:not_in_library",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={"matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(15)]},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    with pytest.raises(PluginError, match="no motif matrix found"):
        transform.apply(record)


def test_attach_motifs_from_config_errors_on_length_mismatch() -> None:
    transform = AttachMotifsFromConfigTransform(config_path=str(_demo_config_path()))
    record = Record(
        id="elite",
        alphabet="DNA",
        sequence="CTGCATATATTTACAG",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=2, end=12, strand="fwd"),
                label="GCATATATTT",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={"matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(10)]},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    with pytest.raises(PluginError, match="motif length mismatch"):
        transform.apply(record)
