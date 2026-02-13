"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_provenance_transform.py

Tests for motif provenance transform that sources PWM matrices from Cruncher config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.baserender.src.core import PluginError, Record, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.pipeline.attach_motifs_from_config import AttachMotifsFromConfigTransform
from dnadesign.baserender.src.pipeline.attach_motifs_from_cruncher_lockfile import (
    AttachMotifsFromCruncherLockfileTransform,
)


def _demo_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "workspaces" / "demo_cruncher_render" / "inputs" / "config_used.yaml"
    )


def _demo_lockfile_path() -> Path:
    return Path(__file__).resolve().parent.parent / "workspaces" / "demo_cruncher_render" / "inputs" / "lockfile.json"


def _demo_motif_store_root() -> Path:
    return Path(__file__).resolve().parent.parent / "workspaces" / "demo_cruncher_render" / "inputs" / "motif_store"


def _build_single_feature_record(*, label: str, start: int, end: int, tf_tag: str) -> Record:
    return Record(
        id="elite",
        alphabet="DNA",
        sequence="CTGCATATATTTACAG",
        features=(
            Feature(
                id="f1",
                kind="kmer",
                span=Span(start=start, end=end, strand="fwd"),
                label=label,
                tags=(tf_tag,),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={"matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(end - start)]},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )


def test_attach_motifs_from_cruncher_lockfile_rewrites_effect_matrix() -> None:
    lockfile = json.loads(_demo_lockfile_path().read_text())
    lexa_ref = lockfile["resolved"]["lexA"]
    motif_path = (
        _demo_motif_store_root() / "normalized" / "motifs" / str(lexa_ref["source"]) / f"{lexa_ref['motif_id']}.json"
    )
    expected_lexa = json.loads(motif_path.read_text())["matrix"]
    transform = AttachMotifsFromCruncherLockfileTransform(
        lockfile_path=str(_demo_lockfile_path()),
        motif_store_root=str(_demo_motif_store_root()),
    )
    record = _build_single_feature_record(
        label="TGCATATATTTACAG",
        start=1,
        end=16,
        tf_tag="tf:lexA",
    )

    rewritten = transform.apply(record)
    matrix = rewritten.effects[0].params["matrix"]
    assert matrix == expected_lexa


def test_attach_motifs_from_cruncher_lockfile_errors_on_checksum_mismatch(tmp_path: Path) -> None:
    payload = json.loads(_demo_lockfile_path().read_text())
    payload["resolved"]["lexA"]["sha256"] = "deadbeef"
    bad_lockfile = tmp_path / "lockfile_bad.json"
    bad_lockfile.write_text(json.dumps(payload))

    with pytest.raises(PluginError, match="checksum mismatch"):
        AttachMotifsFromCruncherLockfileTransform(
            lockfile_path=str(bad_lockfile),
            motif_store_root=str(_demo_motif_store_root()),
        )


def test_attach_motifs_from_config_rewrites_effect_matrix() -> None:
    config = yaml.safe_load(_demo_config_path().read_text())
    expected_lexa = config["cruncher"]["pwms_info"]["lexA"]["pwm_matrix"]
    transform = AttachMotifsFromConfigTransform(config_path=str(_demo_config_path()))
    record = _build_single_feature_record(
        label="TGCATATATTTACAG",
        start=1,
        end=16,
        tf_tag="tf:lexA",
    )

    rewritten = transform.apply(record)
    matrix = rewritten.effects[0].params["matrix"]
    assert matrix == expected_lexa


def test_attach_motifs_from_config_errors_on_missing_tf_matrix() -> None:
    transform = AttachMotifsFromConfigTransform(config_path=str(_demo_config_path()))
    record = _build_single_feature_record(
        label="TGCATATATTTACAG",
        start=1,
        end=16,
        tf_tag="tf:not_in_library",
    )

    with pytest.raises(PluginError, match="no motif matrix found"):
        transform.apply(record)


def test_attach_motifs_from_config_errors_on_length_mismatch() -> None:
    transform = AttachMotifsFromConfigTransform(config_path=str(_demo_config_path()))
    record = _build_single_feature_record(
        label="GCATATATTT",
        start=2,
        end=12,
        tf_tag="tf:lexA",
    )

    with pytest.raises(PluginError, match="motif length mismatch"):
        transform.apply(record)
