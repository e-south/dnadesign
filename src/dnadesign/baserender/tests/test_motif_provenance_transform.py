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


def _make_matrix(length: int) -> list[list[float]]:
    return [[0.10, 0.10, 0.70, 0.10] for _ in range(length)]


def _write_config_used(tmp_path: Path, *, tf_name: str, matrix: list[list[float]]) -> Path:
    path = tmp_path / "config_used.yaml"
    payload = {
        "cruncher": {
            "pwms_info": {
                tf_name: {
                    "pwm_matrix": matrix,
                }
            }
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _write_lockfile_and_store(
    tmp_path: Path,
    *,
    tf_name: str,
    source: str,
    motif_id: str,
    checksum: str,
    matrix: list[list[float]],
) -> tuple[Path, Path]:
    store_root = tmp_path / "motif_store"
    motif_path = store_root / "normalized" / "motifs" / source / f"{motif_id}.json"
    motif_path.parent.mkdir(parents=True, exist_ok=True)
    motif_path.write_text(
        json.dumps(
            {
                "matrix": matrix,
                "checksums": {"sha256_norm": checksum},
            }
        )
    )
    lockfile_path = tmp_path / "lockfile.json"
    lockfile_path.write_text(
        json.dumps(
            {
                "resolved": {
                    tf_name: {
                        "source": source,
                        "motif_id": motif_id,
                        "sha256": checksum,
                    }
                }
            }
        )
    )
    return lockfile_path, store_root


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


def test_attach_motifs_from_cruncher_lockfile_rewrites_effect_matrix(tmp_path: Path) -> None:
    expected_lexa = _make_matrix(15)
    lockfile_path, store_root = _write_lockfile_and_store(
        tmp_path,
        tf_name="lexA",
        source="demo",
        motif_id="lexa_demo",
        checksum="sha256-demo-lexa",
        matrix=expected_lexa,
    )
    transform = AttachMotifsFromCruncherLockfileTransform(
        lockfile_path=str(lockfile_path),
        motif_store_root=str(store_root),
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
    lockfile_path, store_root = _write_lockfile_and_store(
        tmp_path,
        tf_name="lexA",
        source="demo",
        motif_id="lexa_demo",
        checksum="sha256-demo-lexa",
        matrix=_make_matrix(15),
    )
    payload = json.loads(lockfile_path.read_text())
    payload["resolved"]["lexA"]["sha256"] = "deadbeef"
    bad_lockfile = tmp_path / "lockfile_bad.json"
    bad_lockfile.write_text(json.dumps(payload))

    with pytest.raises(PluginError, match="checksum mismatch"):
        AttachMotifsFromCruncherLockfileTransform(
            lockfile_path=str(bad_lockfile),
            motif_store_root=str(store_root),
        )


def test_attach_motifs_from_config_rewrites_effect_matrix(tmp_path: Path) -> None:
    config_path = _write_config_used(tmp_path, tf_name="lexA", matrix=_make_matrix(15))
    config = yaml.safe_load(config_path.read_text())
    expected_lexa = config["cruncher"]["pwms_info"]["lexA"]["pwm_matrix"]
    transform = AttachMotifsFromConfigTransform(config_path=str(config_path))
    record = _build_single_feature_record(
        label="TGCATATATTTACAG",
        start=1,
        end=16,
        tf_tag="tf:lexA",
    )

    rewritten = transform.apply(record)
    matrix = rewritten.effects[0].params["matrix"]
    assert matrix == expected_lexa


def test_attach_motifs_from_config_errors_on_missing_tf_matrix(tmp_path: Path) -> None:
    config_path = _write_config_used(tmp_path, tf_name="lexA", matrix=_make_matrix(15))
    transform = AttachMotifsFromConfigTransform(config_path=str(config_path))
    record = _build_single_feature_record(
        label="TGCATATATTTACAG",
        start=1,
        end=16,
        tf_tag="tf:not_in_library",
    )

    with pytest.raises(PluginError, match="no motif matrix found"):
        transform.apply(record)


def test_attach_motifs_from_config_errors_on_length_mismatch(tmp_path: Path) -> None:
    config_path = _write_config_used(tmp_path, tf_name="lexA", matrix=_make_matrix(15))
    transform = AttachMotifsFromConfigTransform(config_path=str(config_path))
    record = _build_single_feature_record(
        label="GCATATATTT",
        start=2,
        end=12,
        tf_tag="tf:lexA",
    )

    with pytest.raises(PluginError, match="motif length mismatch"):
        transform.apply(record)
