"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_library_transform.py

Tests for motif library transform that rewrites motif_logo matrices from a
tool-provided motif primitives artifact.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.baserender.src.core import PluginError, Record, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.pipeline.attach_motifs_from_library import AttachMotifsFromLibraryTransform


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


def _write_demo_library(path: Path) -> None:
    payload = {
        "schema_version": "1",
        "alphabet": "DNA",
        "motifs": {
            "lexA": {
                "source": "demo_source",
                "motif_id": "lexA_demo",
                "matrix": [[0.1, 0.8, 0.05, 0.05] for _ in range(15)],
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def test_attach_motifs_from_library_rewrites_effect_matrix(tmp_path: Path) -> None:
    library_path = tmp_path / "motif_library.json"
    _write_demo_library(library_path)

    transform = AttachMotifsFromLibraryTransform(library_path=str(library_path))
    record = _build_single_feature_record(
        label="TGCATATATTTACAG",
        start=1,
        end=16,
        tf_tag="tf:lexA",
    )
    rewritten = transform.apply(record)
    matrix = rewritten.effects[0].params["matrix"]
    assert matrix == [[0.1, 0.8, 0.05, 0.05] for _ in range(15)]


def test_attach_motifs_from_library_errors_on_unknown_top_level_key(tmp_path: Path) -> None:
    library_path = tmp_path / "motif_library.json"
    payload = {
        "schema_version": "1",
        "alphabet": "DNA",
        "motifs": {},
        "unexpected": True,
    }
    library_path.write_text(json.dumps(payload))
    with pytest.raises(PluginError, match="Unknown keys"):
        AttachMotifsFromLibraryTransform(library_path=str(library_path))
