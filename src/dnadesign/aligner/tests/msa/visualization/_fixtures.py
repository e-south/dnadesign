"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/msa/visualization/_fixtures.py

Shared fixtures for MSA visualization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from dnadesign.aligner.msa import write_fasta_records

TARGET = "ACDEFGHIKLMNPQRSTVWY"


def write_alignment_inputs(tmp_path: Path, *, profile_ids: tuple[str, ...]) -> Path:
    """Write tiny aligned FASTA profiles for visualization tests."""

    root = tmp_path / "alignments"
    root.mkdir()
    target_aligned = TARGET[:10] + "-" + TARGET[10:]
    homolog_one = target_aligned[:3] + "A" + target_aligned[4:]
    homolog_two = target_aligned[:12] + "-" + target_aligned[13:]
    for profile_id in profile_ids:
        write_fasta_records(
            root / f"{profile_id}.aligned.fasta",
            {
                "target": target_aligned,
                f"{profile_id}_homolog_01": homolog_one,
                f"{profile_id}_homolog_02": homolog_two,
            },
        )
    return root


def target_hash(sequence: str = TARGET) -> str:
    """Return the hash shape used by target-row contracts."""

    return "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def write_annotation_tracks(
    tmp_path: Path,
    *,
    feature_end: int = 16,
    fill_opacity: float = 0.25,
    label_position: str = "auto",
) -> Path:
    """Write a generic annotation-track fixture."""

    path = tmp_path / "annotation_tracks.yaml"
    payload = {
        "schema_id": "dnadesign.aligner.msa.visualization.annotation_tracks",
        "schema_version": 1,
        "coordinate_space": "target_ungapped_position",
        "tracks": [
            {
                "id": "motifs",
                "label": "Motifs",
                "color": "#5b5f97",
                "features": [
                    {
                        "id": "motif_a",
                        "label": "Motif A",
                        "start": 4,
                        "end": 6,
                        "color": "#d95f02",
                        "fill_opacity": fill_opacity,
                        "stroke_color": "#d95f02",
                        "stroke_width": 2,
                        "text_color": "#d95f02",
                        "label_position": label_position,
                    },
                    {
                        "id": "region_b",
                        "label": "Region B",
                        "start": 14,
                        "end": feature_end,
                        "color": "#1b9e77",
                    },
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def write_exemplar_rows(
    tmp_path: Path,
    *,
    missing: bool = False,
    profile_ids: tuple[str, ...] = ("profile_a", "profile_b"),
) -> Path:
    """Write profile-scoped exemplar-row selections."""

    path = tmp_path / "exemplar_rows.yaml"
    second_record = "missing_record" if missing else "profile_a_homolog_01"
    profiles = {}
    if "profile_a" in profile_ids:
        profiles["profile_a"] = {
            "rows": [
                {
                    "record_id": "target",
                    "label": "Reference target",
                    "group": "target",
                },
                {
                    "record_id": second_record,
                    "label": "Homolog one",
                    "group": "example",
                },
            ],
        }
    if "profile_b" in profile_ids:
        profiles["profile_b"] = {
            "rows": [
                {
                    "record_id": "target",
                    "label": "Reference target",
                    "group": "target",
                },
                {
                    "record_id": "profile_b_homolog_01",
                    "label": "Homolog one",
                    "group": "example",
                },
            ],
        }
    payload = {
        "schema_id": "dnadesign.aligner.msa.visualization.exemplar_rows",
        "schema_version": 1,
        "profiles": profiles,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def write_panel_spec(
    tmp_path: Path,
    *,
    high_gap_trim_threshold: float = 0.9,
    row_source: str = "exemplar_rows",
    max_display_rows: int | str = 4,
    profiles: dict[str, object] | None = None,
) -> Path:
    """Write a generic display-only panel spec fixture."""

    path = tmp_path / "panel_spec.yaml"
    payload = {
        "schema_id": "dnadesign.aligner.msa.visualization.panel_spec",
        "schema_version": 1,
        "display_columns": {
            "coordinate_space": "target_ungapped_position",
            "high_gap_trim_threshold": high_gap_trim_threshold,
            "note": "Display-only trimming policy.",
        },
        "overview": {
            "enabled": True,
            "row_source": row_source,
            "max_display_rows": max_display_rows,
        },
        "consensus_histogram": {
            "enabled": True,
        },
        "sidecar_note": "Display sidecar only; not a conservation denominator.",
    }
    if profiles is not None:
        payload["profiles"] = profiles
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path
