"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/hairpin_structure_fixtures.py

Retron-hairpin structure materialization fixtures for RT-SPOP source tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path


def write_hairpin_structure_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a compact retron-hairpin materialized output fixture."""

    repo_root = tmp_path
    hairpin_output_dir = Path("hairpin_fixture")
    materialized_root = repo_root / hairpin_output_dir
    sequence_index_path = materialized_root / "manifest/indexes/sequence_index.tsv"
    reference_index_path = materialized_root / "manifest/indexes/reference_index.tsv"
    sequence_index_path.parent.mkdir(parents=True, exist_ok=True)
    variants = ("26", "43", "195", "196", "197", "198", "199", "200")

    with sequence_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "construct_id",
                "secondary_structure_native_png",
                "composition_overview_png",
                "artifact_bundle",
                "sequence_sha256",
                "sequence_length",
                "folding_status",
                "features_csv",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for variant in variants:
            construct_id = f"pES-retron-{variant}"
            bundle = f"variants/{construct_id}__fixture"
            writer.writerow(
                {
                    "construct_id": construct_id,
                    "secondary_structure_native_png": f"{bundle}/plots/secondary_structure.native.png",
                    "composition_overview_png": f"{bundle}/plots/composition_overview.png",
                    "artifact_bundle": bundle,
                    "sequence_sha256": f"sha256:fixture-{variant}",
                    "sequence_length": "82",
                    "folding_status": "ok",
                    "features_csv": f"{bundle}/sequences/features.csv",
                }
            )
            _write_structure_bundle(materialized_root / bundle, variant=variant)

    with reference_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("construct_id", "left_base", "right_base"), delimiter="\t")
        writer.writeheader()
        for variant in variants:
            writer.writerow({"construct_id": f"pES-retron-{variant}", "left_base": "CGGG", "right_base": "ACAG"})

    _write_msd_region_record_fixture(repo_root, variants=variants)
    return repo_root, hairpin_output_dir


def _write_structure_bundle(bundle_root: Path, *, variant: str) -> None:
    (bundle_root / "plots").mkdir(parents=True, exist_ok=True)
    (bundle_root / "manifest/visual/secondary_structure").mkdir(parents=True, exist_ok=True)
    (bundle_root / "sequences").mkdir(parents=True, exist_ok=True)
    _write_png(bundle_root / "plots/secondary_structure.native.png")
    _write_png(bundle_root / "plots/composition_overview.png")
    native_svg_path = bundle_root / "manifest/visual/secondary_structure/native.svg"
    native_svg_path.write_text(_fixture_structure_svg(), encoding="utf-8")
    annotation_manifest_path = bundle_root / "manifest/visual/secondary_structure/annotation_manifest.json"
    annotation_manifest_path.write_text(json.dumps(_fixture_annotation_manifest()), encoding="utf-8")
    with (bundle_root / "sequences/features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "sequence"))
        writer.writeheader()
        writer.writerows(
            [
                {"role": "stem_base_left", "sequence": "CGGG"},
                {"role": "payload_primary", "sequence": "TCCCTATCAGTGATAGAGA"[: _payload_bp(variant)]},
                {"role": "snapback_foldback_geometry", "sequence": "AGGC"},
                {"role": "payload_complement", "sequence": "TCTCTATCACTGATAGGGA"[: _payload_bp(variant)]},
                {"role": "stem_base_right", "sequence": "ACAG"},
            ]
        )


def _write_msd_region_record_fixture(repo_root: Path, *, variants: tuple[str, ...]) -> None:
    record_root = (
        repo_root / "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/"
        "reader_spop_msd_structure_panel_v1"
    )
    variants_dir = record_root / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for variant in variants:
        filename = f"pes-retron-{int(variant):03d}-msd-region.yaml"
        if variant in {"195", "196", "197", "198", "199", "200"}:
            filename = f"pes-retron-{variant}-msd-region.yaml"
        record_path = variants_dir / filename
        record_path.write_text(
            json.dumps(
                {
                    "display_id": f"pES-retron-{variant}",
                    "features": [
                        {"role": "stem_base_left", "sequence_5to3": "CGGG"},
                        {"role": "snapback_foldback_return", "sequence_5to3": "AGGC"},
                        {"role": "snapback_retained_stem", "sequence_5to3": "GCCT"},
                        {"role": "stem_base_right", "sequence_5to3": "ACAG"},
                    ],
                    "pairing_segments": _pairing_segments_for_variant(variant),
                }
            ),
            encoding="utf-8",
        )
        manifest_rows.append({"display_id": f"pES-retron-{variant}", "record": f"variants/{filename}"})
    (record_root / "manifest.yaml").write_text(json.dumps({"records": manifest_rows}), encoding="utf-8")


def _pairing_segments_for_variant(variant: str) -> list[dict[str, object]]:
    payload_bp = _payload_bp(variant)
    foldback_bp = _foldback_bp(variant)
    return [
        {
            "segment": "payload_stem",
            "length_bp": payload_bp,
            "watson_crick_bp": payload_bp,
            "wobble_bp": 0,
            "mismatch_bp": 0,
            "pairing_status": "canonical_wc",
        },
        {
            "segment": "foldback_stem",
            "length_bp": foldback_bp,
            "watson_crick_bp": foldback_bp,
            "wobble_bp": 0,
            "mismatch_bp": 0,
            "pairing_status": "canonical_wc",
        },
    ]


def _payload_bp(variant: str) -> int:
    return 13 if variant in {"196", "198", "200"} else 15 if variant in {"195", "197", "199"} else 19


def _foldback_bp(variant: str) -> int:
    return {"43": 7, "197": 7, "198": 7, "200": 3}.get(variant, 0)


def _write_png(path: Path) -> None:
    png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lO0b4wAAAABJRU5ErkJggg=="
    path.write_bytes(base64.b64decode(png_data))


def _fixture_structure_svg() -> str:
    points = " ".join(f"{index % 5},{index * 3}" for index in range(50))
    basepair_lines = "\n".join(
        f'<line class="basepairs" id="{index},{51 - index}" x1="{index % 5}" y1="{index * 3}" '
        f'x2="{(51 - index) % 5}" y2="{(51 - index) * 3}" stroke="#444444" stroke-width="1" />'
        for index in range(1, 15)
    )
    texts = "\n".join(f'<text x="{index % 5}" y="{index * 3}">{"ACGT"[index % 4]}</text>' for index in range(50))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="20" height="160">
<polyline class="backbone" points="{points}" stroke="#777777" stroke-width="1" fill="none" />
{basepair_lines}
{texts}
</svg>
"""


def _fixture_annotation_manifest() -> dict[str, object]:
    palette = {
        "flank_5p": "#8d8d8d",
        "flank_3p": "#8d8d8d",
        "stem_base_left": "#0072b2",
        "stem_base_right": "#56b4e9",
        "payload_primary": "#d55e00",
        "payload_complement": "#e69f00",
        "snapback_cap": "#cc79a7",
        "snapback_retained_stem": "#009e73",
        "snapback_foldback_return": "#44aa99",
    }
    return {
        "palette": palette,
        "nucleotides": [{"index": index} for index in range(50)],
        "section_annotations": [
            {"label": "5p flank", "section_semantic": "flank_5p", "start": 0, "end": 4},
            {"label": "Left Base", "section_kind": "stem_base", "start": 4, "end": 8},
            {"label": "payload", "section_semantic": "payload_primary", "start": 8, "end": 23},
            {"label": "cap", "section_semantic": "snapback_cap", "start": 23, "end": 27},
            {"label": "return", "section_semantic": "snapback_foldback_return", "start": 27, "end": 34},
            {"label": "payload complement", "section_semantic": "payload_complement", "start": 34, "end": 42},
            {"label": "Right Base", "section_kind": "stem_base", "start": 42, "end": 46},
            {"label": "3p flank", "section_semantic": "flank_3p", "start": 46, "end": 50},
        ],
    }


__all__ = ["write_hairpin_structure_fixture"]
