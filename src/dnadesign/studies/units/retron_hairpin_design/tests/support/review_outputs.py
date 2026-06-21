"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/support/review_outputs.py

Review-output fixtures for Retron hairpin study tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


def write_fake_materialized_bundle(root: Path, *, repo_root: Path, row_count: int = 9) -> Path:
    design_set = yaml.safe_load(
        (
            repo_root
            / "docs"
            / "studies"
            / "retron_hairpin_design"
            / "workbench"
            / "design_sets"
            / "teto_pwm_trim_rescue_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    rows = []
    for idx, design in enumerate(design_set["designs"][:row_count], start=1):
        variant_dir = root / "variants" / f"{design['construct_id']}__{design['expected_msd_design_id']}"
        sequences_dir = variant_dir / "sequences"
        plots_dir = variant_dir / "plots"
        manifest_dir = variant_dir / "manifest"
        sequences_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        forward_sequence = ("ATGCGTACCTAGGCTAAGTC" + "G" * idx).upper()
        reverse_complement_sequence = _reverse_complement(forward_sequence)
        for filename in _FAKE_ARTIFACT_FILENAMES:
            target_dir = sequences_dir if filename.endswith((".gb", ".fa", ".csv")) else plots_dir
            if filename.endswith(".png"):
                write_png(target_dir / filename, color=(40 + idx * 12, 90, 160))
            elif filename == "forward.fa":
                (target_dir / filename).write_text(f">forward_{idx}\n{forward_sequence}\n", encoding="utf-8")
            elif filename == "reverse_complement.fa":
                (target_dir / filename).write_text(
                    f">reverse_complement_{idx}\n{reverse_complement_sequence}\n",
                    encoding="utf-8",
                )
            else:
                (target_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
        write_png(plots_dir / "composition_overview.png", color=(40 + idx * 12, 130, 190))
        rows.append(_sequence_index_row(root=root, variant_dir=variant_dir, design=design, idx=idx))
    _write_sequence_index(root / "manifest" / "indexes" / "sequence_index.tsv", rows)
    return root


def fake_video_writer(
    *,
    frames: object,
    still_paths: tuple[Path, ...],
    output_path: Path,
    fps: int,
    seconds_per_frame: int,
) -> None:
    _ = (frames, fps, seconds_per_frame)
    assert len(still_paths) == 9
    assert all(path.is_file() for path in still_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake-mp4")


def write_png(path: Path, *, color: tuple[int, int, int]) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 180), color=color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 304, 164), outline="white", width=3)
    image.save(path)


_FAKE_ARTIFACT_FILENAMES = (
    "forward.gb",
    "reverse_complement.gb",
    "forward.fa",
    "reverse_complement.fa",
    "features.csv",
    "visual_contract.json",
    "construct_manifest.json",
    "folding_prediction.json",
    "composition_overview.svg",
    "secondary_structure.native.png",
)


def _sequence_index_row(*, root: Path, variant_dir: Path, design: dict[str, object], idx: int) -> dict[str, str]:
    rel_variant = variant_dir.relative_to(root).as_posix()
    return {
        "construct_id": str(design["construct_id"]),
        "construct_label": str(design["label"]),
        "msd_design_id": str(design["expected_msd_design_id"]),
        "payload_trim_id": str(design["payload_trim_id"]),
        "payload_trim_class": str(design["payload_trim_id"]).replace("TetR_", ""),
        "parent_payload_id": "TetR_full",
        "pwm_source_ref": "cruncher:westmann_tetr_mitomi:tetR",
        "variant_role": str(design["variant_role"]),
        "scaffold_context": str(design["scaffold_context"]),
        "cap_selector_id": "",
        "stem_base_selector_id": "",
        "rt_mode": str(design["rt_mode"]),
        "decision_group": "teto_pwm_trim_rescue_v1",
        "control_id": "",
        "composition_id": f"composition-{idx}",
        "unit_count": "1",
        "sequence_length": "60",
        "sequence_sha256": f"sha-{idx}",
        "composition_config": f"{rel_variant}/manifest/composition.yaml",
        "artifact_bundle": rel_variant,
        "construct_bundle": f"{rel_variant}/runtime/construct",
        "genbank": f"{rel_variant}/sequences/forward.gb",
        "reverse_complement_genbank": f"{rel_variant}/sequences/reverse_complement.gb",
        "forward_fasta": f"{rel_variant}/sequences/forward.fa",
        "reverse_complement_fasta": f"{rel_variant}/sequences/reverse_complement.fa",
        "features_csv": f"{rel_variant}/sequences/features.csv",
        "visual_contract": f"{rel_variant}/plots/visual_contract.json",
        "construct_manifest": f"{rel_variant}/plots/construct_manifest.json",
        "folding_prediction": f"{rel_variant}/plots/folding_prediction.json",
        "folding_status": "ok",
        "composition_overview_svg": f"{rel_variant}/plots/composition_overview.svg",
        "composition_overview_png": f"{rel_variant}/plots/composition_overview.png",
        "secondary_structure_native_png": f"{rel_variant}/plots/secondary_structure.native.png",
        "finder_reveal": "",
    }


def _write_sequence_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()


__all__ = ["fake_video_writer", "write_fake_materialized_bundle", "write_png"]
