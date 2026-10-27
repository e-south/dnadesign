#!/usr/bin/env python3
"""Verify browser-manifest and ChimeraX-script molecular scene contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

DNA_COLOR = "#B97700"
RNA_COLOR = "#C84C5A"
SURFACE_ALPHA = 0.65
CHIMERAX_SURFACE_TRANSPARENCY = 35
RIBBON_WIDTH = 1.35
RIBBON_THICKNESS = 0.28
EXPECTED_VISUAL_CONTRACT = {
    "protein_surface_scope": "protein_only",
    "protein_surface_alpha": SURFACE_ALPHA,
    "dna_color": DNA_COLOR,
    "rna_color": RNA_COLOR,
    "py3dmol_nucleic_display": "backbone_ribbon_with_base_spokes",
    "py3dmol_nucleic_ribbon_width_angstrom": RIBBON_WIDTH,
    "py3dmol_nucleic_ribbon_thickness_angstrom": RIBBON_THICKNESS,
    "chimerax_nucleic_display": "ladder",
    "chimerax_surface_transparency_percent": CHIMERAX_SURFACE_TRANSPARENCY,
    "chimerax_nucleotide_color_target": "acf",
}


def _verify_browser_manifest(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ["browser manifest is not a mapping"]
    failures: list[str] = []
    contract = payload.get("visual_contract")
    if contract != EXPECTED_VISUAL_CONTRACT:
        failures.append(f"visual_contract mismatch: {contract!r}")
    surface_default = payload.get("protein_surface_default")
    if not isinstance(surface_default, bool):
        failures.append("protein_surface_default must be declared as a boolean")
        surface_default = False
    structures = payload.get("structures")
    if not isinstance(structures, list) or not structures:
        failures.append("browser manifest has no structures")
        return failures
    for scene in structures:
        if not isinstance(scene, dict):
            failures.append("browser manifest contains a non-mapping scene")
            continue
        scene_id = str(scene.get("candidate_id") or "unnamed_scene")
        styles = scene.get("molecule_styles")
        if not isinstance(styles, list):
            failures.append(f"{scene_id}: molecule_styles is missing")
            continue
        by_class = {str(style.get("molecule_class")): style for style in styles if isinstance(style, dict)}
        protein = by_class.get("protein", {})
        if surface_default and (
            protein.get("style") != "surface" or float(protein.get("opacity", -1.0)) != SURFACE_ALPHA
        ):
            failures.append(f"{scene_id}: protein surface must use alpha {SURFACE_ALPHA}")
        elif protein and protein.get("style") == "surface" and float(protein.get("opacity", -1.0)) != SURFACE_ALPHA:
            failures.append(f"{scene_id}: declared protein surface must use alpha {SURFACE_ALPHA}")
        for molecule_class, color in (("dna", DNA_COLOR), ("rna", RNA_COLOR)):
            style = by_class.get(molecule_class, {})
            if style.get("style") != "backbone_ribbon_with_base_spokes":
                failures.append(f"{scene_id}: {molecule_class} browser style is invalid")
            if str(style.get("color")) != color:
                failures.append(f"{scene_id}: {molecule_class} color must be {color}")
            if float(style.get("width", -1.0)) != RIBBON_WIDTH:
                failures.append(f"{scene_id}: {molecule_class} ribbon width must be {RIBBON_WIDTH} A")
            if float(style.get("thickness", -1.0)) != RIBBON_THICKNESS:
                failures.append(f"{scene_id}: {molecule_class} ribbon thickness must be {RIBBON_THICKNESS} A")
    return failures


def _verify_review_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ["review manifest is not a mapping"], []
    deliverables = payload.get("deliverables")
    if not isinstance(deliverables, list):
        return ["review manifest has no deliverables list"], []
    failures: list[str] = []
    audited: list[dict[str, str]] = []
    for row in deliverables:
        if not isinstance(row, dict):
            failures.append("review manifest contains a non-mapping deliverable")
            continue
        artifact_kind = str(row.get("artifact_kind") or "")
        if artifact_kind not in {"structure_browser_manifest", "chimerax_script"}:
            continue
        status = str(row.get("status") or "")
        if status.startswith("skipped"):
            continue
        deliverable_id = str(row.get("deliverable_id") or "unnamed_deliverable")
        relative_path = str(row.get("path") or "")
        artifact_path = (path.parent / relative_path).resolve()
        audited.append(
            {
                "deliverable_id": deliverable_id,
                "artifact_kind": artifact_kind,
                "path": str(artifact_path),
            }
        )
        if not artifact_path.exists():
            failures.append(f"{deliverable_id}: artifact is missing: {artifact_path}")
            continue
        artifact_failures = (
            _verify_browser_manifest(artifact_path)
            if artifact_kind == "structure_browser_manifest"
            else _verify_chimerax_script(artifact_path)
        )
        failures.extend(f"{deliverable_id}: {failure}" for failure in artifact_failures)
    if not audited:
        failures.append("review manifest exposes no molecular structure artifacts")
    return failures, audited


def _verify_chimerax_script(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    required = (
        "nucleotides #1/D,E,F ladder",
        f"color #1/D {DNA_COLOR} target acf",
        f"color #1/E,F {RNA_COLOR} target acf",
    )
    for command in required:
        if command not in text:
            failures.append(f"missing ChimeraX command: {command}")
    if "nucleotides #1/D,E,F atoms" in text:
        failures.append("default ChimeraX script resets nucleotides to atoms")
    surface_count = len(re.findall(r"(?m)^surface\s+", text))
    transparency_values = [
        int(value) for value in re.findall(r"(?m)^transparency\s+.+?\s+([0-9]+)\s+target\s+s\s*$", text)
    ]
    if surface_count and not transparency_values:
        failures.append("ChimeraX surface has no declared transparency")
    if any(value != CHIMERAX_SURFACE_TRANSPARENCY for value in transparency_values):
        failures.append(f"ChimeraX surface transparency must be {CHIMERAX_SURFACE_TRANSPARENCY}: {transparency_values}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--review-manifest", type=Path)
    input_group.add_argument("--browser-manifest", type=Path)
    parser.add_argument("--chimerax-script", type=Path)
    args = parser.parse_args()
    audited: list[dict[str, str]] = []
    if args.review_manifest is not None:
        if args.chimerax_script is not None:
            parser.error("--chimerax-script cannot be combined with --review-manifest")
        failures, audited = _verify_review_manifest(args.review_manifest)
    else:
        if args.chimerax_script is None:
            parser.error("--chimerax-script is required with --browser-manifest")
        failures = [
            *_verify_browser_manifest(args.browser_manifest),
            *_verify_chimerax_script(args.chimerax_script),
        ]
        audited = [
            {
                "deliverable_id": "browser_manifest",
                "artifact_kind": "structure_browser_manifest",
                "path": str(args.browser_manifest),
            },
            {
                "deliverable_id": "chimerax_script",
                "artifact_kind": "chimerax_script",
                "path": str(args.chimerax_script),
            },
        ]
    payload = {
        "schema_id": "molecular_scene_contract_verification_v1",
        "status": "pass" if not failures else "fail",
        "review_manifest": str(args.review_manifest) if args.review_manifest is not None else None,
        "browser_manifest": str(args.browser_manifest) if args.browser_manifest is not None else None,
        "chimerax_script": str(args.chimerax_script) if args.chimerax_script is not None else None,
        "audited_artifacts": audited,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
