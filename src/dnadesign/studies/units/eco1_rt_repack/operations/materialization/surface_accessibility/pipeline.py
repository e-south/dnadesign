"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/pipeline.py

Materialize Eco1 RT surface-accessibility evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from Bio.PDB.SASA import ShrakeRupley

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    load_yaml,
    require_hash,
    require_mapping,
    require_text,
    resolve_output_root,
    resolve_source_ref,
    sha256,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.structure_io import (
    load_first_model,
    protein_residue_index,
    validate_preprocessing_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.constants import (
    _DEFAULT_CREATED_AT,
    _SHRAKE_RUPLEY_N_POINTS,
    _STRUCTURE_SOURCES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.models import (
    MaterializedSurfaceAccessibilityArtifacts,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.rows import surface_row
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.writer import (
    write_surface_accessibility_profile,
)


def materialize_surface_accessibility_profile(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedSurfaceAccessibilityArtifacts:
    """Materialize complex-context SASA evidence for every canonical Eco1 RT position."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    structure_sources = load_yaml(root / _STRUCTURE_SOURCES)
    selected_source = require_mapping(structure_sources.get("selected_source"), "selected_source")
    model_path = resolve_source_ref(root, require_text(selected_source, "ec86kit_model_ref"))
    require_hash(model_path, require_text(selected_source, "ec86kit_model_sha256"))

    preprocessing_manifest_path = out_root / "structure_preprocessing_manifest.yaml"
    backbone_bundle_path = out_root / "backbone_bundle.yaml"
    residue_map_path = out_root / "residue_map.parquet"
    for path in (preprocessing_manifest_path, backbone_bundle_path, residue_map_path):
        if not path.exists():
            raise FileNotFoundError(path)

    preprocessing_manifest = load_yaml(preprocessing_manifest_path)
    validate_preprocessing_manifest(preprocessing_manifest, selected_source=selected_source)

    model = load_first_model(model_path)
    ShrakeRupley(n_points=_SHRAKE_RUPLEY_N_POINTS).compute(model, level="R")
    residue_rows = pq.read_table(residue_map_path).to_pylist()
    residue_index = protein_residue_index(model, rt_chain_id=require_text(selected_source, "rt_chain_id"))
    rows = [surface_row(residue=residue, residue_index=residue_index) for residue in residue_rows]

    output_path = out_root / "surface_accessibility_profile.parquet"
    write_surface_accessibility_profile(
        output_path,
        rows=rows,
        upstream_hashes={
            "structure_sources_yaml": "sha256:" + sha256(root / _STRUCTURE_SOURCES),
            "structure_preprocessing_manifest": "sha256:" + sha256(preprocessing_manifest_path),
            "backbone_bundle": "sha256:" + sha256(backbone_bundle_path),
            "residue_map": "sha256:" + sha256(residue_map_path),
            "ec86kit_model": "sha256:" + require_text(selected_source, "ec86kit_model_sha256"),
        },
        selected_source=selected_source,
        created_at=created_at,
    )
    return MaterializedSurfaceAccessibilityArtifacts(surface_accessibility_profile_path=output_path)
