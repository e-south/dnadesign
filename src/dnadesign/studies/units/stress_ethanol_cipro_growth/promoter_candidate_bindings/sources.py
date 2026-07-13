"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/sources.py

Assemble the study promoter candidate-binding projection from its registry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifact import preview_promoter_candidate_bindings
from .contracts import BindingSourceArtifact, PromoterCandidateBindingsError, PromoterCandidateBindingsPreview
from .source_adapters import load_alias_source
from .source_io import candidate_selection_sha256, read_parquet, source_artifact
from .source_registry import load_source_registry

_CANDIDATE_COLUMNS = (
    "id",
    "sequence",
    "usr_label__primary",
    "opal_candidate__source_class",
    "opal_candidate__design_family",
    "densegen__plan",
    "densegen__run_id",
    "densegen__sampling_library_hash",
    "densegen__used_tfbs_detail",
    "densegen__required_regulators",
)


def preview_promoter_candidate_bindings_from_repo(repo_root: Path) -> PromoterCandidateBindingsPreview:
    """Build the binding projection declared by the checked-in study registry."""

    root = Path(repo_root).expanduser().resolve()
    registry = load_source_registry(root)
    alias_rows: list[dict[str, str]] = []
    artifacts: list[BindingSourceArtifact] = [
        source_artifact(root, "promoter-candidate-binding-source-registry", root / registry.path)
    ]
    annotation_frames: list[pd.DataFrame] = []
    for source in registry.alias_sources:
        result = load_alias_source(root, source)
        alias_rows.extend(result.alias_rows)
        artifacts.extend(result.source_artifacts)
        if not result.genbank_annotations.empty:
            annotation_frames.append(result.genbank_annotations)
    aliases = pd.DataFrame(alias_rows).sort_values(["alias_namespace", "alias"], kind="stable").reset_index(drop=True)
    if aliases.empty:
        raise PromoterCandidateBindingsError("Promoter candidate-binding registry produced no aliases.")

    candidate_ids = sorted(set(aliases["candidate_id"].astype(str)))
    candidate_path = root / registry.candidate_table.records_path
    candidates = read_parquet(
        candidate_path,
        columns=list(_CANDIDATE_COLUMNS),
        filters=[("id", "in", candidate_ids)],
    )
    missing_candidates = sorted(set(candidate_ids) - set(candidates["id"].astype(str)))
    if missing_candidates:
        raise PromoterCandidateBindingsError(
            f"Promoter aliases reference candidates absent from {registry.candidate_table.dataset_id}: "
            f"{missing_candidates}"
        )
    annotations = pd.concat(annotation_frames, ignore_index=True) if annotation_frames else pd.DataFrame()
    return preview_promoter_candidate_bindings(
        alias_rows=aliases,
        candidate_records=candidates,
        genbank_annotations=annotations,
        candidate_table_id=registry.candidate_table.dataset_id,
        candidate_selection_sha256=candidate_selection_sha256(candidates),
        source_artifacts=tuple(artifacts),
    )


__all__ = ["preview_promoter_candidate_bindings_from_repo"]
