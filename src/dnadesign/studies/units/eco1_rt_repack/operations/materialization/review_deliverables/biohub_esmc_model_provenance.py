"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_model_provenance.py

Model-provenance helpers for Eco1 Biohub ESMC review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def sae_request_manifest_summary(path: Path) -> dict[str, Any]:
    """Return non-secret Biohub ESMC SAE model identity from a redacted manifest."""

    loaded = _load_yaml_mapping(path)
    return {
        "model": str(loaded.get("model") or ""),
        "sae_model": str(loaded.get("sae_model") or ""),
        "normalize_features": bool(loaded.get("normalize_features")),
        "biohub_request_hash": str(loaded.get("biohub_request_hash") or ""),
    }


def combined_sae_fold_llr_model_summary(
    *,
    sae_request_manifest_path: Path,
    wt_mutation_scoring_manifest_path: Path,
) -> dict[str, Any]:
    """Return model identities for the combined SAE, fold, and WT LLR panel."""

    sae_manifest = _load_yaml_mapping(sae_request_manifest_path)
    wt_manifest = _load_yaml_mapping(wt_mutation_scoring_manifest_path)
    return {
        "model": str(sae_manifest.get("model") or ""),
        "sae_model": str(sae_manifest.get("sae_model") or ""),
        "normalize_features": bool(sae_manifest.get("normalize_features")),
        "biohub_request_hash": str(sae_manifest.get("biohub_request_hash") or ""),
        "wt_mutation_scoring_model": str(wt_manifest.get("model") or ""),
        "wt_mutation_scoring_request_hash": str(wt_manifest.get("biohub_request_hash") or ""),
        "wt_mutation_scoring_method": str(wt_manifest.get("scoring_method_id") or ""),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
