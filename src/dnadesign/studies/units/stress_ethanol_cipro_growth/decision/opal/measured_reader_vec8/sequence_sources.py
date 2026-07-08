"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/sequence_sources.py

Loads sequence and X sources for measured Reader vec8 staging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import X_COLUMN
from .contracts import MeasuredReaderVec8Error

CONTROL_LABELS_BY_READER_DESIGN = {
    "pDual-10": "J23105",
    "pDual-10-spyp": "spyp",
    "pDual-10-sulAp": "sulAp",
}

EXCLUDED_READER_DESIGNS = frozenset(
    {
        "pDual-10-ES5p",
        "pDual-10-ES8p",
        "pDual-10-ES9p",
        "pDual-10-ES14p",
    }
)

DATASETS_ROOT = Path("src/dnadesign/usr/datasets")
SFXI_DATASET = DATASETS_ROOT / "usr_sfxi_pdual10_densegen_promoters"
PROMOTER_REFERENCES = DATASETS_ROOT / "usr_promoter_references"
OPAL_CANDIDATES = DATASETS_ROOT / "usr_prom_eth_cip_opal_candidates"
LATENTDNA_VIEW = (
    Path("src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/views")
    / "intermediate_embedding_7b_context_anchor_mean_bidir_concat"
    / "rows.parquet"
)


def load_reader_sequence_resolution_sources(repo_root: Path) -> dict[str, pd.DataFrame]:
    root = Path(repo_root).expanduser().resolve()
    return {
        "sfxi_records": _read_parquet(root / SFXI_DATASET / "records.parquet"),
        "sfxi_views": _read_parquet(root / SFXI_DATASET / "_views" / "sequence_views.parquet"),
        "promoter_references": _read_parquet(root / PROMOTER_REFERENCES / "records.parquet"),
        "candidate_records": _read_parquet(root / OPAL_CANDIDATES / "records.parquet"),
        "latentdna_rows": _read_parquet(root / LATENTDNA_VIEW),
    }


def build_reader_sequence_resolution_table(repo_root: Path) -> pd.DataFrame:
    sources = load_reader_sequence_resolution_sources(repo_root)
    rows: list[dict[str, Any]] = []
    rows.extend(_sfxi_source_rows(sources["sfxi_records"], sources["sfxi_views"], sources["latentdna_rows"]))
    rows.extend(_control_rows(sources["promoter_references"], sources["latentdna_rows"]))
    return pd.DataFrame(rows)


def candidate_x_status(candidate_records: pd.DataFrame, candidate_id: str) -> str:
    hits = candidate_records.loc[candidate_records["id"].astype(str).eq(str(candidate_id))]
    if hits.empty:
        return "missing_candidate_id"
    if X_COLUMN not in hits.columns:
        return "missing_x_column"
    if not all(_present_vector(value) for value in hits[X_COLUMN].tolist()):
        return "missing_x_value"
    return "available"


def _sfxi_source_rows(records: pd.DataFrame, views: pd.DataFrame, latentdna_rows: pd.DataFrame) -> list[dict[str, Any]]:
    records_by_id = records.set_index("id")
    latent_ids = set(latentdna_rows.get("construct__anchor_id", pd.Series(dtype=str)).dropna().astype(str))
    latent_sfxi_refs = set(
        latentdna_rows.get("sfxi_ref__reference_instance_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    rows: list[dict[str, Any]] = []
    for _, view in views.iterrows():
        sequence_id = str(view["sequence_id"])
        if sequence_id not in records_by_id.index:
            continue
        sequence = str(records_by_id.loc[sequence_id, "sequence"])
        for alias in _aliases(view.get("aliases")):
            if not alias.startswith("pDual-10"):
                continue
            rows.append(
                {
                    "reader_design_id": alias,
                    "sequence": sequence,
                    "sequence_source": "usr_sfxi_pdual10_densegen_promoters",
                    "sequence_source_id": sequence_id,
                    "x_scope": "selected_latentdna_view"
                    if sequence_id in latent_ids or sequence_id in latent_sfxi_refs
                    else "missing",
                }
            )
    return rows


def _control_rows(promoter_references: pd.DataFrame, latentdna_rows: pd.DataFrame) -> list[dict[str, Any]]:
    latent_ids = set(latentdna_rows.get("construct__anchor_id", pd.Series(dtype=str)).dropna().astype(str))
    latent_labels = set(latentdna_rows.get("usr_label__primary", pd.Series(dtype=str)).dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for reader_design_id, label in CONTROL_LABELS_BY_READER_DESIGN.items():
        hits = promoter_references.loc[promoter_references["usr_label__primary"].astype(str).eq(label)]
        if len(hits) != 1:
            raise MeasuredReaderVec8Error(
                f"Expected one promoter reference for {reader_design_id}/{label}; found {len(hits)}."
            )
        record = hits.iloc[0]
        sequence_id = str(record["id"])
        rows.append(
            {
                "reader_design_id": reader_design_id,
                "sequence": str(record["sequence"]),
                "sequence_source": "usr_promoter_references",
                "sequence_source_id": sequence_id,
                "x_scope": "selected_latentdna_view"
                if sequence_id in latent_ids or label in latent_labels
                else "missing",
            }
        )
    return rows


def _aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in list(value) if str(item).strip()]
    except TypeError:
        return []


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MeasuredReaderVec8Error(f"Required measured-reader-vec8 source table not found: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise MeasuredReaderVec8Error(f"Could not read measured-reader-vec8 source table: {path}") from exc


def _present_vector(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    try:
        return len(value) > 0
    except TypeError:
        return True
