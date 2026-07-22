"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/baserender_validation.py

Compatibility validation through BaseRender's public sequence-panel API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dnadesign.baserender import SchemaError, adapt_record, sequence_panel_config_for_adapter

from .contracts import PromoterCandidateBindingsError


def validate_baserender_rows(rows: pd.DataFrame) -> None:
    """Prove that each canonical candidate can be adapted without rendering a figure."""

    candidate_rows = rows.drop_duplicates(subset=["candidate_id"], keep="first")
    for _, row in candidate_rows.iterrows():
        candidate_id = str(row["candidate_id"])
        adapter_kind = str(row["baserender_adapter_kind"])
        config = sequence_panel_config_for_adapter(adapter_kind)
        try:
            adapt_record(
                _adapter_row(row, adapter_kind=adapter_kind),
                adapter_kind=adapter_kind,
                adapter_columns=config.adapter_columns,
                adapter_policies=config.adapter_policies,
            )
        except (SchemaError, TypeError, ValueError) as exc:
            raise PromoterCandidateBindingsError(
                f"Candidate {candidate_id!r} is incompatible with BaseRender adapter {adapter_kind!r}: {exc}"
            ) from exc


def _adapter_row(row: pd.Series, *, adapter_kind: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": str(row["candidate_id"]),
        "sequence": str(row["canonical_sequence"]),
    }
    if adapter_kind == "densegen_tfbs":
        record["densegen__used_tfbs_detail"] = row["densegen__used_tfbs_detail"]
        return record
    record.update(
        {
            "seq_annot__features": row["seq_annot__features"],
            "seq_annot__source_file": row["seq_annot__source_file"],
            "usr_label__primary": row["usr_label__primary"],
            "derived__product_kind": row["derived__product_kind"],
        }
    )
    return record


__all__ = ["validate_baserender_rows"]
