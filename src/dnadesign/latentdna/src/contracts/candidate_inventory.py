"""Candidate X inventory contracts shared by status and notebook surfaces."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateInventoryRow(BaseModel):
    """One machine-readable candidate representation ledger row."""

    model_config = ConfigDict(extra="forbid")

    study_id: str
    candidate_set_ids: list[str] = Field(default_factory=list)
    view_id: str
    source_id: str | None = None
    dataset: str | None = None
    row_basis: str | None = None
    model_name: str | None = None
    feature_family: str | None = None
    modality: str
    sequence_scope: str | None = None
    emitted_length_bp: int | None = None
    source_interval_length_bp: int | str | None = None
    pooling_span_bp: int | str | None = None
    focal_rule: str | None = None
    window_selection_rule: str | None = None
    pooling_operation: str | None = None
    orientation: str | None = None
    coordinate_space_id: str | None = None
    role: str | None = None
    n_rows: int | None = None
    n_dims: int | None = None
    materialization_status: str
    freshness_status: str


__all__ = ["CandidateInventoryRow"]
