"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/scorecard.py

Validated inputs for representation scorecard computations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Names = Annotated[list[Text], Field(min_length=1)]


class ScorecardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassificationTask(ScorecardModel):
    source_scalar: Text | None = None
    source_view: Text | None = None
    label_column: Text
    positive_values: Names
    score_columns: Names
    metrics: dict[Literal["auroc", "auprc"], Text] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_and_metrics(self):
        if (self.source_scalar is None) == (self.source_view is None):
            raise ValueError("each classification task requires exactly one source_scalar or source_view")
        if len(set(self.metrics.values())) != len(self.metrics):
            raise ValueError("classification metric ids must be distinct")
        return self


class ScorecardCandidate(ScorecardModel):
    candidate_id: Text
    candidate_model: Text | None = None
    candidate_family: Text | None = None
    candidate_scope: Text | None = None
    candidate_label: Text | None = None
    classification_tasks: list[ClassificationTask] = Field(default_factory=list)
    neighbors_id: Text | None = None
    context_source: Text | None = None
    agreement_id: Text | None = None
    reducer_id: Text | None = None

    @model_validator(mode="after")
    def require_computation(self):
        if not any(
            (self.classification_tasks, self.neighbors_id, self.context_source, self.agreement_id, self.reducer_id)
        ):
            raise ValueError("each candidate must request at least one scorecard computation")
        return self


class ReferenceNeighborTask(ScorecardModel):
    label_column: Text
    positive_values: Names
    reference_column: Text
    reference_labels: Names


class NeighborEnrichment(ScorecardModel):
    label_column: Text
    metric_name: Text = Field(validation_alias=AliasChoices("metric_name", "metric_id"))


class ScorecardParameters(ScorecardModel):
    candidates: list[ScorecardCandidate] = Field(min_length=1)
    reference_neighbor_tasks: list[ReferenceNeighborTask] = Field(default_factory=list)
    neighbor_label_enrichments: list[NeighborEnrichment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_candidates(self):
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be distinct")
        if not self.reference_neighbor_tasks and not self.neighbor_label_enrichments:
            if any(candidate.neighbors_id for candidate in self.candidates):
                raise ValueError("neighbors_id requires reference_neighbor_tasks or neighbor_label_enrichments")
        return self
