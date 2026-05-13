"""Static semantic validation for LatentDNA workspace sequence contracts."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pyarrow.compute as pc
import pyarrow.types as pa_types

from ..contracts.errors import WorkspaceValidationError
from ..contracts.workspace import (
    InferFeatureScalarSidecarSourceConfig,
    InferFeatureSidecarSourceConfig,
    SourceBackedViewConfig,
)
from ..sources.resolver import iter_records_batches, resolve_source
from ..workspaces.loader import WorkspaceContext

_FIXED_60_LABEL_RE = re.compile(r"(?<!\d)60\s*bp\b|(?<!\d)60bp\b", re.IGNORECASE)
_INFER_SOURCE_TYPES = (InferFeatureSidecarSourceConfig, InferFeatureScalarSidecarSourceConfig)
_POOLING_COLUMNS = ("pooling_start_0", "pooling_end_0")


@dataclass(slots=True)
class SourceSemanticProfile:
    source_id: str
    sequence_scope: str | None
    emitted_length_bp: int | None
    source_interval_length_bp: int | str | None
    pooling_span_bp: int | str | None
    focal_rule: str | None
    window_selection_rule: str | None
    length_counts: dict[int, int] = field(default_factory=dict)
    pooling_span_counts: dict[int, int] = field(default_factory=dict)

    @property
    def observed_lengths(self) -> set[int]:
        return set(self.length_counts)

    @property
    def observed_pooling_spans(self) -> set[int]:
        return set(self.pooling_span_counts)

    @property
    def fixed_60(self) -> bool:
        if self.emitted_length_bp == 60:
            return True
        return self.observed_lengths == {60}

    def to_detail(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "sequence_scope": self.sequence_scope,
            "emitted_length_bp": self.emitted_length_bp,
            "source_interval_length_bp": self.source_interval_length_bp,
            "pooling_span_bp": self.pooling_span_bp,
            "focal_rule": self.focal_rule,
            "window_selection_rule": self.window_selection_rule,
            "length_counts": dict(sorted(self.length_counts.items())),
            "pooling_span_counts": dict(sorted(self.pooling_span_counts.items())),
        }


def _claims_fixed_60bp(value: object) -> bool:
    return bool(_FIXED_60_LABEL_RE.search(str(value or "")))


def _counter_to_preview(counter: Counter[int]) -> str:
    if not counter:
        return "unavailable"
    return ", ".join(f"{length}:{count}" for length, count in sorted(counter.items())[:12])


def _source_declares_sequence_semantics(source: object) -> bool:
    return any(
        getattr(source, field_name, None) is not None
        for field_name in (
            "sequence_scope",
            "emitted_length_bp",
            "source_interval_length_bp",
            "pooling_span_bp",
            "focal_rule",
            "window_selection_rule",
        )
    )


def _schema_columns(source_columns: dict[str, set[str]], source_id: str) -> set[str]:
    return set(source_columns.get(source_id, set()))


def _length_counter(
    context: WorkspaceContext,
    source_id: str,
    source: object,
    *,
    columns: set[str],
) -> Counter[int]:
    wanted_columns: list[str]
    if "length" in columns:
        wanted_columns = ["length"]
    elif "sequence" in columns:
        wanted_columns = ["sequence"]
    else:
        return Counter()

    resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    counts: Counter[int] = Counter()
    for batch in iter_records_batches(resolved, columns=wanted_columns, batch_size=65536):
        if "length" in batch.schema.names:
            for value in batch.column("length").to_pylist():
                if value is not None:
                    counts[int(value)] += 1
            continue
        for value in batch.column("sequence").to_pylist():
            if value is not None:
                counts[len(str(value))] += 1
    return counts


def _pooling_span_counter(
    context: WorkspaceContext,
    source_id: str,
    source: object,
    *,
    columns: set[str],
) -> Counter[int]:
    if not isinstance(source, _INFER_SOURCE_TYPES):
        return Counter()
    if not set(_POOLING_COLUMNS).issubset(columns):
        return Counter()

    resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    counts: Counter[int] = Counter()
    for batch in iter_records_batches(resolved, columns=list(_POOLING_COLUMNS), batch_size=65536):
        if not set(_POOLING_COLUMNS).issubset(batch.schema.names):
            continue
        valid_mask = pc.and_(pc.is_valid(batch["pooling_start_0"]), pc.is_valid(batch["pooling_end_0"]))
        if int(pc.sum(pc.cast(valid_mask, "int64")).as_py() or 0) != batch.num_rows:
            raise WorkspaceValidationError(
                f"infer source {source_id} has null pooling bounds despite declaring span semantics"
            )
        if not (
            pa_types.is_integer(batch["pooling_start_0"].type) and pa_types.is_integer(batch["pooling_end_0"].type)
        ):
            raise WorkspaceValidationError(f"infer source {source_id} pooling bounds must be integer columns")
        spans = pc.subtract(batch["pooling_end_0"], batch["pooling_start_0"]).to_pylist()
        for span in spans:
            if span is not None:
                counts[int(span)] += 1
    return counts


def _build_source_profiles(
    context: WorkspaceContext,
    *,
    source_columns: dict[str, set[str]],
    source_schemas: dict[str, dict[str, object]],
) -> dict[str, SourceSemanticProfile]:
    profiles: dict[str, SourceSemanticProfile] = {}
    for source_id, source in sorted(context.config.sources.items()):
        where = dict(getattr(source, "where", None) or {})
        pooling_operation = str(where.get("pooling_operation") or "").strip()
        needs_infer_pooling_check = pooling_operation in {"anchor_mean", "core60_mean"}
        needs_length_profile = (
            _source_declares_sequence_semantics(source)
            or _claims_fixed_60bp(source_id)
            or _claims_fixed_60bp(getattr(source, "sequence_scope", None))
        )
        columns = _schema_columns(source_columns, source_id)
        length_counts = (
            _length_counter(context, source_id, source, columns=columns) if needs_length_profile else Counter()
        )
        span_counts = (
            _pooling_span_counter(context, source_id, source, columns=columns)
            if needs_infer_pooling_check
            else Counter()
        )
        profile = SourceSemanticProfile(
            source_id=source_id,
            sequence_scope=getattr(source, "sequence_scope", None),
            emitted_length_bp=getattr(source, "emitted_length_bp", None),
            source_interval_length_bp=getattr(source, "source_interval_length_bp", None),
            pooling_span_bp=getattr(source, "pooling_span_bp", None),
            focal_rule=getattr(source, "focal_rule", None),
            window_selection_rule=getattr(source, "window_selection_rule", None),
            length_counts=dict(length_counts),
            pooling_span_counts=dict(span_counts),
        )
        source_row_count = int(source_schemas.get(source_id, {}).get("row_count") or 0)
        expected_length = profile.emitted_length_bp
        if expected_length is not None:
            if not profile.observed_lengths and source_row_count > 0:
                raise WorkspaceValidationError(
                    f"source {source_id} declares emitted_length_bp={expected_length} "
                    "but exposes no length/sequence column"
                )
            if profile.observed_lengths != {int(expected_length)}:
                raise WorkspaceValidationError(
                    f"source {source_id} declares emitted_length_bp={expected_length} but observed lengths are "
                    f"{_counter_to_preview(length_counts)}"
                )
        expected_span = profile.pooling_span_bp
        if isinstance(expected_span, int) and needs_infer_pooling_check:
            if not profile.observed_pooling_spans and source_row_count > 0:
                raise WorkspaceValidationError(
                    f"infer source {source_id} declares pooling_span_bp={expected_span} but exposes no pooling bounds"
                )
            if profile.observed_pooling_spans != {expected_span}:
                raise WorkspaceValidationError(
                    f"infer source {source_id} declares pooling_span_bp={expected_span} but observed spans are "
                    f"{_counter_to_preview(span_counts)}"
                )
        if (
            pooling_operation == "core60_mean"
            and profile.observed_pooling_spans
            and profile.observed_pooling_spans != {60}
        ):
            raise WorkspaceValidationError(
                f"infer source {source_id} uses core60_mean but observed pooling spans are "
                f"{_counter_to_preview(span_counts)}"
            )
        if pooling_operation == "anchor_mean" and source_row_count > 0 and not profile.observed_pooling_spans:
            raise WorkspaceValidationError(f"infer source {source_id} uses anchor_mean but exposes no pooling bounds")
        profiles[source_id] = profile
    return profiles


def _view_profile(
    context: WorkspaceContext,
    view_id: str,
    *,
    profiles: dict[str, SourceSemanticProfile],
) -> SourceSemanticProfile | None:
    view = context.config.views.get(view_id)
    if not isinstance(view, SourceBackedViewConfig):
        return None
    return profiles.get(view.source)


def _validate_view_scope_labels(
    context: WorkspaceContext,
    *,
    profiles: dict[str, SourceSemanticProfile],
) -> None:
    for view_id, view in sorted(context.config.views.items()):
        if not isinstance(view, SourceBackedViewConfig):
            continue
        profile = profiles.get(view.source)
        if profile is None or profile.fixed_60 or not profile.length_counts:
            continue
        scope = dict(view.tags or {}).get("scope")
        if _claims_fixed_60bp(scope):
            raise WorkspaceValidationError(
                f"view {view_id} uses fixed-60bp scope label {scope!r}, but source {view.source} has observed "
                f"lengths {_counter_to_preview(Counter(profile.length_counts))}"
            )


def _validate_candidate_set_labels(
    context: WorkspaceContext,
    *,
    profiles: dict[str, SourceSemanticProfile],
) -> None:
    for candidate_set_id, candidate_set in sorted(context.config.candidate_sets.items()):
        for view_id, title in sorted(candidate_set.panel_titles.items()):
            profile = _view_profile(context, view_id, profiles=profiles)
            if profile is None or profile.fixed_60 or not profile.length_counts or not _claims_fixed_60bp(title):
                continue
            raise WorkspaceValidationError(
                f"candidate_set {candidate_set_id} labels {view_id} as {title!r}, but the backing source has mixed "
                f"lengths {_counter_to_preview(Counter(profile.length_counts))}"
            )


def _validate_notebook_panel_labels(
    context: WorkspaceContext,
    *,
    profiles: dict[str, SourceSemanticProfile],
) -> None:
    for notebook_id, notebook in sorted(context.config.notebooks.items()):
        for view_id, title in zip(notebook.candidate_grid_views, notebook.candidate_grid_panel_titles, strict=False):
            profile = _view_profile(context, view_id, profiles=profiles)
            if profile is None or profile.fixed_60 or not profile.length_counts or not _claims_fixed_60bp(title):
                continue
            raise WorkspaceValidationError(
                f"notebook {notebook_id} labels {view_id} as {title!r}, but the backing source has mixed lengths "
                f"{_counter_to_preview(Counter(profile.length_counts))}"
            )


def validate_workspace_sequence_semantics(
    context: WorkspaceContext,
    *,
    source_columns: dict[str, set[str]],
    source_schemas: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    """Validate static sequence-scope claims against observable source metadata."""

    profiles = _build_source_profiles(context, source_columns=source_columns, source_schemas=source_schemas)
    _validate_view_scope_labels(context, profiles=profiles)
    _validate_candidate_set_labels(context, profiles=profiles)
    _validate_notebook_panel_labels(context, profiles=profiles)
    warnings: list[str] = []
    for source_id, profile in profiles.items():
        source = context.config.sources[source_id]
        where: dict[str, Any] = dict(getattr(source, "where", None) or {})
        if where.get("pooling_operation") == "seq_mean" and profile.pooling_span_bp == 60:
            warnings.append(
                f"source {source_id} uses seq_mean while declaring pooling_span_bp=60; prefer core60_mean/anchor_mean "
                "for fixed-window pooling"
            )
    details = [
        profile.to_detail()
        for profile in profiles.values()
        if _source_declares_sequence_semantics(context.config.sources[profile.source_id])
    ]
    return details, warnings


__all__ = ["validate_workspace_sequence_semantics"]
