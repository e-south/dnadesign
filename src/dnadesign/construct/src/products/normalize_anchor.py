"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/products/normalize_anchor.py

Normalize-anchor Construct product builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone

from dnadesign.usr import compute_id, normalize_sequence

from ..contracts.config import JobConfig
from ..contracts.errors import ValidationError
from ..persistence.records import BuiltRecord
from ..realization.normalize_anchor import NormalizeTemplateLoader, realize_normalize_anchor
from ..realization.sequences import alphabet_for_sequence
from ..sources.input_rows import input_usr_labels
from .sequence_views import build_normalize_sequence_view


def build_normalize_anchor_record(
    *,
    row: dict[str, object],
    cfg: JobConfig,
    spec_id: str,
    output_dataset_id: str,
    load_template: NormalizeTemplateLoader,
) -> BuiltRecord:
    if cfg.job.normalize_anchor is None:
        raise ValidationError("job.normalize_anchor is required when job.mode='normalize_anchor'.")
    normalize_cfg = cfg.job.normalize_anchor
    realization = realize_normalize_anchor(
        row=row,
        cfg=cfg,
        load_template=load_template,
    )
    sequence = realization.source_sequence
    analysis_sequence = realization.analysis_sequence
    source_start_0 = realization.source_start_0
    source_end_0 = realization.source_end_0
    template = realization.template
    template_sha256 = realization.template_sha256
    added_left_bp = realization.added_left_bp
    added_right_bp = realization.added_right_bp
    focal_selection = realization.focal_selection
    retention = realization.retention

    alphabet = alphabet_for_sequence(analysis_sequence)
    output_id = compute_id("dna", normalize_sequence(analysis_sequence, "dna", alphabet))
    label_primary, label_aliases = input_usr_labels(row)
    if label_primary is not None and not label_primary.endswith("_core60"):
        derived_primary = f"{label_primary}_core60"
    else:
        derived_primary = label_primary
    if label_primary is not None and label_primary not in label_aliases:
        label_aliases = [label_primary, *label_aliases]
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "id": output_id,
        "construct__job": cfg.job.id,
        "construct__spec_id": spec_id,
        "construct__context_id": f"{cfg.job.id}:analysis_window",
        "construct__context_kind": "analysis_window",
        "construct__template_id": template.id if template is not None else None,
        "construct__template_kind": template.kind if template is not None else None,
        "construct__template_source": template.source if template is not None else None,
        "construct__template_dataset": template.dataset if template is not None else None,
        "construct__template_field": template.field if template is not None else None,
        "construct__template_record_id": template.record_id if template is not None else None,
        "construct__template_sha256": template_sha256,
        "construct__template_length": len(template.sequence) if template is not None else None,
        "construct__template_circular": bool(template.circular) if template is not None else None,
        "construct__input_dataset": cfg.job.input.source.dataset,
        "construct__input_fields": [cfg.job.input.field],
        "construct__input_id": str(row["id"]),
        "construct__input_length": len(sequence),
        "construct__assembly_mode": "analysis_window",
        "construct__slot_count": 0,
        "construct__slots": [],
        "construct__anchor_id": str(row["id"]),
        "construct__anchor_orientation": "forward",
        "construct__anchor_start": 0,
        "construct__anchor_end": len(analysis_sequence),
        "construct__orientation": "forward",
        "construct__forward_anchor_start": 0,
        "construct__forward_anchor_end": len(analysis_sequence),
        "construct__parent_forward_construct_id": "",
        "construct__mode": "normalize_anchor",
        "construct__focal_part": "analysis_window",
        "construct__focal_part_length": len(analysis_sequence),
        "construct__window_semantics": "normalize_anchor",
        "construct__window_reference": focal_selection.focal_rule,
        "construct__window_direction": (
            "upstream" if normalize_cfg.over_length_policy.window_anchor == "upstream_of_focal" else "symmetric"
        ),
        "construct__window_size_bp": len(analysis_sequence),
        "construct__window_upstream_bp": (
            len(analysis_sequence) if normalize_cfg.over_length_policy.window_anchor == "upstream_of_focal" else None
        ),
        "construct__window_downstream_bp": (
            0 if normalize_cfg.over_length_policy.window_anchor == "upstream_of_focal" else None
        ),
        "construct__window_offset_bp": None,
        "construct__window_start": source_start_0,
        "construct__window_end": source_end_0,
        "construct__resolved_length": len(analysis_sequence),
        "construct__full_construct_length": len(analysis_sequence),
        "construct__parts": [],
    }
    derived_metadata = {
        "id": output_id,
        "derived__parent_id": str(row["id"]),
        "derived__parent_dataset": cfg.job.input.source.dataset,
        "derived__operation": "construct.normalize_anchor",
        "derived__product_kind": normalize_cfg.product_kind,
        "derived__target_length": normalize_cfg.target_length,
        "derived__source_interval_start_0": source_start_0,
        "derived__source_interval_end_0": source_end_0,
        "derived__source_intervals_0": [
            {"start_0": source_start_0, "end_0": source_end_0, "strand": 1, "partial": False}
        ],
        "derived__orientation": "forward",
        "derived__template_id": template.id if template is not None else None,
        "derived__template_dataset": template.dataset if template is not None else None,
        "derived__focal_rule": focal_selection.focal_rule,
        "derived__focal_features": list(focal_selection.focal_features),
        "derived__focal_confidence": focal_selection.focal_confidence,
        "derived__analysis_only": True,
        "derived__added_left_bp": added_left_bp or None,
        "derived__added_right_bp": added_right_bp or None,
        "derived__added_sequence_source": (
            f"{template.source}:{normalize_cfg.under_length_policy.placement_ref}"
            if template is not None and normalize_cfg.under_length_policy is not None
            else None
        ),
        "derived__features_retained": retention.retained if normalize_cfg.emit_feature_retention_report else None,
        "derived__features_clipped": retention.clipped if normalize_cfg.emit_feature_retention_report else None,
        "derived__features_lost": retention.lost if normalize_cfg.emit_feature_retention_report else None,
        "derived__created_by": "construct",
        "derived__spec_id": spec_id,
    }
    record = BuiltRecord(
        output_id=output_id,
        sequence=analysis_sequence,
        alphabet=alphabet,
        metadata=metadata,
        label_primary=derived_primary,
        label_aliases=label_aliases,
        created_at=created_at,
        derived_metadata=derived_metadata,
    )
    if normalize_cfg.output_sequence_view.create:
        record.sequence_view = build_normalize_sequence_view(
            record=record,
            output_dataset_id=output_dataset_id,
            parent_row=row,
            source_start_0=source_start_0,
            source_end_0=source_end_0,
            anchor_start_0=0,
            anchor_end_0=len(analysis_sequence),
            recommended_pooling=normalize_cfg.output_sequence_view.recommended_pooling,
        )
    return record
