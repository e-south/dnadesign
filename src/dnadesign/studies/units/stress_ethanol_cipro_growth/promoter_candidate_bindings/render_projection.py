"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/render_projection.py

BaseRender-ready metadata projection for bound promoter candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from .contracts import (
    DENSEGEN_RENDER_ANNOTATION_KEYS,
    GENBANK_RENDER_ANNOTATION_KEYS,
    PromoterCandidateBindingsError,
)
from .values import has_value, nonempty_collection, optional_value, required_text, scalar_missing


def annotation_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "id" not in frame.columns:
        raise PromoterCandidateBindingsError("GenBank annotation table missing required columns: ['id']")
    out = frame.copy()
    out["id"] = out["id"].astype(str)
    duplicates = sorted(out.loc[out["id"].duplicated(keep=False), "id"].unique().tolist())
    if duplicates:
        raise PromoterCandidateBindingsError(f"GenBank annotation IDs must be unique; duplicates={duplicates}")
    return out.set_index("id", drop=False)


def project_baserender_values(
    *,
    candidate: pd.Series,
    candidate_id: str,
    source_class: str,
    annotation_by_id: pd.DataFrame,
    canonical_sequence: str,
) -> tuple[str, str, dict[str, Any]]:
    if source_class == "densegen":
        return _densegen_values(candidate, candidate_id=candidate_id, canonical_sequence=canonical_sequence)
    if source_class == "construct_derived":
        return _genbank_values(
            candidate,
            candidate_id=candidate_id,
            annotation_by_id=annotation_by_id,
            canonical_sequence=canonical_sequence,
        )
    raise PromoterCandidateBindingsError(
        f"Candidate {candidate_id!r} has unsupported BaseRender source class {source_class!r}."
    )


def _densegen_values(
    candidate: pd.Series,
    *,
    candidate_id: str,
    canonical_sequence: str,
) -> tuple[str, str, dict[str, Any]]:
    identity_fields = ("densegen__plan", "densegen__run_id", "densegen__sampling_library_hash")
    missing_identity = [field for field in identity_fields if not has_value(candidate.get(field))]
    raw_annotations = candidate.get("densegen__used_tfbs_detail")
    if missing_identity or not nonempty_collection(raw_annotations):
        raise PromoterCandidateBindingsError(
            f"Candidate {candidate_id!r} has incomplete DenseGen render metadata; "
            f"missing={missing_identity or ['densegen__used_tfbs_detail']}."
        )
    regulators = candidate.get("densegen__required_regulators")
    if regulators is None or scalar_missing(regulators):
        raise PromoterCandidateBindingsError(
            f"Candidate {candidate_id!r} has incomplete DenseGen render metadata; "
            "missing=['densegen__required_regulators']."
        )
    annotations = _normalize_densegen_annotations(raw_annotations, sequence=canonical_sequence)
    return (
        "densegen_tfbs",
        "densegen__used_tfbs_detail",
        {
            "densegen__used_tfbs_detail": annotations,
            "densegen__required_regulators": _normalize_string_collection(regulators),
            "seq_annot__features": None,
            "seq_annot__source_file": None,
            "usr_label__primary": optional_value(candidate.get("usr_label__primary")),
            "derived__product_kind": None,
        },
    )


def _genbank_values(
    candidate: pd.Series,
    *,
    candidate_id: str,
    annotation_by_id: pd.DataFrame,
    canonical_sequence: str,
) -> tuple[str, str, dict[str, Any]]:
    if annotation_by_id.empty or candidate_id not in annotation_by_id.index:
        raise PromoterCandidateBindingsError(f"Candidate {candidate_id!r} is missing required GenBank render metadata.")
    annotation = annotation_by_id.loc[candidate_id]
    required = ("seq_annot__features", "seq_annot__source_artifact_uri")
    missing = [field for field in required if field not in annotation.index]
    if missing or not nonempty_collection(annotation.get("seq_annot__features")):
        raise PromoterCandidateBindingsError(
            f"Candidate {candidate_id!r} has incomplete GenBank render metadata; "
            f"missing={missing or ['seq_annot__features']}."
        )
    label = required_text(candidate.get("usr_label__primary"), field="GenBank display label", row_id=candidate_id)
    return (
        "usr_genbank_annotations_v1",
        "seq_annot__features",
        {
            "densegen__used_tfbs_detail": None,
            "densegen__required_regulators": None,
            "seq_annot__features": _normalize_genbank_annotations(
                annotation["seq_annot__features"], sequence=canonical_sequence
            ),
            "seq_annot__source_file": _relative_artifact_uri(
                annotation["seq_annot__source_artifact_uri"], row_id=candidate_id
            ),
            "usr_label__primary": label,
            "derived__product_kind": "source_record",
        },
    )


def _normalize_densegen_annotations(value: Any, *, sequence: str) -> list[dict[str, Any]]:
    items = value.tolist() if hasattr(value, "tolist") else list(value)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise PromoterCandidateBindingsError(f"DenseGen render annotation at index {index} must be a mapping.")
        row = _allowed_fields(item, DENSEGEN_RENDER_ANNOTATION_KEYS)
        part_kind = required_text(row.get("part_kind"), field=f"DenseGen annotation {index}.part_kind")
        literal = required_text(row.get("sequence"), field=f"DenseGen annotation {index}.sequence")
        if part_kind == "tfbs":
            _validate_tfbs_span(row, literal=literal, sequence=sequence, index=index)
        elif part_kind == "fixed_element":
            required_text(row.get("role"), field=f"DenseGen annotation {index}.role")
            required_text(row.get("constraint_name"), field=f"DenseGen annotation {index}.constraint_name")
        else:
            raise PromoterCandidateBindingsError(f"DenseGen annotation {index} has unsupported part_kind.")
        normalized.append(row)
    return normalized


def _validate_tfbs_span(row: dict[str, Any], *, literal: str, sequence: str, index: int) -> None:
    try:
        start, length, end = (int(row[field]) for field in ("offset", "length", "end"))
    except (KeyError, TypeError, ValueError) as exc:
        raise PromoterCandidateBindingsError(f"DenseGen annotation {index} has an invalid TFBS span.") from exc
    if start < 0 or length < 1 or end != start + length or end > len(sequence) or len(literal) != length:
        raise PromoterCandidateBindingsError(f"DenseGen annotation {index} has an invalid TFBS span.")
    required_text(row.get("orientation"), field=f"DenseGen annotation {index}.orientation")
    required_text(row.get("regulator"), field=f"DenseGen annotation {index}.regulator")


def _normalize_genbank_annotations(value: Any, *, sequence: str) -> list[dict[str, Any]]:
    items = value.tolist() if hasattr(value, "tolist") else list(value)
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise PromoterCandidateBindingsError(f"GenBank render annotation at index {index} must be a mapping.")
        row = _allowed_fields(item, GENBANK_RENDER_ANNOTATION_KEYS)
        for field in ("feature_id", "feature_type", "label"):
            required_text(row.get(field), field=f"GenBank annotation {index}.{field}")
        try:
            start, end, strand = (int(row[field]) for field in ("start_0", "end_0", "strand"))
        except (KeyError, TypeError, ValueError) as exc:
            raise PromoterCandidateBindingsError(f"GenBank annotation {index} has an invalid span.") from exc
        if start < 0 or end <= start or end > len(sequence) or strand not in {-1, 0, 1}:
            raise PromoterCandidateBindingsError(f"GenBank annotation {index} has an invalid span.")
        normalized.append(row)
    return normalized


def _allowed_fields(item: Mapping[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    return {key: _normalize_nested(item[key]) for key in allowed if key in item and not scalar_missing(item[key])}


def _relative_artifact_uri(value: Any, *, row_id: str) -> str:
    text = required_text(value, field="GenBank source artifact URI", row_id=row_id)
    normalized = text.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or normalized.startswith("~") or ":" in path.parts[0]:
        raise PromoterCandidateBindingsError(
            f"GenBank source artifact URI for {row_id!r} must be a relative confined path."
        )
    return str(path)


def _normalize_string_collection(value: Any) -> list[str]:
    items = value.tolist() if hasattr(value, "tolist") else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_nested(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value
