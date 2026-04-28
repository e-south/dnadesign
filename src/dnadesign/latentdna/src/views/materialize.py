"""
View materialization helpers for latentdna.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import PromoterMetadataCohortConfig
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..metadata.derivations import derive_metadata_value
from ..sources.resolver import (
    ResolvedSource,
    inspect_source_schema,
    iter_records_batches,
    missing_overlay_merge_columns,
    require_matrix_bundle_paths,
    resolve_source,
)
from ..workspaces.loader import WorkspaceContext
from .row_contracts import source_backed_view_row_contract

_MATERIALIZE_BATCH_SIZE = 2048
_SIG35_PATTERN = re.compile(r"__sig35[=_]([A-Za-z0-9]+)")
_CONTROL_LABELS = {"spyp", "sulap", "soxsp", "j23105", "spy_p", "sul_ap", "sox_sp"}
_REGULONDB_NATIVE_PROMOTER_DERIVATIONS = {
    "regulondb__sigma_factor_set",
    "regulondb__regulator_composition",
    "regulondb__box_pattern",
    "regulondb__confidence_level_set",
    "regulondb__metadata_completeness_class",
}


def _reraise_missing_vector_column(
    exc: Exception,
    *,
    view_id: str,
    source_id: str,
    vector_column: str,
) -> None:
    missing_columns = missing_overlay_merge_columns(exc)
    if vector_column not in missing_columns:
        return
    raise ContractViolationError(
        f"view {view_id} vector column is missing from source {source_id}: {vector_column}"
    ) from exc


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_regulator_name(value: object) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    token = text.split("_", 1)[0].strip()
    if not token:
        return None
    return {
        "baer": "baeR",
        "background": "background",
        "background_only": "background",
        "cpxr": "cpxR",
        "control": "control",
        "lexa": "lexA",
    }.get(token.lower(), token)


def _normalized_regulators(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        values = value
    else:
        values = [value]
    normalized = sorted(
        {_canonical_regulator_name(item) for item in values if _canonical_regulator_name(item) is not None},
        key=str.casefold,
    )
    return [str(item) for item in normalized]


def _is_control_row(row: dict[str, object]) -> bool:
    label = (_normalize_text(row.get("usr_label__primary")) or "").lower()
    template_id = (_construct_template_id(row) or "").lower()
    plan = _normalize_text(row.get("densegen__plan"))
    if template_id in {"wt", "wildtype", "manual"}:
        return True
    if label in _CONTROL_LABELS:
        return True
    return plan is None


def _construct_template_id(row: dict[str, object]) -> str | None:
    return (
        _normalize_text(row.get("construct_template_id"))
        or _normalize_text(row.get("template_id"))
        or _normalize_text(row.get("construct__template_id"))
    )


def _design_family(row: dict[str, object]) -> str:
    plan = _normalize_text(row.get("densegen__plan"))
    if plan is not None:
        if plan.startswith("background_only"):
            return "background_only"
        if plan.startswith("ethanol_ciprofloxacin"):
            return "ethanol_ciprofloxacin"
        if plan.startswith("ethanol"):
            return "ethanol"
        if plan.startswith("ciprofloxacin"):
            return "ciprofloxacin"
    if _is_control_row(row):
        return "control"
    return "control"


def _design_regulator_composition(row: dict[str, object]) -> str:
    if _is_control_row(row):
        return "control"
    family = _design_family(row)
    regulators = _normalized_regulators(row.get("densegen__required_regulators"))
    if family == "background_only" and not regulators:
        return "background"
    if regulators:
        return regulators[0] if len(regulators) == 1 else "+".join(regulators)

    plan = _normalize_text(row.get("densegen__plan")) or ""
    tokens = [token for token in plan.split("__") if token]
    if len(tokens) >= 2 and not tokens[1].startswith("sigma70_"):
        composition_parts = [_canonical_regulator_name(token) for token in tokens[1].replace("_", "+").split("+")]
        composition_parts = sorted(
            {
                part
                for part in composition_parts
                if part not in {None, "control"} and not str(part).startswith("sig35=")
            },
            key=str.casefold,
        )
        if composition_parts:
            return composition_parts[0] if len(composition_parts) == 1 else "+".join(composition_parts)
    if family == "background_only":
        return "background"
    return "unknown"


def _configured_derivation_value(
    context: WorkspaceContext,
    row: dict[str, object],
    *,
    column_name: str,
) -> object:
    derivation = (context.config.metadata.derivations or {}).get(column_name)
    if derivation is None:
        return None
    return derive_metadata_value(row, derivation)


def _sig35_variant(row: dict[str, object], *, context: WorkspaceContext | None = None) -> str:
    if context is not None:
        configured = _configured_derivation_value(context, row, column_name="sig35_variant")
        if configured is not None:
            text = _normalize_text(configured)
            return "unknown" if text is None else text.lower()
    plan = _normalize_text(row.get("densegen__plan")) or ""
    match = _SIG35_PATTERN.search(plan)
    if match is not None:
        return match.group(1).lower()
    annotated = (
        _sig35_variant_from_feature_detail(row)
        or _sig35_variant_from_seq_annot_features(row)
        or _sig35_variant_from_derived_retention(row)
    )
    if annotated is not None:
        return annotated
    if _is_control_row(row):
        return "control"
    raise ContractViolationError(
        "sig35_variant could not be derived for a synthetic promoter row; expected densegen__plan to contain "
        "__sig35=, densegen__used_tfbs_detail to contain an upstream sigma70_core fixed element, "
        "or seq_annot__features to contain a Sigma-35 feature sequence"
    )


def _sig35_variant_from_feature_detail(row: dict[str, object]) -> str | None:
    for entry in _used_tfbs_detail_entries(row.get("densegen__used_tfbs_detail")):
        if str(entry.get("part_kind") or "").strip().lower() != "fixed_element":
            continue
        if str(entry.get("role") or "").strip().lower() != "upstream":
            continue
        if str(entry.get("constraint_name") or "").strip().lower() != "sigma70_core":
            continue
        variant = _normalize_text(entry.get("variant_id")) or _normalize_text(entry.get("core_sequence"))
        if variant is None:
            variant = _normalize_text(entry.get("sequence"))
        if variant is not None:
            return variant.lower() if len(variant) == 1 else variant.upper()
    return None


def _used_tfbs_detail_entries(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed payloads are caught by callers
            raise ContractViolationError("densegen__used_tfbs_detail must be valid JSON when encoded as text") from exc
    if not isinstance(value, list) and hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            value = converted
    if not isinstance(value, list):
        raise ContractViolationError("densegen__used_tfbs_detail must decode to a list of dict entries")
    entries: list[dict[str, object]] = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            raise ContractViolationError("densegen__used_tfbs_detail entries must be dictionaries")
        entries.append(dict(item))
    return entries


def _sig35_variant_from_seq_annot_features(row: dict[str, object]) -> str | None:
    sequence = _normalize_text(row.get("sequence"))
    bounds_are_current_sequence = _annotation_bounds_match_current_sequence(row, sequence=sequence)
    matches: set[str] = set()
    for feature in _seq_annot_feature_entries(row.get("seq_annot__features")):
        role_hint = str(feature.get("role_hint") or "").strip().lower()
        label = str(feature.get("label") or "").strip().lower()
        if role_hint != "sigma70_minus35" and label != "-35":
            continue
        feature_sequence = _feature_sequence_from_qualifiers(feature)
        if feature_sequence is None and sequence is not None and bounds_are_current_sequence:
            feature_sequence = _feature_sequence_from_bounds(feature, sequence=sequence)
        if feature_sequence is not None:
            matches.add(feature_sequence.upper())
    if len(matches) > 1:
        raise ContractViolationError(f"seq_annot__features contains multiple Sigma-35 feature sequences: {matches}")
    return next(iter(matches), None)


def _sig35_variant_from_derived_retention(row: dict[str, object]) -> str | None:
    sequence = _normalize_text(row.get("sequence"))
    if sequence is None:
        return None
    target_length = row.get("derived__target_length")
    if target_length is not None:
        try:
            if int(target_length) != len(sequence):
                return None
        except (TypeError, ValueError):
            return None
    matches: set[str] = set()
    for feature in _generic_feature_entries(
        row.get("derived__features_retained"),
        field_name="derived__features_retained",
    ):
        role_hint = str(feature.get("role_hint") or "").strip().lower()
        label = str(feature.get("label") or "").strip().lower()
        if role_hint != "sigma70_minus35" and label != "-35":
            continue
        for interval in _generic_feature_entries(
            feature.get("derived_intervals_0"),
            field_name="derived_intervals_0",
        ):
            feature_sequence = _feature_sequence_from_bounds(interval, sequence=sequence)
            if feature_sequence is not None:
                matches.add(feature_sequence.upper())
    if len(matches) > 1:
        raise ContractViolationError(
            f"derived__features_retained contains multiple Sigma-35 feature sequences: {matches}"
        )
    return next(iter(matches), None)


def _annotation_bounds_match_current_sequence(row: dict[str, object], *, sequence: str | None) -> bool:
    if sequence is None:
        return False
    start = row.get("seq_annot__sequence_region_start_0")
    end = row.get("seq_annot__sequence_region_end_0")
    if start is None or end is None:
        return False
    try:
        return int(start) == 0 and int(end) == len(sequence)
    except (TypeError, ValueError):
        return False


def _seq_annot_feature_entries(value: object) -> list[dict[str, object]]:
    return _generic_feature_entries(value, field_name="seq_annot__features")


def _generic_feature_entries(value: object, *, field_name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed payloads are caught by callers
            raise ContractViolationError(f"{field_name} must be valid JSON when encoded as text") from exc
    if not isinstance(value, list) and hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            value = converted
    if not isinstance(value, list):
        raise ContractViolationError(f"{field_name} must decode to a list of dict entries")
    entries: list[dict[str, object]] = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            raise ContractViolationError(f"{field_name} entries must be dictionaries")
        entries.append(dict(item))
    return entries


def _feature_sequence_from_qualifiers(feature: dict[str, object]) -> str | None:
    qualifiers = feature.get("qualifiers")
    if qualifiers is None:
        return None
    if hasattr(qualifiers, "as_py"):
        qualifiers = qualifiers.as_py()
    if not isinstance(qualifiers, list) and hasattr(qualifiers, "tolist"):
        converted = qualifiers.tolist()
        if isinstance(converted, list):
            qualifiers = converted
    if not isinstance(qualifiers, list):
        raise ContractViolationError("seq_annot__features qualifiers must decode to a list")
    for qualifier in qualifiers:
        if hasattr(qualifier, "as_py"):
            qualifier = qualifier.as_py()
        if not isinstance(qualifier, dict):
            raise ContractViolationError("seq_annot__features qualifier entries must be dictionaries")
        key = str(qualifier.get("key") or "").strip().lower()
        value = _normalize_text(qualifier.get("value"))
        if value is None:
            continue
        if key == "feature_sequence":
            return value
        if key != "note":
            continue
        for token in value.replace(";", " ").split():
            if token.startswith("feature_sequence="):
                return token.split("=", 1)[1]
    return None


def _feature_sequence_from_bounds(feature: dict[str, object], *, sequence: str) -> str | None:
    start = feature.get("start_0")
    end = feature.get("end_0")
    if start is None or end is None:
        return None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if start_i < 0 or end_i <= start_i or end_i > len(sequence):
        return None
    return sequence[start_i:end_i]


def _spacer_length(row: dict[str, object]) -> int | None:
    if _is_control_row(row):
        return None
    detail_entries = _used_tfbs_detail_entries(row.get("densegen__used_tfbs_detail"))
    if not detail_entries:
        return None
    spacer_values: set[int] = set()
    for entry in detail_entries:
        if str(entry.get("part_kind") or "").strip().lower() != "fixed_element":
            continue
        spacer_raw = entry.get("spacer_length")
        if spacer_raw is None:
            continue
        try:
            spacer_values.add(int(spacer_raw))
        except (TypeError, ValueError) as exc:
            raise ContractViolationError("spacer_length metadata must be integer-valued") from exc
    if not spacer_values:
        raise ContractViolationError(
            "spacer_length could not be derived for a synthetic promoter row; expected densegen__used_tfbs_detail"
        )
    if len(spacer_values) != 1:
        raise ContractViolationError(
            f"spacer_length derivation expected one realized spacer length, found {sorted(spacer_values)}"
        )
    return next(iter(spacer_values))


def _campaign_prior(row: dict[str, object]) -> str:
    family = _design_family(row)
    return {
        "background_only": "background",
        "ethanol": "ethanol",
        "ciprofloxacin": "cipro",
        "ethanol_ciprofloxacin": "and",
        "control": "control",
    }.get(family, "control")


def _source_class(row: dict[str, object]) -> str:
    if _normalize_text(row.get("densegen__plan")) is not None:
        return "densegen"
    source_family = _normalize_text(row.get("source_family"))
    if source_family is not None:
        normalized = source_family.lower()
        if "densegen" in normalized:
            return "densegen"
        if "reference" in normalized or "genbank" in normalized or "standard" in normalized:
            return "reference_control"
        return normalized
    if _normalize_text(row.get("promoter_standard__collection_id")) is not None:
        return "synthetic_reference_standard"
    return "manual_or_wildtype" if _is_control_row(row) else "densegen"


def _promoter_metadata_value(row: dict[str, object], *, derive: str, context: WorkspaceContext | None = None) -> object:
    if derive in _REGULONDB_NATIVE_PROMOTER_DERIVATIONS:
        if derive not in row:
            raise ContractViolationError(f"native RegulonDB promoter metadata column is missing: {derive}")
        return row[derive]
    if derive == "design_family":
        return _design_family(row)
    if derive == "design_regulator_composition":
        return _design_regulator_composition(row)
    if derive == "sig35_variant":
        return _sig35_variant(row, context=context)
    if derive == "spacer_length":
        return _spacer_length(row)
    if derive == "campaign_prior":
        return _campaign_prior(row)
    if derive == "is_control":
        return _is_control_row(row)
    if derive == "source_class":
        return _source_class(row)
    raise ContractViolationError(f"unsupported promoter metadata derivation: {derive}")


def _promoter_metadata_columns(
    rows: list[dict[str, object]],
    *,
    context: WorkspaceContext,
    configs: list[tuple[str, PromoterMetadataCohortConfig]],
) -> dict[str, pa.Array]:
    arrays: dict[str, pa.Array] = {}
    for cohort_id, config in configs:
        values = [_promoter_metadata_value(row, derive=config.derive, context=context) for row in rows]
        arrays[cohort_id] = pa.array(values)
    return arrays


def _derived_metadata_value(
    context: WorkspaceContext,
    row: dict[str, object],
    *,
    column_name: str,
) -> object:
    if column_name == "construct_template_id":
        return _construct_template_id(row)
    derivation = (context.config.metadata.derivations or {}).get(column_name)
    if derivation is not None:
        return derive_metadata_value(row, derivation)
    raise ContractViolationError(f"metadata column {column_name!r} is requested but no derivation is configured")


def _matrix_from_vector_column(column: pa.Array | pa.ChunkedArray, *, dtype: str, label: str) -> np.ndarray:
    if isinstance(column, pa.Array):
        column = pa.chunked_array([column])
    if column.null_count:
        raise ContractViolationError(f"view {label} vector column contains null rows")

    array = column.combine_chunks()
    value_type = getattr(array.type, "value_type", None)
    if value_type is None or not (
        pa.types.is_integer(value_type) or pa.types.is_floating(value_type) or pa.types.is_unsigned_integer(value_type)
    ):
        raise ContractViolationError(f"view {label} vector column must contain numeric list values")

    if pa.types.is_fixed_size_list(array.type):
        dims = array.type.list_size
        values = array.values.to_numpy(zero_copy_only=False)
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        lengths = np.diff(offsets)
        if lengths.size == 0:
            return np.empty((0, 0), dtype=dtype)
        dims = int(lengths[0])
        if np.any(lengths != dims):
            raise ContractViolationError(f"view {label} vector column must be rectangular, not ragged")
        values = array.values.to_numpy(zero_copy_only=False)[int(offsets[0]) : int(offsets[-1])]
    else:
        raise ContractViolationError(f"view {label} vector column must materialize from an Arrow list array")

    matrix = np.asarray(values, dtype=dtype)
    expected = len(array) * dims
    if matrix.size != expected:
        raise ContractViolationError(
            f"view {label} vector column produced {matrix.size} values for {len(array)} rows x {dims} dims"
        )
    return np.ascontiguousarray(matrix.reshape(len(array), dims))


def _assert_unique_keys(
    seen: set[tuple[object, ...]],
    batch: pa.RecordBatch,
    *,
    key_names: list[str],
    label: str,
) -> None:
    key_columns = [batch.column(name).to_pylist() for name in key_names]
    for key in zip(*key_columns, strict=True):
        if key in seen:
            raise ContractViolationError(f"{label} must be unique for keys {key_names}: duplicate {key!r}")
        seen.add(key)


def _validate_rows_unique(rows: pa.Table, *, source, label: str) -> None:
    key_tables = [
        ([source.record_key], f"source {label} record_key"),
    ]
    if source.context_key:
        key_tables.append(([source.subject_key, source.context_key], f"source {label} (subject_key, context_key)"))
    else:
        key_tables.append(([source.subject_key], f"source {label} subject_key"))

    for key_names, error_label in key_tables:
        seen: set[tuple[object, ...]] = set()
        for row in rows.select(key_names).to_pylist():
            key = tuple(row[name] for name in key_names)
            if key in seen:
                raise ContractViolationError(f"{error_label} must be unique for keys {key_names}: duplicate {key!r}")
            seen.add(key)


def _materialize_matrix_bundle_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    resolved: ResolvedSource,
    source,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    rows_path, matrix_path, _ = require_matrix_bundle_paths(resolved)
    inspect_source_schema(resolved)
    rows = read_table(rows_path)
    required_columns = [source.record_key, source.subject_key]
    if source.context_key is not None:
        required_columns.append(source.context_key)
    missing = [column for column in required_columns if column not in rows.column_names]
    if missing:
        raise ContractViolationError(f"view {view_id} matrix bundle rows are missing required columns: {missing}")
    _validate_rows_unique(rows, source=source, label=view_id)

    matrix = np.asarray(read_matrix(matrix_path), dtype=context.analysis_dtype)
    if matrix.ndim != 2:
        raise ContractViolationError(f"view {view_id} matrix bundle must be 2D")
    if matrix.shape[0] != rows.num_rows:
        raise ContractViolationError(
            f"view {view_id} matrix bundle row count mismatch: matrix has {matrix.shape[0]} rows but "
            f"rows.parquet has {rows.num_rows}"
        )

    write_matrix(artifact_dir / "matrix.npy", np.ascontiguousarray(matrix))
    write_table(rows, artifact_dir / "rows.parquet")
    row_columns = list(rows.column_names)
    return artifact_dir, rows.num_rows, int(matrix.shape[1]), source.record_key, row_columns, row_columns


def _materialize_tabular_vector_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    resolved: ResolvedSource,
    source,
    vector_column: str,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    view = context.require_source_view(view_id)
    try:
        source_schema = inspect_source_schema(resolved)
    except Exception as exc:
        _reraise_missing_vector_column(exc, view_id=view_id, source_id=view.source, vector_column=vector_column)
        raise
    available_columns = set(source_schema["columns"])
    if vector_column not in available_columns:
        raise ContractViolationError(
            f"view {view_id} vector column is missing from source {view.source}: {vector_column}"
        )

    row_contract = source_backed_view_row_contract(
        context,
        source_id=view.source,
        source=source,
        available_columns=available_columns,
    )
    promoter_cohorts = [
        (cohort_id, cohort)
        for cohort_id, cohort in context.config.cohorts.items()
        if isinstance(cohort, PromoterMetadataCohortConfig) and cohort_id in set(row_contract.derived_row_columns)
    ]
    promoter_cohort_ids = set(row_contract.promoter_cohort_ids)
    columns = [*row_contract.processing_row_columns, vector_column]
    processing_row_columns = row_contract.processing_row_columns
    output_row_columns = row_contract.output_row_columns
    derived_row_columns = row_contract.derived_row_columns
    artifact_dir.mkdir(parents=True, exist_ok=True)

    row_count = int(source_schema["row_count"])
    rows_path = artifact_dir / "rows.parquet"
    matrix_path = artifact_dir / "matrix.npy"
    seen_record_keys: set[tuple[object, ...]] = set()
    seen_subject_keys: set[tuple[object, ...]] = set()
    row_writer: pq.ParquetWriter | None = None
    matrix: np.memmap | None = None
    dims: int | None = None
    write_offset = 0

    try:
        for batch in iter_records_batches(resolved, columns=columns, batch_size=_MATERIALIZE_BATCH_SIZE):
            _assert_unique_keys(
                seen_record_keys,
                batch,
                key_names=[source.record_key],
                label=f"source {view.source} record_key",
            )
            if source.context_key:
                _assert_unique_keys(
                    seen_subject_keys,
                    batch,
                    key_names=[source.subject_key, source.context_key],
                    label=f"source {view.source} (subject_key, context_key)",
                )
            else:
                _assert_unique_keys(
                    seen_subject_keys,
                    batch,
                    key_names=[source.subject_key],
                    label=f"source {view.source} subject_key",
                )

            matrix_chunk = _matrix_from_vector_column(
                batch.column(vector_column),
                dtype=context.analysis_dtype,
                label=view_id,
            )
            if dims is None:
                dims = int(matrix_chunk.shape[1])
                matrix = np.lib.format.open_memmap(
                    matrix_path,
                    mode="w+",
                    dtype=np.dtype(context.analysis_dtype),
                    shape=(row_count, dims),
                )
            elif matrix_chunk.shape[1] != dims:
                raise ContractViolationError(
                    "view "
                    f"{view_id} vector column changed dimensionality within one source: "
                    f"{dims} vs {matrix_chunk.shape[1]}"
                )

            next_offset = write_offset + matrix_chunk.shape[0]
            if matrix is None or next_offset > row_count:
                raise ContractViolationError(
                    f"view {view_id} materialized more rows than expected: {next_offset} > {row_count}"
                )
            matrix[write_offset:next_offset] = matrix_chunk

            processing_rows_batch = batch.select(processing_row_columns)
            row_dicts = processing_rows_batch.to_pylist()
            rows_batch = batch.select(output_row_columns)
            for derived_column in [
                column
                for column in derived_row_columns
                if column not in promoter_cohort_ids and column not in rows_batch.column_names
            ]:
                rows_batch = rows_batch.append_column(
                    derived_column,
                    pa.array([_derived_metadata_value(context, row, column_name=derived_column) for row in row_dicts]),
                )
            if promoter_cohorts:
                for cohort_id, array in _promoter_metadata_columns(
                    row_dicts,
                    context=context,
                    configs=promoter_cohorts,
                ).items():
                    if cohort_id in rows_batch.column_names:
                        continue
                    rows_batch = rows_batch.append_column(cohort_id, array)
            if row_writer is None:
                row_writer = pq.ParquetWriter(rows_path, rows_batch.schema)
            row_writer.write_batch(rows_batch)
            write_offset = next_offset
    except Exception as exc:
        _reraise_missing_vector_column(exc, view_id=view_id, source_id=view.source, vector_column=vector_column)
        raise
    finally:
        if row_writer is not None:
            row_writer.close()
        if matrix is not None:
            matrix.flush()
            del matrix

    if dims is None:
        raise ContractViolationError(f"view {view_id} source produced no rows")
    if write_offset != row_count:
        raise ContractViolationError(
            f"view {view_id} materialized {write_offset} rows but source schema reported {row_count}"
        )
    return (
        artifact_dir,
        row_count,
        dims,
        source.record_key,
        row_contract.materialized_row_columns,
        processing_row_columns,
    )


def materialize_view_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    artifact_dir: Path | None = None,
) -> tuple[Path, int, int, str, list[str], list[str]]:
    view = context.require_source_view(view_id)
    source = context.require_source(view.source)
    resolved = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
    target_dir = artifact_dir or (context.output_root / "views" / view_id)
    if view.vector.kind == "bundle_matrix":
        return _materialize_matrix_bundle_artifact(
            context,
            view_id=view_id,
            resolved=resolved,
            source=source,
            artifact_dir=target_dir,
        )
    if resolved.records_path is None:
        raise ContractViolationError(f"view {view_id} source does not expose a records table")
    return _materialize_tabular_vector_artifact(
        context,
        view_id=view_id,
        resolved=resolved,
        source=source,
        vector_column=view.vector.name,
        artifact_dir=target_dir,
    )
