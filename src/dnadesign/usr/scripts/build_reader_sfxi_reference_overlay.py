"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/scripts/build_reader_sfxi_reference_overlay.py

Build a provenance-aware USR SFXI reference metric overlay from Reader vec8 outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.contracts import SchemaError
from dnadesign.usr.src.overlays import overlay_path
from dnadesign.usr.src.registry import (
    SFXI_REF_COLUMNS,
    SFXI_REF_NAMESPACE,
    arrow_type_from_str,
    load_registry,
    registry_entry,
    validate_overlay_schema,
)

DEFAULT_OUTPUT_DATASET = "usr_sfxi_pdual10_densegen_promoters"
DEFAULT_COLLECTION_ID = "reader_sfxi_pdual10_latest"
DEFAULT_CAMPAIGN_ID = "20260501_sfxi_promoter_setpoint_scatter"
DEFAULT_READER_VEC8 = Path(
    "experiments/2026/20260501_sfxi_promoter_setpoint_scatter/"
    "outputs/artifacts/sfxi_vec8_latest_preview_input.preview_import_csv/vec8.parquet"
)
DEFAULT_SETPOINT_NAME = "and"
DEFAULT_SETPOINT_VECTOR = (0.0, 0.0, 0.0, 1.0)
DEFAULT_METRIC_COLUMN = "sfxi"
DEFAULT_METRIC_PROVENANCE = "reader.vec8.sfxi_setpoint_scatter+dnadesign.opal.api.sfxi"
DEFAULT_SCORE_REF = "dnadesign.opal.api.sfxi.score_vec8"
SFXI_OVERLAY_NAMESPACE = SFXI_REF_NAMESPACE

REQUIRED_BASE_COLUMNS = ("id", "sequence")
REQUIRED_SCORED_COLUMNS = (
    "design_id",
    "sequence",
    "setpoint_name",
    "objective_name",
    "api_version",
    "state_order",
    "setpoint_vector",
    "denom_percentile",
    "denom_used",
    "logic_fidelity",
    "effect_raw",
    "effect_scaled",
)
OPTIONAL_SCORED_COLUMNS = (
    "sequence_source_id",
    "experiment_id",
    "experiment_date",
    "time_selected_h",
    "reference_design_id",
    "r_logic",
    "clip_lo_mask",
    "clip_hi_mask",
    "intensity_disabled",
    "flat_logic",
)


@dataclass(frozen=True, slots=True)
class SFXIReferenceOverlayResult:
    dataset: str
    namespace: str
    source_vec8: str
    collection_id: str
    campaign_id: str
    metric_id: str
    rows: int
    registry_validated: bool
    written: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_usr_root() -> Path:
    return _repo_root() / "src" / "dnadesign" / "usr" / "datasets"


def _default_reader_root() -> Path:
    return _repo_root().parent / "reader"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, list | tuple | dict):
        return False
    if hasattr(value, "tolist") and not isinstance(value, str | bytes):
        converted = value.tolist()
        if isinstance(converted, list):
            return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, list | tuple) or hasattr(result, "tolist"):
        return False
    return bool(result)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{context} missing required column(s): {missing}")


def _normalized_dna_sequence(value: object) -> str:
    text = "" if _is_missing(value) else "".join(str(value).split()).upper()
    if not text:
        raise SchemaError("SFXI overlay sequence values must be non-empty.")
    return text


def _finite_float(value: object, *, context: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{context} must be numeric; got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise SchemaError(f"{context} must be finite; got non-finite value {value!r}.")
    return numeric


def _optional_float(value: object) -> float | None:
    return None if _is_missing(value) else _finite_float(value, context="optional SFXI numeric field")


def _optional_int(value: object) -> int | None:
    if _is_missing(value):
        return None
    numeric = _finite_float(value, context="optional SFXI integer field")
    if not float(numeric).is_integer():
        raise SchemaError(f"optional SFXI integer field must be integral; got {value!r}.")
    return int(numeric)


def _optional_bool(value: object) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and float(value) in {0.0, 1.0}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    raise SchemaError(f"optional SFXI boolean field must be boolean-like; got {value!r}.")


def _optional_string(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_list(value: object, *, context: str) -> list[object]:
    if _is_missing(value):
        raise SchemaError(f"{context} must be present.")
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, str | bytes):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise SchemaError(f"{context} must be a list-like value; got {value!r}.") from exc
        if isinstance(parsed, list | tuple):
            return list(parsed)
    raise SchemaError(f"{context} must be a list-like value; got {value!r}.")


def _state_order(value: object) -> list[str]:
    state_order = [str(item) for item in _parse_list(value, context="SFXI state_order")]
    if len(state_order) != 4 or any(not item for item in state_order):
        raise SchemaError(f"SFXI state_order must contain four non-empty state labels; got {state_order!r}.")
    return state_order


def _setpoint_vector(value: object) -> list[float]:
    vector = [
        _finite_float(item, context="SFXI setpoint_vector item")
        for item in _parse_list(value, context="SFXI setpoint_vector")
    ]
    if len(vector) != 4:
        raise SchemaError(f"SFXI setpoint_vector must contain four numeric values; got {vector!r}.")
    return vector


def _validate_unique_normalized_sequences(frame: pd.DataFrame, *, context: str) -> None:
    duplicates = frame.loc[frame["_sequence_norm"].duplicated(keep=False), "_sequence_norm"].drop_duplicates().tolist()
    if duplicates:
        sample = duplicates[:5]
        raise SchemaError(f"{context} contains duplicate normalized sequence value(s): {sample}")


def _validate_metric_contract(frame: pd.DataFrame) -> None:
    if frame["sfxi_ref__id_pair"].duplicated().any():
        duplicate_pairs = frame.loc[
            frame["sfxi_ref__id_pair"].duplicated(keep=False),
            "sfxi_ref__id_pair",
        ].drop_duplicates()
        raise SchemaError(
            "SFXI overlay contains duplicate (reference_instance_id, metric_id) pair(s): "
            f"{duplicate_pairs.head(5).tolist()}"
        )
    if frame["id"].duplicated().any():
        duplicate_ids = frame.loc[frame["id"].duplicated(keep=False), "id"].drop_duplicates().head(5).tolist()
        raise SchemaError(f"SFXI overlay contains duplicate USR id value(s): {duplicate_ids}")


def metric_id_for_setpoint(*, objective_name: object, setpoint_name: object, metric_column: str) -> str:
    objective = _optional_string(objective_name) or "sfxi_v1"
    setpoint = _optional_string(setpoint_name) or DEFAULT_SETPOINT_NAME
    return f"{objective}/{setpoint}/{metric_column}"


def build_sfxi_reference_overlay_frame(
    *,
    base_records: pd.DataFrame,
    scored_rows: pd.DataFrame,
    collection_id: str,
    campaign_id: str,
    source_ref: str,
    metric_column: str = DEFAULT_METRIC_COLUMN,
    metric_provenance: str = DEFAULT_METRIC_PROVENANCE,
    score_ref: str = DEFAULT_SCORE_REF,
) -> pd.DataFrame:
    """Build one ``sfxi_ref`` overlay row per matched USR base record."""

    if not str(metric_provenance).strip():
        raise SchemaError("SFXI overlay requires non-empty metric provenance.")
    required_scored_columns = tuple(dict.fromkeys([*REQUIRED_SCORED_COLUMNS, metric_column]))
    selected_scored_columns = [
        *required_scored_columns,
        *(column for column in OPTIONAL_SCORED_COLUMNS if column in scored_rows.columns),
    ]
    _require_columns(base_records, REQUIRED_BASE_COLUMNS, context="USR base records")
    _require_columns(scored_rows, required_scored_columns, context="Reader scored SFXI rows")

    base = base_records.loc[:, list(REQUIRED_BASE_COLUMNS)].copy()
    scored = scored_rows.loc[:, selected_scored_columns].copy()
    scored = scored.loc[:, ~scored.columns.duplicated()].copy()
    base["_sequence_norm"] = [_normalized_dna_sequence(value) for value in base["sequence"]]
    scored["_sequence_norm"] = [_normalized_dna_sequence(value) for value in scored["sequence"]]
    _validate_unique_normalized_sequences(base, context="USR base records")
    _validate_unique_normalized_sequences(scored, context="Reader scored SFXI rows")

    merged = scored.merge(base[["id", "_sequence_norm"]], on="_sequence_norm", how="left", validate="one_to_one")
    missing = merged.loc[merged["id"].isna(), ["design_id", "sequence"]].head(5).to_dict(orient="records")
    if missing:
        raise SchemaError(f"Reader scored SFXI row(s) do not map to USR base records by sequence: {missing}")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        metric_value = _finite_float(row[metric_column], context="SFXI metric value")
        metric_id = metric_id_for_setpoint(
            objective_name=row["objective_name"],
            setpoint_name=row["setpoint_name"],
            metric_column=metric_column,
        )
        reference_instance_id = str(row["design_id"]).strip()
        if not reference_instance_id:
            raise SchemaError("SFXI overlay requires non-empty reference_instance_id values.")
        reader_experiment_id = _optional_string(row.get("experiment_id"))
        rows.append(
            {
                "id": str(row["id"]),
                "sfxi_ref__reference_instance_id": reference_instance_id,
                "sfxi_ref__collection_id": str(collection_id),
                "sfxi_ref__batch_id": reader_experiment_id,
                "sfxi_ref__campaign_id": str(campaign_id),
                "sfxi_ref__reader_experiment_id": reader_experiment_id,
                "sfxi_ref__reader_experiment_date": _optional_int(row.get("experiment_date")),
                "sfxi_ref__metric_id": metric_id,
                "sfxi_ref__metric_value": metric_value,
                "sfxi_ref__metric_provenance": str(metric_provenance),
                "sfxi_ref__source_ref": str(source_ref),
                "sfxi_ref__score_ref": str(score_ref),
                "sfxi_ref__objective_name": str(row["objective_name"]),
                "sfxi_ref__api_version": str(row["api_version"]),
                "sfxi_ref__state_order": _state_order(row["state_order"]),
                "sfxi_ref__setpoint_name": str(row["setpoint_name"]),
                "sfxi_ref__setpoint_vector": _setpoint_vector(row["setpoint_vector"]),
                "sfxi_ref__denom_used": _finite_float(row["denom_used"], context="SFXI denom_used"),
                "sfxi_ref__denom_percentile": _optional_int(row["denom_percentile"]),
                "sfxi_ref__logic_fidelity": _finite_float(row["logic_fidelity"], context="SFXI logic_fidelity"),
                "sfxi_ref__effect_raw": _finite_float(row["effect_raw"], context="SFXI effect_raw"),
                "sfxi_ref__effect_scaled": _finite_float(row["effect_scaled"], context="SFXI effect_scaled"),
                "sfxi_ref__sfxi": metric_value
                if metric_column == DEFAULT_METRIC_COLUMN
                else _optional_float(row.get(DEFAULT_METRIC_COLUMN)),
                "sfxi_ref__r_logic": _optional_float(row.get("r_logic")),
                "sfxi_ref__time_selected_h": _optional_float(row.get("time_selected_h")),
                "sfxi_ref__reference_design_id": _optional_string(row.get("reference_design_id")),
                "sfxi_ref__sequence_source_id": _optional_string(row.get("sequence_source_id")),
                "sfxi_ref__clip_lo_mask": _optional_bool(row.get("clip_lo_mask")),
                "sfxi_ref__clip_hi_mask": _optional_bool(row.get("clip_hi_mask")),
                "sfxi_ref__intensity_disabled": _optional_bool(row.get("intensity_disabled")),
                "sfxi_ref__flat_logic": _optional_bool(row.get("flat_logic")),
                "sfxi_ref__id_pair": f"{reference_instance_id}\n{metric_id}",
            }
        )

    frame = pd.DataFrame(rows)
    _validate_metric_contract(frame)
    frame = frame.drop(columns=["sfxi_ref__id_pair"]).sort_values("id").reset_index(drop=True)
    return frame


def sfxi_reference_overlay_schema(usr_root: Path | None = None) -> pa.Schema:
    columns = SFXI_REF_COLUMNS
    if usr_root is not None:
        columns = registry_entry(load_registry(usr_root, required=True), SFXI_OVERLAY_NAMESPACE).columns
    return pa.schema(
        [pa.field("id", pa.string())] + [pa.field(column.name, arrow_type_from_str(column.type)) for column in columns]
    )


def _clean_for_arrow(value: object, target_type: pa.DataType) -> object:
    if _is_missing(value):
        return None
    if pa.types.is_string(target_type):
        return str(value)
    if pa.types.is_integer(target_type):
        return _optional_int(value)
    if pa.types.is_floating(target_type):
        return _finite_float(value, context="SFXI Arrow float field")
    if pa.types.is_boolean(target_type):
        return _optional_bool(value)
    if pa.types.is_list(target_type) or pa.types.is_large_list(target_type):
        return [
            _clean_for_arrow(item, target_type.value_type) for item in _parse_list(value, context="SFXI list field")
        ]
    return value


def sfxi_reference_overlay_table(frame: pd.DataFrame, *, usr_root: Path | None = None) -> pa.Table:
    schema = sfxi_reference_overlay_schema(usr_root)
    _require_columns(frame, schema.names, context="SFXI overlay frame")
    rows = [
        {field.name: _clean_for_arrow(row.get(field.name), field.type) for field in schema}
        for row in frame.to_dict(orient="records")
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def validate_sfxi_reference_overlay_contract(frame: pd.DataFrame, *, usr_root: Path) -> pa.Table:
    table = sfxi_reference_overlay_table(frame, usr_root=usr_root)
    validate_overlay_schema(
        SFXI_OVERLAY_NAMESPACE,
        table.schema,
        registry=load_registry(usr_root, required=True),
        key="id",
    )
    return table


def read_usr_base_records(usr_root: Path, dataset_name: str) -> pd.DataFrame:
    dataset = Dataset(usr_root, dataset_name)
    if not dataset.records_path.exists():
        raise FileNotFoundError(f"USR dataset records are missing: {dataset.records_path}")
    return pq.read_table(dataset.records_path, columns=list(REQUIRED_BASE_COLUMNS)).to_pandas()


def read_reader_vec8(vec8_path: Path) -> pd.DataFrame:
    if not vec8_path.exists():
        raise FileNotFoundError(f"Reader SFXI vec8 artifact is missing: {vec8_path}")
    return pq.read_table(vec8_path).to_pandas()


def score_reader_vec8(
    vec8: pd.DataFrame,
    *,
    reader_root: Path,
    setpoint_name: str = DEFAULT_SETPOINT_NAME,
    setpoint_vector: Sequence[float] = DEFAULT_SETPOINT_VECTOR,
) -> pd.DataFrame:
    reader_src = reader_root / "src"
    if reader_src.exists() and str(reader_src) not in sys.path:
        sys.path.insert(0, str(reader_src))
    try:
        from reader.domains.logic.sfxi.setpoint_scatter import score_sfxi_setpoints
    except ImportError as exc:
        raise SchemaError(
            "Unable to import Reader SFXI public API. Pass --reader-root pointing at the sibling reader checkout."
        ) from exc
    return score_sfxi_setpoints(vec8, setpoints={setpoint_name: list(setpoint_vector)})


def build_overlay_from_reader(
    *,
    usr_root: Path,
    dataset_name: str,
    reader_root: Path,
    vec8_path: Path,
    collection_id: str,
    campaign_id: str,
    setpoint_name: str = DEFAULT_SETPOINT_NAME,
    setpoint_vector: Sequence[float] = DEFAULT_SETPOINT_VECTOR,
) -> pd.DataFrame:
    reader_root = reader_root.expanduser().resolve()
    vec8_path = vec8_path.expanduser().resolve()
    base_records = read_usr_base_records(usr_root, dataset_name)
    vec8 = read_reader_vec8(vec8_path)
    scored = score_reader_vec8(
        vec8,
        reader_root=reader_root,
        setpoint_name=setpoint_name,
        setpoint_vector=setpoint_vector,
    )
    return build_sfxi_reference_overlay_frame(
        base_records=base_records,
        scored_rows=scored,
        collection_id=collection_id,
        campaign_id=campaign_id,
        source_ref=str(vec8_path),
    )


def write_sfxi_reference_overlay(
    *,
    usr_root: Path,
    dataset_name: str,
    frame: pd.DataFrame,
) -> int:
    dataset = Dataset(usr_root, dataset_name)
    existing = overlay_path(dataset.dir, SFXI_OVERLAY_NAMESPACE)
    existing_parts = dataset.dir / "_derived" / SFXI_OVERLAY_NAMESPACE
    if existing.exists() or existing_parts.exists():
        raise FileExistsError(
            f"SFXI reference overlay already exists for {dataset.name}: {existing}. "
            "Refuse to append without an explicit replacement workflow."
        )
    return dataset.write_overlay_part(
        SFXI_OVERLAY_NAMESPACE,
        validate_sfxi_reference_overlay_contract(frame, usr_root=usr_root),
        key="id",
        allow_missing=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an additive USR SFXI reference metric overlay from Reader.")
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--reader-root", type=Path, default=_default_reader_root())
    parser.add_argument("--dataset", default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--vec8", type=Path, default=None)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--setpoint-name", default=DEFAULT_SETPOINT_NAME)
    parser.add_argument(
        "--setpoint-vector",
        type=float,
        nargs=4,
        default=list(DEFAULT_SETPOINT_VECTOR),
        metavar=("Y00", "Y10", "Y01", "Y11"),
    )
    parser.add_argument("--expected-count", type=int, default=23)
    parser.add_argument("--write", action="store_true", help="Write the additive sfxi_ref overlay. Default is dry-run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vec8_path = args.vec8 or (args.reader_root / DEFAULT_READER_VEC8)
    frame = build_overlay_from_reader(
        usr_root=args.usr_root,
        dataset_name=args.dataset,
        reader_root=args.reader_root,
        vec8_path=vec8_path,
        collection_id=args.collection_id,
        campaign_id=args.campaign_id,
        setpoint_name=args.setpoint_name,
        setpoint_vector=args.setpoint_vector,
    )
    if args.expected_count is not None and len(frame) != args.expected_count:
        raise SchemaError(f"Expected {args.expected_count} SFXI overlay rows, found {len(frame)}.")
    validate_sfxi_reference_overlay_contract(frame, usr_root=args.usr_root)
    metric_ids = sorted(set(str(value) for value in frame["sfxi_ref__metric_id"]))
    written = False
    if args.write:
        write_sfxi_reference_overlay(usr_root=args.usr_root, dataset_name=args.dataset, frame=frame)
        written = True
    result = SFXIReferenceOverlayResult(
        dataset=args.dataset,
        namespace=SFXI_OVERLAY_NAMESPACE,
        source_vec8=str(vec8_path),
        collection_id=args.collection_id,
        campaign_id=args.campaign_id,
        metric_id=metric_ids[0] if len(metric_ids) == 1 else ",".join(metric_ids),
        rows=len(frame),
        registry_validated=True,
        written=written,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
