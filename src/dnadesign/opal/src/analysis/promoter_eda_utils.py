"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/promoter_eda_utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import yaml

from ..core.round_context import PluginRegistryView, RoundCtx
from ..registries.transforms_y import run_y_ops_pipeline
from .facade import read_predictions, read_runs


@dataclass(frozen=True)
class NumericRule:
    enabled: bool
    column: str | None
    op: str
    value: float | None = None


SUPPORTED_NUMERIC_OPS = {">=", "<=", "is null", "is not null"}
_OBJECTIVE_MODES = {"maximize", "minimize"}


def resolve_objective_mode(selection_params: Mapping[str, Any]) -> tuple[str, list[str]]:
    """
    Resolve objective_mode with legacy alias support and warnings.
    Returns (mode, warnings).
    """
    warnings: list[str] = []
    mode_raw = selection_params.get("objective_mode")
    legacy_raw = selection_params.get("objective")
    if mode_raw is not None and legacy_raw is not None:
        mode_str = str(mode_raw).strip().lower()
        legacy_str = str(legacy_raw).strip().lower()
        if mode_str != legacy_str:
            raise ValueError(
                "selection.params has both 'objective_mode' and legacy 'objective' with conflicting values "
                f"({mode_str!r} vs {legacy_str!r})"
            )
    if mode_raw is None and legacy_raw is not None:
        warnings.append("selection.params.objective is deprecated; prefer selection.params.objective_mode.")
        mode_raw = legacy_raw
    if mode_raw is None:
        mode_raw = "maximize"
    mode = str(mode_raw).strip().lower()
    if mode not in _OBJECTIVE_MODES:
        warnings.append(f"Unknown objective mode {mode!r}; defaulting to 'maximize'.")
        mode = "maximize"
    return mode, warnings


def find_repo_root(start: Path) -> Path | None:
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def resolve_usr_root(repo_root: Path | None, env_override: str | None) -> Path:
    if env_override:
        override_path = Path(env_override).expanduser().resolve()
        if not override_path.exists():
            raise ValueError(f"DNADESIGN_USR_ROOT does not exist: {override_path}")
        return override_path
    if repo_root is None:
        raise ValueError("Could not find repo root (pyproject.toml). Provide an absolute path.")
    return repo_root / "src" / "dnadesign" / "usr" / "datasets"


def list_usr_datasets(usr_root: Path) -> list[str]:
    if not usr_root.exists():
        return []
    datasets: list[str] = []
    for child in usr_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "records.parquet").is_file():
            datasets.append(child.name)
    return sorted(datasets)


def resolve_dataset_path(
    *,
    repo_root: Path | None,
    usr_root: Path | None,
    dataset_name: str | None,
    custom_path: str | None,
) -> tuple[Path, str]:
    custom = (custom_path or "").strip()
    if custom:
        custom_path_obj = Path(custom).expanduser()
        if custom_path_obj.is_absolute():
            return custom_path_obj, "custom"
        if repo_root is None:
            raise ValueError("Relative custom paths require a repo root.")
        return (repo_root / custom_path_obj).resolve(), "custom"
    if usr_root is None:
        raise ValueError("USR root is unavailable; provide a custom path.")
    if not dataset_name or dataset_name in {"(none found)", "(none)"}:
        raise ValueError("Select a dataset or provide a custom path.")
    return (usr_root / dataset_name / "records.parquet").resolve(), "usr"


def namespace_summary(columns: Sequence[str], max_examples: int = 3) -> pl.DataFrame:
    buckets: dict[str, list[str]] = {}
    for name in columns:
        if "__" in name:
            namespace = name.split("__", 1)[0]
        else:
            namespace = "core"
        buckets.setdefault(namespace, []).append(name)
    rows = []
    for namespace, cols in sorted(buckets.items()):
        cols_sorted = sorted(cols)
        examples = ", ".join(cols_sorted[:max_examples])
        rows.append({"namespace": namespace, "count": len(cols), "examples": examples})
    if not rows:
        return pl.DataFrame({"namespace": [], "count": [], "examples": []})
    return pl.DataFrame(rows)


def missingness_summary(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame({"column": [], "null_pct": [], "non_null_count": []})
    total = df.height
    null_counts = df.null_count()
    null_long = null_counts.transpose(
        include_header=True,
        header_name="column",
        column_names=["null_count"],
    )
    return (
        null_long.with_columns(
            (pl.col("null_count") / total * 100).alias("null_pct"),
            (pl.lit(total) - pl.col("null_count")).alias("non_null_count"),
        )
        .select(["column", "null_pct", "non_null_count"])
        .sort("null_pct", descending=True)
    )


def opal_labeled_mask(df: pl.DataFrame, label_hist_cols: Sequence[str]) -> pl.Series:
    if not label_hist_cols:
        return pl.Series([False] * df.height)
    exprs = [(pl.col(col).is_not_null()) & (pl.col(col).list.len().fill_null(0) > 0) for col in label_hist_cols]
    return df.select(pl.any_horizontal(exprs).alias("opal_labeled"))["opal_labeled"]


def build_numeric_rule_exprs(rules: Sequence[NumericRule]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for rule in rules:
        if not rule.enabled:
            continue
        if not rule.column:
            continue
        if rule.op not in SUPPORTED_NUMERIC_OPS:
            raise ValueError(f"Unsupported operator: {rule.op}")
        if rule.op == ">=":
            if rule.value is None:
                continue
            exprs.append(pl.col(rule.column) >= float(rule.value))
        elif rule.op == "<=":
            if rule.value is None:
                continue
            exprs.append(pl.col(rule.column) <= float(rule.value))
        elif rule.op == "is null":
            exprs.append(pl.col(rule.column).is_null())
        elif rule.op == "is not null":
            exprs.append(pl.col(rule.column).is_not_null())
    return exprs


def apply_numeric_rules(df: pl.DataFrame, rules: Sequence[NumericRule]) -> pl.DataFrame:
    exprs = build_numeric_rule_exprs(rules)
    if not exprs:
        return df
    return df.filter(pl.all_horizontal(exprs))


@dataclass(frozen=True)
class CampaignInfo:
    label: str
    path: Path
    workdir: Path | None
    slug: str
    x_column: str
    y_column: str
    y_expected_length: int | None
    model_name: str
    model_params: dict
    objective_name: str
    objective_params: dict
    selection_name: str
    selection_params: dict
    training_policy: dict
    y_ops: list[dict]


@dataclass(frozen=True)
class CampaignDatasetRef:
    campaign_label: str
    campaign_path: Path
    kind: str | None
    dataset_name: str | None
    records_path: Path | None


@dataclass(frozen=True)
class YOpEntry:
    name: str
    params: dict


def normalize_y_ops_config(y_ops: Sequence[Mapping[str, Any]]) -> list[YOpEntry]:
    out: list[YOpEntry] = []
    for entry in y_ops or []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        params = dict(entry.get("params") or {})
        out.append(YOpEntry(name=str(name), params=params))
    return out


def build_round_ctx_for_notebook(
    *,
    info: CampaignInfo,
    run_id: str,
    round_index: int,
    y_dim: int,
    n_train: int,
) -> RoundCtx:
    registry = PluginRegistryView(
        model=info.model_name,
        objective=info.objective_name,
        selection=info.selection_name,
        transform_x="unknown",
        transform_y="unknown",
    )
    ctx = RoundCtx(
        core={
            "core/run_id": str(run_id),
            "core/round_index": int(round_index),
            "core/campaign_slug": info.slug,
            "core/labels_as_of_round": int(round_index),
            "core/plugins/transforms_x/name": registry.transform_x,
            "core/plugins/transforms_y/name": registry.transform_y,
            "core/plugins/model/name": registry.model,
            "core/plugins/objective/name": registry.objective,
            "core/plugins/selection/name": registry.selection,
            "core/data/y_dim": int(y_dim),
            "core/data/n_train": int(n_train),
        },
        registry=registry,
    )
    return ctx


def apply_y_ops_fit_transform(
    *,
    y_ops: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    ctx: RoundCtx,
) -> np.ndarray:
    entries = normalize_y_ops_config(y_ops)
    return run_y_ops_pipeline(stage="fit_transform", y_ops=entries, Y=y, ctx=ctx)


def apply_y_ops_inverse(
    *,
    y_ops: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    ctx: RoundCtx,
) -> np.ndarray:
    entries = normalize_y_ops_config(y_ops)
    return run_y_ops_pipeline(stage="inverse", y_ops=entries, Y=y, ctx=ctx)


def list_campaign_paths(repo_root: Path | None) -> list[Path]:
    if repo_root is None:
        return []
    campaigns_root = repo_root / "src" / "dnadesign" / "opal" / "campaigns"
    if not campaigns_root.exists():
        return []
    return sorted(campaigns_root.rglob("campaign.yaml"))


def list_campaign_dataset_refs(repo_root: Path | None) -> list[CampaignDatasetRef]:
    refs: list[CampaignDatasetRef] = []
    for campaign_path in list_campaign_paths(repo_root):
        campaign_label = campaign_label_from_path(campaign_path, repo_root)
        try:
            raw = load_campaign_yaml(campaign_path)
        except Exception:
            continue
        data = raw.get("data") or {}
        location = data.get("location") or {}
        kind = str(location.get("kind")) if location.get("kind") is not None else None
        dataset_name = None
        records_path = None
        if kind == "usr":
            dataset_name = location.get("dataset")
            base_path_raw = location.get("path")
            base_path = Path(str(base_path_raw)) if base_path_raw else None
            if base_path is not None and not base_path.is_absolute():
                base_path = (campaign_path.parent / base_path).resolve()
            if base_path is not None and dataset_name:
                records_path = (base_path / str(dataset_name) / "records.parquet").resolve()
        elif kind == "local":
            local_path_raw = location.get("path")
            if local_path_raw:
                local_path = Path(str(local_path_raw))
                if not local_path.is_absolute():
                    local_path = (campaign_path.parent / local_path).resolve()
                records_path = local_path
        refs.append(
            CampaignDatasetRef(
                campaign_label=campaign_label,
                campaign_path=campaign_path,
                kind=kind,
                dataset_name=str(dataset_name) if dataset_name else None,
                records_path=records_path,
            )
        )
    return refs


def campaign_label_from_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(path)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def load_campaign_yaml(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Campaign config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Campaign YAML must be a mapping.")
    return raw


def parse_campaign_info(*, raw: dict, path: Path, label: str) -> CampaignInfo:
    campaign = raw.get("campaign") or {}
    slug = campaign.get("slug")
    if not slug:
        raise ValueError("Campaign YAML missing campaign.slug.")
    workdir = None
    workdir_raw = campaign.get("workdir")
    if workdir_raw:
        workdir_path = Path(str(workdir_raw))
        if not workdir_path.is_absolute():
            workdir_path = (path.parent / workdir_path).resolve()
        workdir = workdir_path
    data = raw.get("data") or {}
    x_column = data.get("x_column_name")
    y_column = data.get("y_column_name")
    if not x_column or not y_column:
        raise ValueError("Campaign YAML missing data.x_column_name or data.y_column_name.")
    y_expected_length = data.get("y_expected_length")

    model_block = raw.get("model") or {}
    model_name = model_block.get("name") or "random_forest"
    model_params = dict(model_block.get("params") or {})

    objective_block = raw.get("objective") or {}
    objective_name = objective_block.get("name") or "sfxi_v1"
    objective_params = dict(objective_block.get("params") or {})

    selection_block = raw.get("selection") or {}
    selection_name = selection_block.get("name") or "top_k"
    selection_params = dict(selection_block.get("params") or {})

    training_block = raw.get("training") or {}
    training_policy = dict(training_block.get("policy") or {})
    y_ops = list(training_block.get("y_ops") or [])

    return CampaignInfo(
        label=label,
        path=path,
        workdir=workdir,
        slug=str(slug),
        x_column=str(x_column),
        y_column=str(y_column),
        y_expected_length=int(y_expected_length) if y_expected_length is not None else None,
        model_name=str(model_name),
        model_params=model_params,
        objective_name=str(objective_name),
        objective_params=objective_params,
        selection_name=str(selection_name),
        selection_params=selection_params,
        training_policy=training_policy,
        y_ops=y_ops,
    )


def resolve_campaign_workdir(info: CampaignInfo) -> Path:
    if info.workdir is not None:
        return info.workdir
    return info.path.parent


def find_latest_model_artifact(info: CampaignInfo) -> tuple[Path | None, Path | None]:
    outputs_dir = resolve_campaign_workdir(info) / "outputs"
    if not outputs_dir.exists():
        return None, None
    candidates: list[tuple[int, Path, Path]] = []
    for round_dir in outputs_dir.glob("round_*"):
        model_path = round_dir / "model.joblib"
        if not model_path.is_file():
            continue
        try:
            round_idx = int(round_dir.name.split("_")[1])
        except Exception:
            round_idx = -1
        candidates.append((round_idx, model_path, round_dir))
    if not candidates:
        return None, None
    _, model_path, round_dir = max(candidates, key=lambda item: item[0])
    return model_path, round_dir


def load_model_artifact(path: Path):
    try:
        import joblib

        return joblib.load(path), None
    except Exception as exc:
        return None, str(exc)


def unwrap_artifact_model(obj):
    if obj is None:
        return None
    if hasattr(obj, "predict"):
        return obj
    model_attr = getattr(obj, "model", None)
    if model_attr is not None and hasattr(model_attr, "predict"):
        return model_attr
    if isinstance(obj, dict):
        for key in ("model", "estimator", "rf", "random_forest"):
            candidate = obj.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
    return None


def get_feature_importances(model):
    if model is None:
        return None
    if hasattr(model, "feature_importances_"):
        return getattr(model, "feature_importances_")
    if hasattr(model, "feature_importances"):
        try:
            return model.feature_importances()
        except Exception:
            return None
    return None


def load_intensity_params_from_round_ctx(round_dir: Path, *, eps_default: float):
    ctx_path = round_dir / "round_ctx.json"
    if not ctx_path.is_file():
        return None
    try:
        data = json.loads(ctx_path.read_text())
    except Exception:
        return None
    med = data.get("yops/intensity_median_iqr/center")
    scale = data.get("yops/intensity_median_iqr/scale")
    if med is None or scale is None:
        return None
    med_arr = np.asarray(med, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    if med_arr.shape != (4,) or scale_arr.shape != (4,):
        return None
    enabled = bool(data.get("yops/intensity_median_iqr/enabled", False))
    eps_val = float(data.get("yops/intensity_median_iqr/eps", eps_default))
    return med_arr, scale_arr, enabled, eps_val


def load_round_ctx_from_dir(round_dir: Path) -> tuple[RoundCtx | None, str | None]:
    ctx_path = round_dir / "round_ctx.json"
    if not ctx_path.is_file():
        return None, f"round_ctx.json not found under {round_dir}"
    try:
        snapshot = json.loads(ctx_path.read_text())
    except Exception as exc:
        return None, f"round_ctx.json read failed: {exc}"
    try:
        return RoundCtx.from_snapshot(snapshot), None
    except Exception as exc:
        return None, f"round_ctx.json parse failed: {exc}"


def _label_hist_sample_value(df: pl.DataFrame, label_hist_col: str) -> str | None:
    try:
        series = df.select(pl.col(label_hist_col).drop_nulls()).to_series()
    except Exception:
        return None
    if series.is_empty():
        return None
    sample = series.head(1).to_list()
    if not sample:
        return None
    try:
        return json.dumps(sample[0])
    except Exception:
        return repr(sample[0])


def _deep_as_py(x: Any) -> Any:
    try:
        if hasattr(x, "as_py"):
            return x.as_py()
        if hasattr(x, "to_pylist"):
            return x.to_pylist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return [_deep_as_py(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pl.Series):
        return [_deep_as_py(v) for v in x.to_list()]
    if isinstance(x, dict):
        return {k: _deep_as_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_deep_as_py(v) for v in x]
    return x


def _coerce_mapping(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    for attr in ("as_dict", "to_dict"):
        if hasattr(value, attr):
            try:
                return dict(getattr(value, attr)())
            except Exception:
                continue
    return None


def _parse_label_hist_cell(
    cell: Any,
    *,
    row_id: str,
    errors: list[dict],
    y_col_name: str,
) -> list[dict]:
    def _record_error(message: str, sample: Any | None = None) -> None:
        if len(errors) >= 5:
            return
        errors.append(
            {
                "id": row_id,
                "error": message,
                "sample": (repr(sample)[:240] if sample is not None else None),
            }
        )

    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    cell = _deep_as_py(cell)
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except Exception as exc:
            _record_error(f"label_hist JSON parse failed: {exc}", sample=cell)
            return []
    if isinstance(cell, dict):
        entries = [cell]
    elif isinstance(cell, (list, tuple)):
        entries = list(cell)
    else:
        _record_error(f"label_hist cell must be list/dict/JSON, got {type(cell).__name__}", sample=cell)
        return []

    out: list[dict] = []
    for entry in entries:
        entry = _deep_as_py(entry)
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception as exc:
                _record_error(f"label_hist entry JSON parse failed: {exc}", sample=entry)
                continue
        entry_map = _coerce_mapping(entry)
        if entry_map is None:
            _record_error("label_hist entry must be dict-like", sample=entry)
            continue
        r = entry_map.get("r", entry_map.get("round", entry_map.get("observed_round")))
        if r is None:
            _record_error("label_hist entry missing round key ('r')", sample=entry_map)
            continue
        try:
            r_int = int(r)
        except Exception as exc:
            _record_error(f"label_hist entry round is not int: {exc}", sample=entry_map)
            continue
        y_val = entry_map.get("y", entry_map.get("y_obs", entry_map.get("value")))
        if y_val is None:
            _record_error("label_hist entry missing 'y'", sample=entry_map)
            continue
        try:
            y_list = [float(v) for v in np.asarray(y_val, dtype=float).ravel().tolist()]
        except Exception as exc:
            _record_error(f"label_hist entry 'y' not numeric: {exc}", sample=y_val)
            continue
        out.append(
            {
                "observed_round": r_int,
                "label_src": entry_map.get("src", entry_map.get("source")),
                "label_ts": entry_map.get("ts", entry_map.get("timestamp")),
                y_col_name: y_list,
            }
        )
    return out


def build_label_events(
    *,
    df: pl.DataFrame,
    label_hist_col: str,
    y_col_name: str,
    id_col: str = "id",
    sequence_col: str = "sequence",
) -> tuple[pl.DataFrame, dict]:
    diagnostics = {
        "status": "ok",
        "label_hist_col": label_hist_col,
        "dtype": None,
        "schema": None,
        "sample": None,
        "rows_with_labels": 0,
        "events_parsed": 0,
        "errors": [],
        "exception": None,
        "suggested_remediation": (
            "Run `opal label-hist repair` or re-ingest labels with `opal ingest-y` if label_hist is malformed."
        ),
    }
    if df.is_empty():
        diagnostics["status"] = "empty_df"
        diagnostics["message"] = "Input DataFrame is empty; no labels to parse."
        return df.head(0), diagnostics
    if label_hist_col not in df.columns:
        diagnostics["status"] = "missing_column"
        diagnostics["message"] = f"Missing label history column '{label_hist_col}'."
        return df.head(0), diagnostics

    diagnostics["dtype"] = str(df.schema.get(label_hist_col))
    diagnostics["schema"] = {name: str(dtype) for name, dtype in df.schema.items()}
    diagnostics["sample"] = _label_hist_sample_value(df, label_hist_col)

    select_cols = [col for col in df.columns if col != label_hist_col]
    if label_hist_col not in select_cols:
        select_cols.append(label_hist_col)

    rows: list[dict] = []
    errors: list[dict] = []
    try:
        for row in df.select(select_cols).iter_rows(named=True):
            _id = str(row.get(id_col))
            cell = row.get(label_hist_col)
            if cell is None:
                continue
            parsed = _parse_label_hist_cell(cell, row_id=_id, errors=errors, y_col_name=y_col_name)
            if parsed:
                diagnostics["rows_with_labels"] += 1
            for entry in parsed:
                for col in select_cols:
                    if col == label_hist_col:
                        continue
                    if col not in entry:
                        entry[col] = row.get(col)
                entry["id"] = _id
                rows.append(entry)
    except Exception as exc:
        diagnostics["status"] = "error"
        diagnostics["message"] = "Label history parsing failed."
        diagnostics["exception"] = str(exc)
        diagnostics["errors"] = errors
        empty_schema = {
            "id": pl.Utf8,
            "observed_round": pl.Int64,
            "label_src": pl.Utf8,
            "label_ts": pl.Utf8,
            y_col_name: pl.Object,
        }
        if sequence_col in select_cols:
            empty_schema["sequence"] = pl.Utf8
        return pl.DataFrame(schema=empty_schema), diagnostics

    diagnostics["events_parsed"] = len(rows)
    diagnostics["errors"] = errors
    if errors:
        diagnostics["status"] = "parse_warning"
        diagnostics["message"] = "Some label_hist cells could not be parsed; labels may be incomplete."

    if not rows:
        empty_schema = {
            "id": pl.Utf8,
            "observed_round": pl.Int64,
            "label_src": pl.Utf8,
            "label_ts": pl.Utf8,
            y_col_name: pl.Object,
        }
        if sequence_col in select_cols:
            empty_schema["sequence"] = pl.Utf8
        return pl.DataFrame(schema=empty_schema), diagnostics

    return pl.DataFrame(rows), diagnostics


def dedup_latest_labels(df: pl.DataFrame, *, id_col: str, round_col: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    if round_col not in df.columns:
        return df
    return df.sort(round_col).unique(subset=[id_col], keep="last")


def safe_is_numeric(dtype: pl.DataType) -> bool:
    if dtype in (pl.Null, pl.Object):
        return False
    unknown = getattr(pl, "Unknown", None)
    if unknown is not None and dtype == unknown:
        return False
    try:
        return bool(dtype.is_numeric())
    except Exception:
        return False


_COLOR_DENYLIST = {
    "__row_id",
    "id",
    "id_",
    "id__",
    "densegen__gap_fill_basis",
    "densegen__gap_fill_end",
    "densegen__gap_fill_gc_actual",
    "densegen__gap_fill_gc_max",
    "densegen__gap_fill_gc_min",
    "densegen__gap_fill_relaxed",
    "densegen__gap_fill_used",
    "densegen__library_size",
    "densegen__visual",
    "densegen__used_tfbs_detail",
}
_COLOR_DENY_PREFIXES = ("densegen__gap_fill_", "id__")


def build_color_dropdown_options(
    df: pl.DataFrame,
    *,
    extra: Sequence[str] | None = None,
    include_none: bool = False,
) -> list[str]:
    options: list[str] = []
    for name, dtype in df.schema.items():
        if name.startswith("__"):
            continue
        if name in _COLOR_DENYLIST:
            continue
        if any(name.startswith(prefix) for prefix in _COLOR_DENY_PREFIXES):
            continue
        _is_nested = False
        try:
            _is_nested = bool(getattr(dtype, "is_nested")())
        except Exception:
            _is_nested = False
        if _is_nested:
            continue
        if safe_is_numeric(dtype) or dtype in (
            pl.Boolean,
            pl.String,
            pl.Categorical,
            pl.Enum,
            pl.Date,
            pl.Datetime,
        ):
            options.append(name)
    if extra:
        for name in extra:
            if name not in options:
                options.append(name)
    if include_none:
        return ["(none)"] + options
    return options


def list_series_to_numpy(series: pl.Series, *, expected_len: int | None = None):
    if series.is_empty():
        return None
    series_name = series.name or "values"
    try:
        df_wide = series.to_frame(series_name).select(pl.col(series_name).list.to_struct()).unnest(series_name)
        if expected_len is not None and df_wide.width != expected_len:
            return None
        if df_wide.null_count().to_numpy().sum() > 0:
            return None
        arr = df_wide.to_numpy()
        if arr.ndim != 2:
            return None
        return arr.astype(float, copy=False)
    except Exception:
        pass

    values = series.to_list()
    if not values:
        return None
    rows = []
    for v in values:
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1:
            return None
        if expected_len is not None and arr.size != expected_len:
            return None
        rows.append(arr)
    return np.vstack(rows) if rows else None


def fit_intensity_median_iqr(y, *, min_labels: int, eps: float):
    if y.ndim != 2 or y.shape[1] < 8:
        raise ValueError("intensity_median_iqr expects y with shape (n, 8+).")
    if y.shape[0] < int(min_labels):
        return np.zeros(4, dtype=float), np.ones(4, dtype=float), False
    block = y[:, 4:8]
    med = np.median(block, axis=0)
    q75 = np.percentile(block, 75, axis=0)
    q25 = np.percentile(block, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr <= 0, float(eps), iqr)
    return med, iqr, True


def apply_intensity_median_iqr(y, med, iqr, *, eps: float, enabled: bool):
    if not enabled:
        return y
    out = y.copy()
    out[:, 4:8] = (out[:, 4:8] - med[None, :]) / np.maximum(iqr[None, :], float(eps))
    return out


def invert_intensity_median_iqr(y, med, iqr, *, enabled: bool):
    if not enabled:
        return y
    out = y.copy()
    out[:, 4:8] = out[:, 4:8] * iqr[None, :] + med[None, :]
    return out


@dataclass(frozen=True)
class SFXIParams:
    setpoint: tuple[float, float, float, float]
    weights: tuple[float, float, float, float]
    d: float
    beta: float
    gamma: float
    delta: float
    p: float
    fallback_p: float
    min_n: int
    eps: float


@dataclass(frozen=True)
class SFXIResult:
    df: pl.DataFrame
    denom: float
    weights: tuple[float, float, float, float]
    d: float
    pool_size: int
    denom_source: str


def compute_sfxi_params(
    *,
    setpoint: Sequence[float],
    beta: float,
    gamma: float,
    delta: float,
    p: float,
    fallback_p: float,
    min_n: int,
    eps: float,
) -> SFXIParams:
    if len(setpoint) != 4:
        raise ValueError("setpoint must have length 4")
    p0, p1, p2, p3 = (float(x) for x in setpoint)
    total = p0 + p1 + p2 + p3
    if total <= eps:
        weights = (0.0, 0.0, 0.0, 0.0)
    else:
        weights = (p0 / total, p1 / total, p2 / total, p3 / total)
    d = math.sqrt(sum(max(v * v, (1.0 - v) * (1.0 - v)) for v in (p0, p1, p2, p3)))
    if d <= 0:
        d = eps
    return SFXIParams(
        setpoint=(p0, p1, p2, p3),
        weights=weights,
        d=d,
        beta=float(beta),
        gamma=float(gamma),
        delta=float(delta),
        p=float(p),
        fallback_p=float(fallback_p),
        min_n=int(min_n),
        eps=float(eps),
    )


def valid_vec8_mask_expr(vec_col: str) -> pl.Expr:
    vec = pl.col(vec_col)
    len_ok = vec.list.len() == 8
    finite_ok = vec.list.eval(pl.element().is_finite()).list.all()
    return vec.is_not_null() & len_ok & finite_ok


def _effect_raw_expr(vec_col: str, weights: Sequence[float], delta: float) -> pl.Expr:
    y0 = pl.col(vec_col).list.get(4)
    y1 = pl.col(vec_col).list.get(5)
    y2 = pl.col(vec_col).list.get(6)
    y3 = pl.col(vec_col).list.get(7)
    y0_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y0) - delta)
    y1_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y1) - delta)
    y2_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y2) - delta)
    y3_lin = pl.max_horizontal(pl.lit(0.0), (pl.lit(2.0) ** y3) - delta)
    return weights[0] * y0_lin + weights[1] * y1_lin + weights[2] * y2_lin + weights[3] * y3_lin


def compute_sfxi_metrics(
    *,
    df: pl.DataFrame,
    vec_col: str,
    params: SFXIParams,
    denom_pool_df: pl.DataFrame,
) -> SFXIResult:
    if vec_col not in df.columns:
        return SFXIResult(
            df=df.head(0),
            denom=params.eps,
            weights=params.weights,
            d=params.d,
            pool_size=0,
            denom_source="empty",
        )
    valid_mask = valid_vec8_mask_expr(vec_col)
    df_valid = df.filter(valid_mask)

    p0, p1, p2, p3 = params.setpoint
    setpoint_sum = p0 + p1 + p2 + p3
    intensity_disabled = not math.isfinite(setpoint_sum) or setpoint_sum <= 1.0e-12
    v0 = pl.col(vec_col).list.get(0)
    v1 = pl.col(vec_col).list.get(1)
    v2 = pl.col(vec_col).list.get(2)
    v3 = pl.col(vec_col).list.get(3)
    dist = ((v0 - p0) ** 2 + (v1 - p1) ** 2 + (v2 - p2) ** 2 + (v3 - p3) ** 2) ** 0.5
    logic_fidelity = (1.0 - dist / params.d).clip(0.0, 1.0)

    if intensity_disabled:
        df_sfxi = df_valid.with_columns(
            [
                logic_fidelity.alias("logic_fidelity"),
                pl.lit(0.0).alias("effect_raw"),
                pl.lit(1.0).alias("effect_scaled"),
            ]
        ).with_columns((pl.col("logic_fidelity") ** params.beta).alias("score"))
        return SFXIResult(
            df=df_sfxi,
            denom=1.0,
            weights=params.weights,
            d=params.d,
            pool_size=0,
            denom_source="disabled",
        )

    effect_raw_expr = _effect_raw_expr(vec_col, params.weights, params.delta)

    pool_size = 0
    denom_source = "p"
    if vec_col not in denom_pool_df.columns or denom_pool_df.is_empty():
        raise ValueError(f"Need at least min_n={params.min_n} labels in current round to scale intensity; got 0.")

    pool_dtype = denom_pool_df.schema.get(vec_col, pl.Null)
    if pool_dtype == pl.Null:
        raise ValueError(f"Need at least min_n={params.min_n} labels in current round to scale intensity; got 0.")

    pool_valid = denom_pool_df.filter(valid_vec8_mask_expr(vec_col))
    pool_effect = pool_valid.select(effect_raw_expr.alias("effect_raw"))
    pool_size = pool_effect.height
    if pool_size < params.min_n:
        raise ValueError(
            f"Need at least min_n={params.min_n} labels in current round to scale intensity; got {pool_size}."
        )

    denom = float(pool_effect["effect_raw"].quantile(params.p / 100.0, interpolation="nearest"))
    if not math.isfinite(denom):
        raise ValueError("Invalid denom computed (non-finite). Check labels and scaling config.")
    if denom < 0.0:
        raise ValueError("Invalid denom computed (negative). Check labels and scaling config.")
    if not math.isfinite(params.eps) or params.eps <= 0.0:
        raise ValueError(f"eps must be positive and finite; got {params.eps}.")
    denom = max(denom, params.eps)

    df_sfxi = (
        df_valid.with_columns(
            [
                logic_fidelity.alias("logic_fidelity"),
                effect_raw_expr.alias("effect_raw"),
            ]
        )
        .with_columns(((pl.col("effect_raw") / pl.lit(denom)).clip(0.0, 1.0).alias("effect_scaled")))
        .with_columns(
            ((pl.col("logic_fidelity") ** params.beta) * (pl.col("effect_scaled") ** params.gamma)).alias("score")
        )
    )

    return SFXIResult(
        df=df_sfxi,
        denom=denom,
        weights=params.weights,
        d=params.d,
        pool_size=pool_size,
        denom_source=denom_source,
    )


def dedupe_columns(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def dedupe_exprs(exprs: Sequence[pl.Expr]) -> list[pl.Expr]:
    seen: set[str] = set()
    out: list[pl.Expr] = []
    for expr in exprs:
        name = None
        try:
            name = expr.meta.output_name()
        except Exception:
            name = None
        if name is not None:
            if name in seen:
                continue
            seen.add(name)
        out.append(expr)
    return out


def is_altair_undefined(value: object) -> bool:
    if value is None:
        return False
    try:
        import altair as alt

        if value is alt.Undefined:
            return True
    except Exception:
        pass
    cls = getattr(value, "__class__", None)
    return bool(cls is not None and cls.__name__ == "UndefinedType")


def coerce_selection_dataframe(selected_raw: object) -> pl.DataFrame | None:
    if selected_raw is None or is_altair_undefined(selected_raw):
        return None
    if isinstance(selected_raw, pl.DataFrame):
        return selected_raw
    try:
        return pl.from_pandas(selected_raw)
    except Exception:
        return None


def _ledger_paths(workdir: Path | None) -> dict[str, Path | None]:
    if workdir is None:
        return {"runs": None, "labels": None, "preds_dir": None}
    outputs = Path(workdir) / "outputs"
    return {
        "runs": outputs / "ledger.runs.parquet",
        "labels": outputs / "ledger.labels.parquet",
        "preds_dir": outputs / "ledger.predictions",
    }


def load_ledger_runs(workdir: Path | None) -> tuple[pl.DataFrame, dict]:
    diag = {"status": "missing_workdir", "path": None, "rows": 0, "error": None}
    paths = _ledger_paths(workdir)
    runs_path = paths["runs"]
    if runs_path is None:
        return pl.DataFrame(), diag
    diag["path"] = str(runs_path)
    if not runs_path.exists():
        diag["status"] = "missing"
        return pl.DataFrame(), diag
    try:
        df = pl.read_parquet(runs_path)
    except Exception as exc:
        diag["status"] = "error"
        diag["error"] = str(exc)
        return pl.DataFrame(), diag
    diag["status"] = "ok"
    diag["rows"] = df.height
    return df, diag


def load_ledger_labels(workdir: Path | None) -> tuple[pl.DataFrame, dict]:
    diag = {"status": "missing_workdir", "path": None, "rows": 0, "error": None}
    paths = _ledger_paths(workdir)
    labels_path = paths["labels"]
    if labels_path is None:
        return pl.DataFrame(), diag
    diag["path"] = str(labels_path)
    if not labels_path.exists():
        diag["status"] = "missing"
        return pl.DataFrame(), diag
    try:
        df = pl.read_parquet(labels_path)
    except Exception as exc:
        diag["status"] = "error"
        diag["error"] = str(exc)
        return pl.DataFrame(), diag
    diag["status"] = "ok"
    diag["rows"] = df.height
    return df, diag


def load_ledger_predictions(
    workdir: Path | None,
    *,
    run_id: str | None,
    as_of_round: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    diag = {
        "status": "missing_workdir",
        "path": None,
        "rows": 0,
        "error": None,
        "run_id": run_id,
        "as_of_round": as_of_round,
    }
    paths = _ledger_paths(workdir)
    preds_dir = paths["preds_dir"]
    if preds_dir is None:
        return pl.DataFrame(), diag
    diag["path"] = str(preds_dir)
    if not preds_dir.exists():
        diag["status"] = "missing"
        return pl.DataFrame(), diag
    if run_id is None:
        diag["status"] = "missing_run_id"
        diag["error"] = "run_id is required to read ledger predictions without ambiguity."
        return pl.DataFrame(), diag

    try:
        runs_path = paths["runs"]
        runs_df = read_runs(runs_path) if runs_path is not None and runs_path.exists() else None
        df = read_predictions(
            preds_dir,
            round_selector=as_of_round,
            run_id=run_id,
            runs_df=runs_df,
            require_run_id=True,
        )
    except Exception as exc:
        diag["status"] = "error"
        diag["error"] = str(exc)
        return pl.DataFrame(), diag

    diag["status"] = "ok"
    diag["rows"] = df.height
    return df, diag


def _coerce_artifacts_map(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    raw = _deep_as_py(raw)
    if not isinstance(raw, dict):
        return None
    out: dict[str, str] = {}
    for key, val in raw.items():
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            out[str(key)] = str(val[1])
        elif isinstance(val, str):
            out[str(key)] = val
    return out or None


def resolve_run_artifacts(
    ledger_runs_df: pl.DataFrame | None,
    *,
    run_id: str | None,
) -> tuple[dict[str, str] | None, str | None]:
    if ledger_runs_df is None or ledger_runs_df.is_empty():
        return None, "Ledger runs unavailable."
    if not run_id:
        return None, "run_id is required to resolve artifacts."
    if "run_id" not in ledger_runs_df.columns or "artifacts" not in ledger_runs_df.columns:
        return None, "Ledger runs missing required columns (run_id, artifacts)."
    rows = ledger_runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("artifacts")).to_series().to_list()
    if not rows:
        return None, f"run_id {run_id} not found in ledger runs."
    artifacts = _coerce_artifacts_map(rows[0])
    if not artifacts:
        return None, "Artifacts field missing or unparseable."
    return artifacts, None
