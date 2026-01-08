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
from typing import Sequence

import numpy as np
import polars as pl
import yaml


@dataclass(frozen=True)
class NumericRule:
    enabled: bool
    column: str | None
    op: str
    value: float | None = None


SUPPORTED_NUMERIC_OPS = {">=", "<=", "is null", "is not null"}


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
    selection_params: dict
    training_policy: dict
    y_ops: list[dict]


def list_campaign_paths(repo_root: Path | None) -> list[Path]:
    if repo_root is None:
        return []
    campaigns_root = repo_root / "src" / "dnadesign" / "opal" / "campaigns"
    if not campaigns_root.exists():
        return []
    return sorted(campaigns_root.rglob("campaign.yaml"))


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


def build_label_events(
    *,
    df: pl.DataFrame,
    label_hist_col: str,
    y_col_name: str,
) -> pl.DataFrame:
    if label_hist_col not in df.columns:
        return df.head(0)
    try:
        df_events = df.explode(label_hist_col).drop_nulls(label_hist_col)
        if df_events.is_empty():
            return df_events
        df_events = df_events.with_columns(
            [
                pl.col(label_hist_col).struct.field("r").alias("observed_round"),
                pl.col(label_hist_col).struct.field("src").alias("label_src"),
                pl.col(label_hist_col).struct.field("ts").alias("label_ts"),
                pl.col(label_hist_col).struct.field("y").alias(y_col_name),
            ]
        ).drop(label_hist_col)
        return df_events
    except Exception:
        return df.head(0)


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
