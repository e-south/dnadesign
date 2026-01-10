"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workflows/campaign_summary.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from dnadesign.cruncher.config.schema_v2 import CampaignConfig, CruncherConfig
from dnadesign.cruncher.services.campaign_service import (
    CampaignExpansion,
    build_campaign_manifest,
    collect_campaign_metrics,
    expand_campaign,
)
from dnadesign.cruncher.services.run_service import list_runs
from dnadesign.cruncher.utils.analysis_layout import load_summary, resolve_analysis_dir
from dnadesign.cruncher.utils.elites import find_elites_parquet
from dnadesign.cruncher.utils.manifest import load_manifest
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from dnadesign.cruncher.utils.parquet import read_parquet

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd


def _load_pandas_numpy():
    import numpy as np
    import pandas as pd

    return pd, np


def _load_plotting():
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


@dataclass(frozen=True)
class CampaignSummaryResult:
    out_dir: Path
    campaign_id: str
    summary_path: Path
    best_path: Path
    plot_paths: list[Path]
    skipped: list[str]


def summarize_campaign(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    campaign_name: str,
    run_inputs: Optional[Iterable[str]] = None,
    analysis_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
    include_metrics: bool = True,
    skip_missing: bool = False,
    skip_non_campaign: bool = False,
    top_k: int = 10,
) -> CampaignSummaryResult:
    pd, _ = _load_pandas_numpy()
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
    expansion = expand_campaign(
        cfg=cfg,
        config_path=config_path,
        campaign_name=campaign_name,
        include_metrics=include_metrics,
    )
    metrics = expansion.metrics
    if include_metrics:
        needs_info_bits = not metrics or any(metric.info_bits is None for metric in metrics.values())
        if needs_info_bits:
            campaign = _find_campaign(cfg, campaign_name)
            metrics = collect_campaign_metrics(
                cfg=cfg,
                config_path=config_path,
                tf_names=_unique_tfs(expansion),
                source_preference=campaign.selectors.source_preference,
                dataset_preference=campaign.selectors.dataset_preference,
            )

    set_index_map = _set_index_map(expansion)
    runs_root = config_path.parent / cfg.out_dir
    run_dirs, implicit = _resolve_run_dirs(run_inputs, runs_root, cfg, config_path)

    summary_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        row, reason = _summarize_run(
            run_dir=run_dir,
            expansion=expansion,
            set_index_map=set_index_map,
            metrics=metrics if include_metrics else None,
            analysis_id=analysis_id,
            skip_missing=skip_missing,
            skip_non_campaign=skip_non_campaign,
            strict_non_campaign=not implicit,
        )
        if reason:
            skipped.append(reason)
            continue
        if row is not None:
            summary_rows.append(row)

    if not summary_rows:
        msg = "No campaign runs matched the summary criteria."
        if skipped:
            msg = msg + " Skipped runs: " + "; ".join(skipped)
        raise ValueError(msg)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(["set_index", "run_name"], inplace=True, ignore_index=True)

    best_df = _best_rows(summary_df, top_k=top_k)

    output_root = out_dir or (runs_root / "campaigns" / expansion.campaign_id)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_campaign_manifest(expansion=expansion, config_path=config_path)
    if include_metrics and metrics:
        manifest["targets"] = {
            name: {
                "source": metric.source,
                "motif_id": metric.motif_id,
                "matrix_length": metric.matrix_length,
                "site_count": metric.site_count,
                "site_total": metric.site_total,
                "site_kind": metric.site_kind,
                "dataset_id": metric.dataset_id,
                "info_bits": metric.info_bits,
            }
            for name, metric in metrics.items()
        }
    manifest_path = output_root / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    summary_path = output_root / "campaign_summary.csv"
    best_path = output_root / "campaign_best.csv"
    summary_df.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)

    plot_paths = [
        _plot_best_jointscore(best_df, plots_dir / "best_jointscore_bar.png"),
        _plot_tf_coverage(expansion, plots_dir / "tf_coverage_heatmap.png"),
        _plot_pairgrid_overview(summary_df, plots_dir / "pairgrid_overview.png"),
        _plot_joint_trend(summary_df, plots_dir / "joint_trend.png"),
        _plot_pareto_projection(summary_df, plots_dir / "pareto_projection.png"),
    ]

    return CampaignSummaryResult(
        out_dir=output_root,
        campaign_id=expansion.campaign_id,
        summary_path=summary_path,
        best_path=best_path,
        plot_paths=plot_paths,
        skipped=skipped,
    )


def _find_campaign(cfg: CruncherConfig, name: str) -> CampaignConfig:
    for campaign in cfg.campaigns:
        if campaign.name == name:
            return campaign
    available = ", ".join(sorted(c.name for c in cfg.campaigns))
    raise ValueError(f"campaign '{name}' not found. Available campaigns: {available or 'none'}")


def _unique_tfs(expansion: CampaignExpansion) -> list[str]:
    tfs: list[str] = []
    for group in expansion.regulator_sets:
        for tf in group:
            if tf not in tfs:
                tfs.append(tf)
    return tfs


def _set_index_map(expansion: CampaignExpansion) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, tfs in enumerate(expansion.regulator_sets, start=1):
        mapping[_tf_key(tfs)] = idx
    return mapping


def _resolve_run_dirs(
    run_inputs: Optional[Iterable[str]],
    runs_root: Path,
    cfg: CruncherConfig,
    config_path: Path,
) -> tuple[list[Path], bool]:
    if run_inputs:
        expanded = _expand_run_inputs(run_inputs, runs_root)
        if not expanded:
            raise FileNotFoundError("No runs matched the provided --runs inputs.")
        return expanded, False
    runs = list_runs(cfg, config_path, stage="sample")
    run_dirs = [run.run_dir for run in runs]
    return run_dirs, True


def _expand_run_inputs(run_inputs: Iterable[str], runs_root: Path) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in run_inputs:
        raw_str = str(raw)
        has_glob = any(ch in raw_str for ch in ("*", "?", "["))
        if has_glob:
            pattern = raw_str
            if not Path(raw_str).is_absolute():
                pattern = str(runs_root / raw_str)
            for match in sorted(glob.glob(pattern)):
                path = Path(match)
                if path not in seen:
                    resolved.append(path)
                    seen.add(path)
            continue
        path = Path(raw_str)
        if not path.is_absolute():
            path = runs_root / path
        if not path.exists():
            raise FileNotFoundError(f"Run path not found: {path}")
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    return resolved


def _summarize_run(
    *,
    run_dir: Path,
    expansion: CampaignExpansion,
    set_index_map: dict[str, int],
    metrics: Optional[dict[str, object]],
    analysis_id: Optional[str],
    skip_missing: bool,
    skip_non_campaign: bool,
    strict_non_campaign: bool,
) -> tuple[Optional[dict[str, object]], Optional[str]]:
    pd, _ = _load_pandas_numpy()
    try:
        manifest = load_manifest(run_dir)
    except FileNotFoundError as exc:
        if skip_missing:
            return None, f"{run_dir}: {exc}"
        raise
    if manifest.get("stage") != "sample":
        raise ValueError(f"Run '{run_dir.name}' is not a sample run (stage={manifest.get('stage')})")
    reg = manifest.get("regulator_set") or {}
    tfs = reg.get("tfs") or []
    if not tfs:
        raise ValueError(f"Run '{run_dir.name}' missing regulator_set.tfs in run_manifest.json")

    set_key = _tf_key(tfs)
    set_index = set_index_map.get(set_key)
    if set_index is None:
        if skip_non_campaign:
            return None, f"{run_dir.name}: regulator_set not in campaign"
        if strict_non_campaign:
            raise ValueError(f"Run '{run_dir.name}' regulator_set does not match campaign sets.")
        return None, f"{run_dir.name}: regulator_set not in campaign"

    analysis_dir, resolved_id = resolve_analysis_dir(
        run_dir,
        analysis_id=analysis_id,
        latest=analysis_id is None,
    )
    summary = load_summary(analysis_dir / "summary.json", required=True)
    tf_names = summary.get("tf_names") if isinstance(summary, dict) else None
    if tf_names:
        if _tf_key(tf_names) != set_key:
            raise ValueError(
                f"Run '{run_dir.name}' analysis tf_names do not match regulator_set ({tf_names} vs {tfs})."
            )

    score_summary_path = analysis_dir / "tables" / "score_summary.csv"
    joint_metrics_path = analysis_dir / "tables" / "joint_metrics.csv"
    if not score_summary_path.exists() or not joint_metrics_path.exists():
        missing = []
        if not score_summary_path.exists():
            missing.append(str(score_summary_path))
        if not joint_metrics_path.exists():
            missing.append(str(joint_metrics_path))
        message = f"Run '{run_dir.name}' missing required analysis tables: {', '.join(missing)}"
        if skip_missing:
            return None, message
        raise FileNotFoundError(message)

    score_df = pd.read_csv(score_summary_path)
    joint_df = pd.read_csv(joint_metrics_path)
    score_stats = _score_stats(score_df, run_dir.name)
    joint_stats = _joint_stats(joint_df, run_dir.name)
    n_sequences, n_elites = _load_counts(run_dir)

    quality_stats = {}
    if metrics is not None:
        quality_stats = _quality_stats(metrics, tfs, run_dir.name)

    row = {
        "campaign_id": expansion.campaign_id,
        "campaign_name": expansion.name,
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "analysis_id": resolved_id,
        "analysis_dir": str(analysis_dir.resolve()),
        "set_index": set_index,
        "tf_list": ",".join(tfs),
        "n_tfs": len(tfs),
        "n_sequences": n_sequences,
        "n_elites": n_elites,
        **score_stats,
        **joint_stats,
        **quality_stats,
    }
    return row, None


def _load_counts(run_dir: Path) -> tuple[int, int]:
    report_path = run_dir / "report.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text())
        n_sequences = payload.get("n_sequences")
        n_elites = payload.get("n_elites")
        if isinstance(n_sequences, int) and isinstance(n_elites, int):
            return n_sequences, n_elites
    seq_path = run_dir / "sequences.parquet"
    elite_path = find_elites_parquet(run_dir)
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequences.parquet for run '{run_dir.name}'.")
    seq_df = read_parquet(seq_path)
    if "phase" in seq_df.columns:
        seq_df = seq_df[seq_df["phase"] == "draw"].copy()
    n_sequences = int(len(seq_df))
    elites_df = read_parquet(elite_path)
    n_elites = int(len(elites_df))
    return n_sequences, n_elites


def _score_stats(score_df: pd.DataFrame, run_name: str) -> dict[str, object]:
    pd, _ = _load_pandas_numpy()
    required = {"tf", "mean", "median", "std", "min", "max"}
    missing = required.difference(score_df.columns)
    if missing:
        raise ValueError(f"Run '{run_name}' score_summary missing columns: {sorted(missing)}")
    mean_vals = pd.to_numeric(score_df["mean"], errors="coerce")
    median_vals = pd.to_numeric(score_df["median"], errors="coerce")
    stats = {
        "mean_score_min": _safe_stat(mean_vals.min()),
        "mean_score_mean": _safe_stat(mean_vals.mean()),
        "mean_score_max": _safe_stat(mean_vals.max()),
        "median_score_min": _safe_stat(median_vals.min()),
        "median_score_mean": _safe_stat(median_vals.mean()),
        "median_score_max": _safe_stat(median_vals.max()),
        "tf_mean_scores": _format_tf_scores(score_df["tf"], mean_vals),
        "tf_median_scores": _format_tf_scores(score_df["tf"], median_vals),
    }
    return stats


def _joint_stats(joint_df: pd.DataFrame, run_name: str) -> dict[str, object]:
    if joint_df.empty:
        raise ValueError(f"Run '{run_name}' joint_metrics.csv is empty.")
    row = joint_df.iloc[0]
    return {
        "joint_min_best": _safe_stat(row.get("joint_min")),
        "joint_mean_best": _safe_stat(row.get("joint_mean")),
        "joint_hmean_best": _safe_stat(row.get("joint_hmean")),
        "balance_index_best": _safe_stat(row.get("balance_index")),
        "pareto_front_size": _safe_stat(row.get("pareto_front_size"), allow_int=True),
        "pareto_fraction": _safe_stat(row.get("pareto_fraction")),
    }


def _quality_stats(metrics: dict[str, object], tfs: list[str], run_name: str) -> dict[str, object]:
    _, np = _load_pandas_numpy()
    info_bits: list[float] = []
    site_counts: list[int] = []
    matrix_lengths: list[int] = []
    for tf in tfs:
        metric = metrics.get(tf)
        if metric is None:
            raise ValueError(f"Run '{run_name}' missing metrics for TF '{tf}'.")
        info = metric.info_bits
        if info is None:
            raise ValueError(f"Run '{run_name}' missing info_bits for TF '{tf}'.")
        info_bits.append(float(info))
        site_counts.append(int(metric.site_count))
        if metric.matrix_length is not None:
            matrix_lengths.append(int(metric.matrix_length))
    stats = {
        "info_bits_min": _safe_stat(min(info_bits) if info_bits else None),
        "info_bits_mean": _safe_stat(np.mean(info_bits) if info_bits else None),
        "info_bits_max": _safe_stat(max(info_bits) if info_bits else None),
        "site_count_min": _safe_stat(min(site_counts) if site_counts else None),
        "site_count_mean": _safe_stat(np.mean(site_counts) if site_counts else None),
        "site_count_max": _safe_stat(max(site_counts) if site_counts else None),
    }
    if matrix_lengths:
        stats["pwm_len_min"] = _safe_stat(min(matrix_lengths))
        stats["pwm_len_mean"] = _safe_stat(np.mean(matrix_lengths))
        stats["pwm_len_max"] = _safe_stat(max(matrix_lengths))
    else:
        stats["pwm_len_min"] = None
        stats["pwm_len_mean"] = None
        stats["pwm_len_max"] = None
    return stats


def _safe_stat(value: object, *, allow_int: bool = False) -> object:
    _, np = _load_pandas_numpy()
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        if not np.isfinite(float(value)):
            return None
        return int(value) if allow_int else float(value)
    return value


def _format_tf_scores(tf_series: pd.Series, values: pd.Series) -> str:
    _, np = _load_pandas_numpy()
    pieces = []
    for tf, val in zip(tf_series.tolist(), values.tolist()):
        if val is None or not np.isfinite(val):
            pieces.append(f"{tf}=nan")
        else:
            pieces.append(f"{tf}={float(val):.3f}")
    return ";".join(pieces)


def _best_rows(summary_df: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    pd, _ = _load_pandas_numpy()
    if "joint_min_best" not in summary_df.columns:
        return summary_df.head(0)
    df = summary_df.copy()
    df["_joint_min_sort"] = pd.to_numeric(df["joint_min_best"], errors="coerce")
    df["_balance_sort"] = pd.to_numeric(df.get("balance_index_best"), errors="coerce")
    df.sort_values(
        ["_joint_min_sort", "_balance_sort"],
        ascending=False,
        inplace=True,
        ignore_index=True,
    )
    df = df.head(min(top_k, len(df)))
    return df.drop(columns=["_joint_min_sort", "_balance_sort"], errors="ignore")


def _plot_best_jointscore(best_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd, _ = _load_pandas_numpy()
    plt, _ = _load_plotting()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(best_df))))
    if best_df.empty or "joint_min_best" not in best_df.columns:
        ax.text(0.5, 0.5, "No joint metrics available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    labels = best_df["run_name"].tolist()
    values = pd.to_numeric(best_df["joint_min_best"], errors="coerce").fillna(0.0).tolist()
    ax.barh(labels, values, color="steelblue")
    ax.set_xlabel("Best joint_min (elites)")
    ax.set_title("Top runs by joint_min")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_tf_coverage(expansion: CampaignExpansion, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd, _ = _load_pandas_numpy()
    plt, sns = _load_plotting()
    tfs = _unique_tfs(expansion)
    data = []
    for idx, group in enumerate(expansion.regulator_sets, start=1):
        row = [1 if tf in group else 0 for tf in tfs]
        data.append(row)
    df = pd.DataFrame(data, columns=tfs, index=[f"set{idx}" for idx in range(1, len(data) + 1)])
    height = max(3, 0.25 * len(df))
    width = max(6, 0.4 * len(tfs))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(df, cmap="Blues", cbar=False, linewidths=0.5, ax=ax)
    ax.set_title("TF coverage by campaign set")
    ax.set_xlabel("TF")
    ax.set_ylabel("Regulator set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_tf_score_blob(blob: object) -> dict[str, float]:
    _, np = _load_pandas_numpy()
    if not isinstance(blob, str):
        return {}
    output: dict[str, float] = {}
    for item in blob.split(";"):
        piece = item.strip()
        if not piece or "=" not in piece:
            continue
        tf, raw = piece.split("=", 1)
        tf = tf.strip()
        if not tf:
            continue
        value = raw.strip()
        if value.lower() == "nan":
            output[tf] = np.nan
            continue
        try:
            output[tf] = float(value)
        except ValueError:
            continue
    return output


def _score_frame_from_summary(summary_df: pd.DataFrame, column: str) -> pd.DataFrame:
    pd, _ = _load_pandas_numpy()
    if summary_df.empty or column not in summary_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, float]] = []
    index: list[str] = []
    for _, row in summary_df.iterrows():
        rows.append(_parse_tf_score_blob(row.get(column)))
        index.append(str(row.get("run_name", "")))
    df = pd.DataFrame(rows, index=index)
    if df.empty:
        return df
    keep_cols = [col for col in df.columns if df[col].notna().sum() >= 2]
    df = df[keep_cols]
    df = df.dropna(how="all")
    return df


def _plot_pairgrid_overview(summary_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt, sns = _load_plotting()
    df = _score_frame_from_summary(summary_df, "tf_mean_scores")
    if df.empty or df.shape[1] < 2 or len(df) < 2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(
            0.5,
            0.5,
            "Pairgrid overview requires >=2 TFs with scores across >=2 runs.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=0)
    sns.set_style("ticks", {"axes.grid": False})
    grid = sns.PairGrid(df, corner=True, diag_sharey=False, height=2.2)
    grid.map_lower(sns.scatterplot, s=20, alpha=0.6, linewidth=0)
    grid.map_diag(sns.histplot, bins=20)
    grid.fig.suptitle("Campaign TF mean score pairgrid", y=1.02)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    return out_path


def _plot_joint_trend(summary_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd, _ = _load_pandas_numpy()
    plt, _ = _load_plotting()
    if summary_df.empty or "joint_min_best" not in summary_df.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No joint_min_best values available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    df = summary_df.copy()
    df["_joint"] = pd.to_numeric(df["joint_min_best"], errors="coerce")
    df = df.dropna(subset=["_joint"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No valid joint_min_best values", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    df = df.sort_values(["run_name"], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(1, len(df) + 1)
    ax.plot(x, df["_joint"], marker="o", linewidth=1.5, color="steelblue")
    ax.set_xlabel("Run index (sorted)")
    ax.set_ylabel("joint_min_best")
    ax.set_title("Joint score trend across runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_pareto_projection(summary_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd, _ = _load_pandas_numpy()
    plt, _ = _load_plotting()
    required = {"pareto_fraction", "joint_min_best"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Pareto projection requires pareto_fraction + joint_min_best", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    df = summary_df.copy()
    df["_pareto"] = pd.to_numeric(df["pareto_fraction"], errors="coerce")
    df["_joint"] = pd.to_numeric(df["joint_min_best"], errors="coerce")
    df = df.dropna(subset=["_pareto", "_joint"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No valid pareto/joint values", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
    fig, ax = plt.subplots(figsize=(5, 4))
    sizes = None
    if "n_tfs" in df.columns:
        sizes = pd.to_numeric(df["n_tfs"], errors="coerce").fillna(2).tolist()
        sizes = [max(30, float(val) * 10) for val in sizes]
    ax.scatter(df["_pareto"], df["_joint"], s=sizes, alpha=0.7, color="slateblue")
    ax.set_xlabel("Pareto fraction")
    ax.set_ylabel("joint_min_best")
    ax.set_title("Pareto projection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _tf_key(tfs: Iterable[str]) -> str:
    return ",".join(sorted(tf.lower() for tf in tfs))
