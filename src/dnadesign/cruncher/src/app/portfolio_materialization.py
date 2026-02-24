"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/portfolio_materialization.py

Materialize portfolio aggregate tables, plots, and manifest entries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.parquet import write_parquet
from dnadesign.cruncher.app.analyze.metadata import load_pwms_from_config
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.portfolio.layout import (
    portfolio_manifests_dir,
    portfolio_plot_path,
    portfolio_plots_dir,
    portfolio_table_path,
    portfolio_tables_dir,
)
from dnadesign.cruncher.portfolio.schema_models import PortfolioSpec


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _plot_portfolio_elite_showcase(*args, **kwargs):
    from dnadesign.cruncher.portfolio.plots.elite_showcase import (
        plot_portfolio_elite_showcase as _plot,
    )

    return _plot(*args, **kwargs)


def _ensure_required_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        write_parquet(df, path)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported portfolio table format: {path.suffix}")


def _write_tradeoff_plot(source_summary_df: pd.DataFrame, out_path: Path) -> Path | None:
    if source_summary_df.empty:
        return None
    x = pd.to_numeric(source_summary_df["median_min_best_score_norm"], errors="coerce")
    y = pd.to_numeric(source_summary_df["mean_pairwise_hamming_bp"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(x[valid], y[valid], color="#0072B2", alpha=0.9, s=45)
    for idx in source_summary_df.index[valid]:
        ax.annotate(
            str(source_summary_df.loc[idx, "source_id"]),
            (float(x.loc[idx]), float(y.loc[idx])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Median selected elite min TF score (norm)")
    ax.set_ylabel("Mean pairwise Hamming distance (bp)")
    ax.set_title("Source-level tradeoff: score vs sequence diversity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_portfolio_showcase_elites(spec: PortfolioSpec, elite_summary_df: pd.DataFrame) -> pd.DataFrame:
    if elite_summary_df.empty:
        raise ValueError("Portfolio elite showcase requires non-empty elite summary rows.")

    frame = elite_summary_df.copy()
    _ensure_required_columns(
        frame,
        ["source_id", "elite_id", "elite_rank", "source_label", "sequence"],
        context="Portfolio elite showcase elite summary",
    )
    frame["source_id"] = frame["source_id"].astype(str)
    frame["elite_id"] = frame["elite_id"].astype(str)
    frame["elite_rank"] = pd.to_numeric(frame["elite_rank"], errors="coerce")
    if frame["elite_rank"].isna().any():
        raise ValueError("Portfolio elite showcase requires numeric elite_rank values.")
    frame["elite_rank"] = frame["elite_rank"].astype(int)

    source_ids = [str(source.id) for source in spec.sources]
    available_source_ids = set(frame["source_id"].tolist())
    processed_source_ids = [source_id for source_id in source_ids if source_id in available_source_ids]
    if not processed_source_ids:
        raise ValueError("Portfolio elite showcase found no processed sources to render.")

    selected_parts: list[pd.DataFrame] = []
    top_n = spec.plots.elite_showcase.top_n_per_source
    selectors = spec.plots.elite_showcase.source_selectors

    for source_id in processed_source_ids:
        source_frame = frame[frame["source_id"] == source_id].copy()
        source_frame = source_frame.sort_values(["elite_rank", "elite_id"]).reset_index(drop=True)
        if source_frame.empty:
            continue

        selector = selectors.get(source_id)
        if selector is None:
            chosen = source_frame.copy() if top_n is None else source_frame.head(int(top_n)).copy()
        elif selector.elite_ids is not None:
            by_id = source_frame.set_index("elite_id", drop=False)
            missing_ids = [elite_id for elite_id in selector.elite_ids if elite_id not in by_id.index]
            if missing_ids:
                raise ValueError(
                    "Portfolio elite showcase selector requested unknown elite ids: "
                    f"source_id={source_id!r} missing_ids={missing_ids}"
                )
            chosen = pd.concat([by_id.loc[[elite_id]] for elite_id in selector.elite_ids], ignore_index=True)
        else:
            by_rank = source_frame.set_index("elite_rank", drop=False)
            assert selector.elite_ranks is not None
            missing_ranks = [rank for rank in selector.elite_ranks if rank not in by_rank.index]
            if missing_ranks:
                raise ValueError(
                    "Portfolio elite showcase selector requested unknown elite ranks: "
                    f"source_id={source_id!r} missing_ranks={missing_ranks}"
                )
            chosen = pd.concat([by_rank.loc[[rank]] for rank in selector.elite_ranks], ignore_index=True)

        selected_parts.append(chosen)

    if not selected_parts:
        raise ValueError("Portfolio elite showcase selected zero elites.")

    selected = pd.concat(selected_parts, ignore_index=True)
    source_order = {source_id: idx for idx, source_id in enumerate(source_ids)}
    selected["source_order"] = selected["source_id"].map(source_order)
    selected = selected.drop_duplicates(subset=["source_id", "elite_id"], keep="first")
    selected = selected.sort_values(["source_order", "elite_rank", "elite_id"]).reset_index(drop=True)
    return selected


def _write_elite_showcase_plot(
    *,
    spec: PortfolioSpec,
    elite_summary_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    out_path: Path,
) -> Path | None:
    if not spec.plots.elite_showcase.enabled:
        return None

    selected_elites = _select_portfolio_showcase_elites(spec, elite_summary_df)
    selected_keys = selected_elites[["source_id", "elite_id"]].drop_duplicates()
    selected_windows = selected_keys.merge(
        handoff_df,
        how="left",
        on=["source_id", "elite_id"],
        validate="one_to_many",
    )
    missing_windows = selected_windows[selected_windows["tf"].isna()][["source_id", "elite_id"]].drop_duplicates()
    if not missing_windows.empty:
        labels = [f"{row.source_id}:{row.elite_id}" for row in missing_windows.itertuples(index=False)]
        raise ValueError("Portfolio elite showcase missing handoff windows for selected elites: " + ", ".join(labels))
    _ensure_required_columns(
        selected_elites,
        ["source_id", "run_dir"],
        context="Portfolio elite showcase selected elites",
    )
    pwms_by_source: dict[str, dict[str, object]] = {}
    for source_id, source_rows in selected_elites.groupby("source_id", sort=False):
        run_dirs = source_rows["run_dir"].astype(str).dropna().unique().tolist()
        if len(run_dirs) != 1:
            raise ValueError(
                "Portfolio elite showcase requires exactly one run_dir per source: "
                f"source_id={source_id!r} run_dirs={run_dirs}"
            )
        source_pwms, _ = load_pwms_from_config(Path(run_dirs[0]))
        pwms_by_source[str(source_id)] = source_pwms
    _plot_portfolio_elite_showcase(
        selected_elites_df=selected_elites,
        handoff_df=selected_windows,
        pwms_by_source=pwms_by_source,
        out_path=out_path,
        ncols=int(spec.plots.elite_showcase.ncols),
        dpi=int(spec.plots.elite_showcase.dpi),
    )
    return out_path


def _write_plot_manifests(
    portfolio_run_dir: Path,
    *,
    table_entries: list[dict[str, object]],
    plot_entries: list[dict[str, object]],
) -> None:
    manifests_dir = portfolio_manifests_dir(portfolio_run_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    table_manifest_file = manifests_dir / "table_manifest.json"
    plot_manifest_file = manifests_dir / "plot_manifest.json"
    atomic_write_json(table_manifest_file, {"tables": table_entries}, allow_nan=False)
    atomic_write_json(plot_manifest_file, {"plots": plot_entries}, allow_nan=False)
    atomic_write_json(
        manifests_dir / "manifest.json",
        {
            "table_manifest": table_manifest_file.name,
            "plot_manifest": plot_manifest_file.name,
        },
        allow_nan=False,
    )


def _materialize_portfolio_outputs(
    *,
    run_dir: Path,
    spec: PortfolioSpec,
    all_window_rows: list[dict[str, object]],
    all_elite_rows: list[dict[str, object]],
    source_summary_rows: list[dict[str, object]],
    study_summary_rows: list[dict[str, object]],
    sequence_length_rows: list[dict[str, object]],
) -> tuple[list[Path], list[Path], pd.DataFrame]:
    handoff_df = pd.DataFrame(all_window_rows)
    if handoff_df.empty:
        raise ValueError("Portfolio produced zero selected elites across all sources.")
    handoff_df = handoff_df.sort_values(["source_id", "elite_rank", "tf", "best_start"]).reset_index(drop=True)

    elite_summary_df = pd.DataFrame(all_elite_rows)
    if elite_summary_df.empty:
        raise ValueError("Portfolio produced zero elite summary rows across all sources.")
    elite_summary_df = elite_summary_df.sort_values(["source_id", "elite_rank"]).reset_index(drop=True)

    source_summary_df = pd.DataFrame(source_summary_rows)
    source_summary_df = source_summary_df.sort_values(["source_id"]).reset_index(drop=True)

    table_paths: list[Path] = []
    main_format = str(spec.artifacts.table_format)
    handoff_main = portfolio_table_path(run_dir, "handoff_windows_long", main_format)
    elite_summary_main = portfolio_table_path(run_dir, "handoff_elites_summary", main_format)
    summary_main = portfolio_table_path(run_dir, "source_summary", main_format)
    _write_table(handoff_df, handoff_main)
    _write_table(elite_summary_df, elite_summary_main)
    _write_table(source_summary_df, summary_main)
    table_paths.extend([handoff_main, elite_summary_main, summary_main])

    if study_summary_rows:
        study_summary_df = (
            pd.DataFrame(study_summary_rows).sort_values(["source_id", "study_name"]).reset_index(drop=True)
        )
        study_main = portfolio_table_path(run_dir, "study_summary", main_format)
        _write_table(study_summary_df, study_main)
        table_paths.append(study_main)

    if sequence_length_rows:
        sequence_length_df = (
            pd.DataFrame(sequence_length_rows)
            .sort_values(["source_id", "sequence_length", "trial_id"])
            .reset_index(drop=True)
        )
        sequence_length_main = portfolio_table_path(run_dir, "handoff_sequence_length", main_format)
        _write_table(sequence_length_df, sequence_length_main)
        table_paths.append(sequence_length_main)

    if bool(spec.artifacts.write_csv) and main_format != "csv":
        handoff_csv = portfolio_table_path(run_dir, "handoff_windows_long", "csv")
        elite_summary_csv = portfolio_table_path(run_dir, "handoff_elites_summary", "csv")
        summary_csv = portfolio_table_path(run_dir, "source_summary", "csv")
        _write_table(handoff_df, handoff_csv)
        _write_table(elite_summary_df, elite_summary_csv)
        _write_table(source_summary_df, summary_csv)
        table_paths.extend([handoff_csv, elite_summary_csv, summary_csv])
        if study_summary_rows:
            study_summary_df = (
                pd.DataFrame(study_summary_rows).sort_values(["source_id", "study_name"]).reset_index(drop=True)
            )
            study_csv = portfolio_table_path(run_dir, "study_summary", "csv")
            _write_table(study_summary_df, study_csv)
            table_paths.append(study_csv)
        if sequence_length_rows:
            sequence_length_df = (
                pd.DataFrame(sequence_length_rows)
                .sort_values(["source_id", "sequence_length", "trial_id"])
                .reset_index(drop=True)
            )
            sequence_length_csv = portfolio_table_path(run_dir, "handoff_sequence_length", "csv")
            _write_table(sequence_length_df, sequence_length_csv)
            table_paths.append(sequence_length_csv)

    plot_paths: list[Path] = []
    tradeoff_plot = _write_tradeoff_plot(
        source_summary_df,
        portfolio_plot_path(run_dir, "source_tradeoff_score_vs_diversity", "pdf"),
    )
    if tradeoff_plot is not None:
        plot_paths.append(tradeoff_plot)
    showcase_plot = _write_elite_showcase_plot(
        spec=spec,
        elite_summary_df=elite_summary_df,
        handoff_df=handoff_df,
        out_path=portfolio_plot_path(run_dir, "elite_showcase_cross_workspace", spec.plots.elite_showcase.plot_format),
    )
    if showcase_plot is not None:
        plot_paths.append(showcase_plot)

    table_entries = [
        {
            "key": path.stem.removeprefix("table__"),
            "path": str(path.relative_to(portfolio_tables_dir(run_dir))),
            "format": path.suffix.lstrip("."),
        }
        for path in table_paths
    ]
    plot_entries = [
        {
            "key": path.stem.removeprefix("plot__"),
            "path": str(path.relative_to(portfolio_plots_dir(run_dir))),
            "format": path.suffix.lstrip("."),
        }
        for path in plot_paths
    ]
    _write_plot_manifests(run_dir, table_entries=table_entries, plot_entries=plot_entries)
    return table_paths, plot_paths, elite_summary_df
