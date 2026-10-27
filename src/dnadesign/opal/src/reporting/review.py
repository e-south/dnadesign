"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/review.py

Builds campaign-scoped review artifacts from OPAL ledgers and per-round outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import polars as pl

from ..analysis.campaign import CampaignAnalysis
from ..analysis.ledger import read_selection_view_predictions
from ..core.leakage import assert_no_leakage_violations, build_prediction_identity_report
from ..core.rounds import resolve_round_index_from_runs
from ..core.utils import ExitCodes, OpalError, read_json, write_json
from ..plots._mpl_utils import pretty_label
from ..storage.ledger import LedgerReader
from ..storage.x_contracts import validate_x_parquet_column
from .review_plots import write_feature_importance_plot, write_score_vs_rank_plot
from .summary import load_round_log, select_run_meta, summarize_round_log, summarize_run_meta

REVIEW_SCHEMA_VERSION = "opal.campaign_review.v1"
_PREDICTION_REVIEW_COLUMNS = (
    "id",
    "run_id",
    "as_of_round",
    "view__selection_score",
    "view__rank_competition",
    "view__is_selected",
)


@dataclass(frozen=True)
class CampaignReviewResult:
    config_path: Path
    workdir: Path
    out_dir: Path
    manifest_path: Path
    review_path: Path
    index_path: Path
    plot_paths: tuple[Path, ...]
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "workdir": str(self.workdir),
            "out_dir": str(self.out_dir),
            "manifest_path": str(self.manifest_path),
            "review_path": str(self.review_path),
            "index_path": str(self.index_path),
            "plot_paths": [str(path) for path in self.plot_paths],
            "manifest": self.manifest,
        }


def build_campaign_review(
    config_path: Path | None,
    *,
    round_selector: str | None = "latest",
    run_id: str | None = None,
    selection_view_id: str | None = None,
    out_dir: Path | None = None,
    include_plots: bool = True,
) -> CampaignReviewResult:
    analysis = CampaignAnalysis.from_config_path(config_path, allow_dir=True)
    cfg = analysis.config
    configured_view_ids = [view.id for view in cfg.selection_views]
    if selection_view_id is None:
        if len(configured_view_ids) != 1:
            raise OpalError(
                f"selection_view_id is required for this multi-view campaign. Available: {configured_view_ids}",
                ExitCodes.BAD_ARGS,
            )
        selection_view_id = configured_view_ids[0]
    if selection_view_id not in configured_view_ids:
        raise OpalError(
            f"Unknown selection view {selection_view_id!r}. Available: {configured_view_ids}",
            ExitCodes.BAD_ARGS,
        )
    ws = analysis.workspace
    reader = LedgerReader(ws)
    x_contract = validate_x_parquet_column(
        analysis.records_store().records_path,
        x_column=cfg.data.x_column_name,
    )
    runs_df = reader.read_runs()
    round_index = resolve_round_index_from_runs(runs_df, round_selector)
    run_meta_row = select_run_meta(runs_df, round_sel=round_index, run_id=run_id)
    run_summary = summarize_run_meta(run_meta_row)
    resolved_run_id = str(run_summary["run_id"])
    resolved_round = int(run_summary["as_of_round"])

    round_log_path = ws.round_logs_dir(resolved_round) / "round.log.jsonl"
    round_log_summary = summarize_round_log(load_round_log(round_log_path), run_id=resolved_run_id)
    round_log_summary["round_index"] = resolved_round
    round_log_summary["path"] = str(round_log_path)

    predictions = _read_review_predictions(
        reader,
        selection_view_id=selection_view_id,
        round_index=resolved_round,
        run_id=resolved_run_id,
    )
    selection_summary, selection_preview = _selection_summary(predictions)

    review_dir = out_dir if out_dir is not None else ws.outputs_dir / "review" / "selection_views" / selection_view_id
    review_dir = Path(review_dir).resolve()
    plots_dir = review_dir / "plots"
    manifest_path = review_dir / "manifest.json"
    review_path = review_dir / "review.md"
    index_path = review_dir / "index.html"
    plot_statuses: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    if include_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        score_plot = plots_dir / f"score_vs_rank__round_{resolved_round}__run_{resolved_run_id}.png"
        write_score_vs_rank_plot(
            predictions,
            _selected_mask(predictions["view__is_selected"]),
            score_plot,
            campaign_name=cfg.campaign.name,
            selection_view_id=selection_view_id,
            round_index=resolved_round,
        )
        plot_statuses.append(
            {
                "name": "score_vs_rank",
                "status": "written",
                "scope": "selection_view",
                "path": str(score_plot),
            }
        )
        plot_paths.append(score_plot)
        fi_status = write_feature_importance_plot(
            ws.round_model_dir(resolved_round) / "feature_importance.csv",
            plots_dir / f"feature_importance_top__round_{resolved_round}__run_{resolved_run_id}.png",
            round_index=resolved_round,
        )
        plot_statuses.append(fi_status)
        if fi_status["status"] == "written":
            plot_paths.append(Path(str(fi_status["path"])))

    referenced_plot_paths = [Path(str(row["path"])) for row in plot_statuses if row.get("path")]
    stale_artifacts = detect_review_stale_artifacts(review_dir, referenced_paths=referenced_plot_paths)
    warnings = [
        {
            "category": "StaleArtifactWarning",
            "severity": "warning",
            "message": f"Review artifact exists on disk but is absent from the active manifest: {row['path']}",
            "path": row["path"],
        }
        for row in stale_artifacts
    ]

    manifest = _jsonable(
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "campaign": {
                "name": cfg.campaign.name,
                "slug": cfg.campaign.slug,
                "workdir": str(ws.workdir),
                "config_path": str(analysis.config_path),
                "x_column": cfg.data.x_column_name,
                "x_contract": {
                    "schema_version": "opal.x_matrix_contract.v1",
                    "physical_type": "fixed_size_list",
                    "x_dim": int(x_contract.x_dim),
                    "row_count": int(x_contract.row_count),
                    "canonical": True,
                },
                "y_column": cfg.data.y_column_name,
                "model": cfg.model.name,
                "selection_views": [
                    {
                        "id": view.id,
                        "objective": view.objective.name,
                        "selection": view.selection.name,
                    }
                    for view in cfg.selection_views
                ],
            },
            "review_scope": {
                "selection_view_id": selection_view_id,
                "round_selector": round_selector,
                "round_index": resolved_round,
                "run_id": resolved_run_id,
            },
            "run": run_summary,
            "progress": round_log_summary,
            "selection": selection_summary,
            "selection_preview": selection_preview,
            "plots": plot_statuses,
            "stale_artifacts": stale_artifacts,
            "warnings": warnings,
            "artifacts": {
                "manifest": str(manifest_path),
                "review_markdown": str(review_path),
                "review_html": str(index_path),
                "round_log": str(round_log_path),
                "ledger_runs": str(ws.ledger_runs_path),
                "ledger_predictions": str(ws.ledger_predictions_dir),
            },
        }
    )

    review_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    review_path.write_text(render_campaign_review_markdown(manifest), encoding="utf-8")
    index_path.write_text(render_campaign_review_html(manifest, base_dir=review_dir), encoding="utf-8")
    return CampaignReviewResult(
        config_path=analysis.config_path,
        workdir=ws.workdir,
        out_dir=review_dir,
        manifest_path=manifest_path,
        review_path=review_path,
        index_path=index_path,
        plot_paths=tuple(plot_paths),
        manifest=manifest,
    )


def load_review_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise OpalError(f"Review manifest is not a JSON object: {manifest_path}")
    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise OpalError(f"Unsupported review manifest schema at {manifest_path}: {payload.get('schema_version')!r}")
    payload.setdefault("stale_artifacts", [])
    payload.setdefault("warnings", [])
    return payload


def detect_review_stale_artifacts(
    review_dir: str | Path,
    *,
    referenced_paths: list[Path] | tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    review_path = Path(review_dir)
    plots_path = review_path / "plots"
    if not plots_path.exists():
        return []
    referenced = {str(Path(path).resolve()) for path in referenced_paths}
    stale = []
    for path in sorted(plots_path.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".svg", ".pdf", ".csv", ".json"}:
            continue
        if str(path.resolve()) in referenced:
            continue
        stat = path.stat()
        stale.append(
            {
                "category": "StaleArtifactWarning",
                "severity": "warning",
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "reason": "file is not referenced by the active review manifest",
            }
        )
    return stale


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def render_campaign_review_markdown(manifest: Mapping[str, Any]) -> str:
    campaign = manifest.get("campaign") or {}
    run = manifest.get("run") or {}
    view = _review_view_contract(manifest)
    progress = manifest.get("progress") or {}
    selection = manifest.get("selection") or {}
    preview = manifest.get("selection_preview") or []
    plots = manifest.get("plots") or []
    warnings = manifest.get("warnings") or []
    lines = [
        "# OPAL campaign review",
        "",
        "## Campaign",
        "",
        f"- name: `{campaign.get('name')}`",
        f"- slug: `{campaign.get('slug')}`",
        f"- workdir: `{campaign.get('workdir')}`",
        f"- config: `{campaign.get('config_path')}`",
        f"- X column: `{campaign.get('x_column')}`",
        f"- model: `{campaign.get('model')}`",
        f"- selection view: `{view['label']}` (`{view['id']}`)",
        f"- objective: `{view['objective']}`",
        f"- selector: `{view['selection']}`",
        "",
        "## Run",
        "",
        f"- run_id: `{run.get('run_id')}`",
        f"- round: `{run.get('as_of_round')}`",
        f"- train rows: `{run.get('stats_n_train')}`",
        f"- scored rows: `{run.get('stats_n_scored')}`",
        f"- score channel: `{view['score_ref']}`",
        f"- requested top-k: `{view['top_k']}`",
        "",
        "## Progress",
        "",
        f"- events: `{progress.get('events')}`",
        f"- duration_sec_total: `{progress.get('duration_sec_total')}`",
        f"- predict_batches: `{progress.get('predict_batches')}`",
        f"- predict_rows: `{progress.get('predict_rows')}`",
        f"- round_log: `{progress.get('path')}`",
        "",
        "## Selection",
        "",
        f"- selected rows: `{selection.get('selected_count')}`",
        f"- score min: `{selection.get('score_min')}`",
        f"- score median: `{selection.get('score_median')}`",
        f"- score max: `{selection.get('score_max')}`",
        "",
        "### Selection Preview",
        "",
    ]
    if preview:
        lines.extend(["| rank | id | score | selected |", "|---:|---|---:|---|"])
        for row in preview:
            lines.append(
                "| {rank} | `{id}` | {score} | {selected} |".format(
                    rank=row.get("rank"),
                    id=row.get("id"),
                    score=row.get("score"),
                    selected=row.get("selected"),
                )
            )
    else:
        lines.append("No selected rows found in the selected ledger scope.")
    lines.extend(["", "## Plots", ""])
    if plots:
        lines.extend(f"- {plot.get('name')}: `{plot.get('status')}` {plot.get('path', '')}" for plot in plots)
    else:
        lines.append("No plots requested.")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning.get('category')}`: {warning.get('message')}")
    lines.append("")
    return "\n".join(lines)


def render_campaign_review_html(manifest: Mapping[str, Any], *, base_dir: Path) -> str:
    campaign = manifest.get("campaign") or {}
    run = manifest.get("run") or {}
    view = _review_view_contract(manifest)
    progress = manifest.get("progress") or {}
    selection = manifest.get("selection") or {}
    preview = manifest.get("selection_preview") or []
    plots = manifest.get("plots") or []
    warnings = manifest.get("warnings") or []
    warning_cards = [
        '<article class="warning">'
        f"<h3>{_e(warning.get('category'))}</h3>"
        f"<p>{_e(warning.get('message'))}</p>"
        f"<code>{_e(warning.get('path'))}</code>"
        "</article>"
        for warning in warnings
    ]
    plot_cards: list[str] = []
    for plot in plots:
        if plot.get("status") != "written" or not plot.get("path"):
            plot_cards.append(
                "<article>"
                f"<h3>{_e(plot.get('name'))}</h3>"
                f"<p>{_e(plot.get('status'))}: {_e(plot.get('reason'))}</p>"
                "</article>"
            )
            continue
        src = _rel(plot.get("path"), base_dir=base_dir)
        plot_cards.append(
            "<article>"
            f"<h3>{_e(plot.get('name'))}</h3>"
            f'<a href="{_e(src)}"><img src="{_e(src)}" alt="{_e(plot.get("name"))}"></a>'
            "</article>"
        )
    preview_rows = [
        "<tr>"
        f"<td>{_e(row.get('rank'))}</td>"
        f"<td><code>{_e(row.get('id'))}</code></td>"
        f"<td>{_e(row.get('score'))}</td>"
        f"<td>{_e(row.get('selected'))}</td>"
        "</tr>"
        for row in preview
    ]
    body = f"""
    <header>
      <p>OPAL campaign review · {_e(view["label"])} selection view</p>
      <h1>{_e(campaign.get("name"))}</h1>
    </header>
    <main>
      <section class="summary-grid">
        {_metric_card("Run", run.get("run_id"))}
        {_metric_card("Round", run.get("as_of_round"))}
        {_metric_card("Train", run.get("stats_n_train"))}
        {_metric_card("Scored", run.get("stats_n_scored"))}
        {_metric_card("Selected", selection.get("selected_count"))}
        {_metric_card("Duration sec", progress.get("duration_sec_total"))}
      </section>
      <section>
        <h2>Contract</h2>
        <dl>
          <dt>Config</dt><dd><code>{_e(campaign.get("config_path"))}</code></dd>
          <dt>Workdir</dt><dd><code>{_e(campaign.get("workdir"))}</code></dd>
          <dt>X column</dt><dd><code>{_e(campaign.get("x_column"))}</code></dd>
          <dt>Model</dt><dd>{_e(campaign.get("model"))}</dd>
          <dt>Selection view</dt><dd>{_e(view["label"])} (<code>{_e(view["id"])}</code>)</dd>
          <dt>Objective</dt><dd><code>{_e(view["objective"])}</code></dd>
          <dt>Selector</dt><dd><code>{_e(view["selection"])}</code></dd>
          <dt>Score channel</dt><dd><code>{_e(view["score_ref"])}</code></dd>
        </dl>
      </section>
      <section>
        <h2>Plots</h2>
        <div class="plot-grid">{"".join(plot_cards) if plot_cards else "<p>No plots written.</p>"}</div>
      </section>
      <section>
        <h2>Warnings</h2>
        <div class="plot-grid">{"".join(warning_cards) if warning_cards else "<p>No warnings.</p>"}</div>
      </section>
      <section>
        <h2>Selected Records</h2>
        <table>
          <thead><tr><th>Rank</th><th>ID</th><th>Score</th><th>Selected</th></tr></thead>
          <tbody>{"".join(preview_rows)}</tbody>
        </table>
      </section>
    </main>
    """
    return _html_document(title=f"OPAL review: {campaign.get('name')} · {view['label']}", body=body)


def _review_view_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scope = manifest.get("review_scope") or {}
    view_id = str(scope.get("selection_view_id") or "").strip()
    if not view_id:
        raise OpalError("Campaign review manifest is missing review_scope.selection_view_id.")

    campaign = manifest.get("campaign") or {}
    campaign_matches = [
        row
        for row in campaign.get("selection_views") or []
        if isinstance(row, Mapping) and str(row.get("id") or "") == view_id
    ]
    run = manifest.get("run") or {}
    run_matches = [
        row
        for row in run.get("selection_views") or []
        if isinstance(row, Mapping) and str(row.get("selection_view_id") or "") == view_id
    ]
    if len(campaign_matches) != 1 or len(run_matches) != 1:
        raise OpalError(
            f"Campaign review selection view {view_id!r} must resolve exactly once in campaign and run metadata."
        )
    campaign_view = campaign_matches[0]
    run_view = run_matches[0]
    return {
        "id": view_id,
        "label": pretty_label(view_id),
        "objective": campaign_view.get("objective"),
        "selection": campaign_view.get("selection"),
        "score_ref": run_view.get("score_ref"),
        "top_k": run_view.get("top_k"),
    }


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _rel(path: Any, *, base_dir: Path) -> str:
    return os.path.relpath(str(path), str(base_dir))


def _metric_card(label: str, value: Any) -> str:
    return f'<article class="metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></article>'


def _html_document(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2528;
      --muted: #667074;
      --line: #d8ddd7;
      --accent: #446a8c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header, main {{ width: min(1180px, calc(100vw - 40px)); margin: 0 auto; }}
    header {{ padding: 34px 0 16px; }}
    header p {{
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
      margin: 0 0 6px;
      text-transform: uppercase;
    }}
    h1 {{ font-size: clamp(1.8rem, 2.8vw, 3rem); margin: 0; }}
    h2 {{
      border-bottom: 1px solid var(--line);
      font-size: 1.18rem;
      margin: 30px 0 14px;
      padding-bottom: 8px;
    }}
    code {{ background: #eef1ef; border-radius: 4px; padding: 1px 5px; }}
    .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
    .metric, .plot-grid article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 4%);
    }}
    .metric {{ min-height: 86px; padding: 14px; }}
    .metric span {{ color: var(--muted); display: block; font-size: 0.78rem; text-transform: uppercase; }}
    .metric strong {{ display: block; font-size: 1.35rem; margin-top: 8px; overflow-wrap: anywhere; }}
    dl {{ display: grid; gap: 8px 16px; grid-template-columns: minmax(120px, max-content) 1fr; }}
    dt {{ color: var(--muted); font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .plot-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .plot-grid article {{ padding: 12px; }}
    .plot-grid h3 {{ font-size: 0.95rem; margin: 0 0 10px; }}
    img {{ display: block; height: auto; max-width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
    @media (max-width: 640px) {{
      header, main {{ width: min(100vw - 24px, 1180px); }}
      dl {{ grid-template-columns: 1fr; }}
      .plot-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _read_review_predictions(
    reader: LedgerReader,
    *,
    selection_view_id: str,
    round_index: int,
    run_id: str,
) -> pd.DataFrame:
    df = read_selection_view_predictions(
        reader.paths.predictions_dir,
        selection_view_id=selection_view_id,
        columns=_PREDICTION_REVIEW_COLUMNS,
        round_selector=int(round_index),
        run_id=str(run_id),
        runs_df=pl.from_pandas(reader.read_runs()),
        require_run_id=True,
    ).to_pandas()
    if df.empty:
        raise OpalError("No prediction rows found for campaign review scope.", ExitCodes.BAD_ARGS)
    assert_no_leakage_violations(
        build_prediction_identity_report(
            prediction_ids=df["id"],
            scope="campaign_review.predictions",
        )
    )
    return df


def _selected_mask(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise OpalError(
            "Campaign review predictions contain null view__is_selected values.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not pd.api.types.is_bool_dtype(values):
        bad = values.loc[~values.map(lambda value: isinstance(value, (bool, np.bool_)))]
        if not bad.empty:
            preview = ", ".join(repr(value) for value in bad.head(5).tolist())
            raise OpalError(
                f"Campaign review predictions view__is_selected must be boolean; got {preview}",
                ExitCodes.CONTRACT_VIOLATION,
            )
    return values.astype(bool)


def _selection_summary(predictions: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = predictions.copy()
    df["view__selection_score"] = pd.to_numeric(df["view__selection_score"], errors="raise")
    df["view__rank_competition"] = pd.to_numeric(df["view__rank_competition"], errors="raise").astype(int)
    selected = df[_selected_mask(df["view__is_selected"])].sort_values(["view__rank_competition", "id"]).copy()
    preview_rows = selected.head(25)
    preview = [
        {
            "rank": int(row["view__rank_competition"]),
            "id": str(row["id"]),
            "score": float(row["view__selection_score"]),
            "selected": bool(row["view__is_selected"]),
        }
        for _, row in preview_rows.iterrows()
    ]
    return (
        {
            "prediction_count": int(len(df)),
            "selected_count": int(len(selected)),
            "score_min": float(df["view__selection_score"].min()),
            "score_median": float(df["view__selection_score"].median()),
            "score_max": float(df["view__selection_score"].max()),
            "best_rank": int(df["view__rank_competition"].min()),
            "worst_rank": int(df["view__rank_competition"].max()),
        },
        preview,
    )
