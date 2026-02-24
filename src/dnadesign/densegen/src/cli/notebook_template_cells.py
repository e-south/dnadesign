"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_template_cells.py

Notebook cell template fragments used by DenseGen marimo notebook scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def records_export_cell_template() -> str:
    return """
@app.cell
def _(Path, df_window_filtered, export_button, export_format, export_path, mo, require, run_root):
    click_count = int(export_button.value or 0)
    status_text = ""
    if click_count > 0:
        selected_format = str(export_format.value or "").strip()
        require(
            selected_format not in {"parquet", "csv"},
            f"Export format must be parquet or csv, got `{selected_format}`.",
        )
        raw_path = str(export_path.value or "").strip()
        if not raw_path:
            raw_path = "outputs/notebooks/records_preview"

        destination = Path(raw_path).expanduser()
        if not destination.is_absolute():
            destination = run_root / destination

        if selected_format == "csv":
            if destination.suffix.lower() != ".csv":
                destination = destination.with_suffix(".csv")
        else:
            if destination.suffix.lower() != ".parquet":
                destination = destination.with_suffix(".parquet")

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if selected_format == "csv":
                df_window_filtered.to_csv(destination, index=False)
            else:
                df_window_filtered.to_parquet(destination, index=False)
        except Exception as exc:
            raise RuntimeError(f"Export failed while writing `{destination}`: {exc}") from exc

        status_text = f"Saved to `{destination}`."
    mo.md(status_text)
    return
"""


def build_run_summary_tables(*, run_manifest, run_items, pd):
    run_rows = []
    plan_quota_rows = []
    if isinstance(run_manifest, dict) and run_manifest:
        total_generated = int(run_manifest.get("total_generated", 0) or 0)
        total_quota = int(run_manifest.get("total_quota", 0) or 0)
        quota_progress = float(run_manifest.get("quota_progress_pct", 0.0) or 0.0)
        run_rows.extend(
            [
                {"Field": "Run id (manifest)", "Value": str(run_manifest.get("run_id", "-"))},
                {"Field": "Quota status", "Value": f"{total_generated}/{total_quota} ({quota_progress:.2f}%)"},
                {"Field": "Solver backend", "Value": str(run_manifest.get("solver_backend", "-"))},
                {"Field": "Solver strategy", "Value": str(run_manifest.get("solver_strategy", "-"))},
            ]
        )
    if not run_items.empty:
        if "generated" in run_items.columns and "quota" in run_items.columns:
            generated_series = run_items["generated"].fillna(0).astype(int)
            quota_series = run_items["quota"].fillna(0).astype(int)
            at_quota = int((generated_series >= quota_series).sum())
            run_rows.append({"Field": "Plans at quota", "Value": f"{at_quota}/{len(run_items)}"})
        if {"plan", "generated", "quota"}.issubset(set(run_items.columns)):
            _plan_frame = run_items[["plan", "generated", "quota"]].copy()
            _plan_frame["plan"] = _plan_frame["plan"].astype(str)
            _plan_frame["generated"] = _plan_frame["generated"].fillna(0).astype(int)
            _plan_frame["quota"] = _plan_frame["quota"].fillna(0).astype(int)
            _plan_totals = _plan_frame.groupby("plan", as_index=False, sort=True)[["generated", "quota"]].sum()
            for _row in _plan_totals.itertuples(index=False):
                _plan_name = str(getattr(_row, "plan", "") or "unscoped")
                _generated = int(getattr(_row, "generated", 0) or 0)
                _quota = int(getattr(_row, "quota", 0) or 0)
                _progress = (100.0 * _generated / _quota) if _quota > 0 else 0.0
                plan_quota_rows.append(
                    {
                        "Plan": _plan_name,
                        "Generated": _generated,
                        "Quota": _quota,
                        "Progress": f"{_generated}/{_quota} ({_progress:.2f}%)",
                    }
                )
    if not run_rows:
        run_rows = [{"Field": "Run summary", "Value": "Run manifest not available yet."}]
    return pd.DataFrame(plan_quota_rows), pd.DataFrame(run_rows)
