"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/reporter.py

Reporter - writes JSONL outputs (single source of truth), MANIFEST, and plots.
CSV export is optional and off by default.

All visualisations live in dedicated modules under dnadesign.permuter.plots.

Each module must expose a function:

      plot(elite_df: pd.DataFrame,
           all_df:   pd.DataFrame,
           output_path: Path,
           job_name: str,
           [ref_sequence: Optional[str]],
           [metric_id: Optional[str]] ) -> None

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .logging_utils import init_logger

_LOG = init_logger()


def _write_jsonl(path: Path, records: List[Dict], *, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _as_dataframe(records: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "MANIFEST.json"


def _load_manifest(output_dir: Path) -> Dict:
    mpath = _manifest_path(output_dir)
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            _LOG.warning(f"Malformed MANIFEST at {mpath}; recreating.")
    return {
        "rounds": [],
        "plots": [],
        "version": "0.4.4",
    }


def _save_manifest(output_dir: Path, manifest: Dict) -> None:
    mpath = _manifest_path(output_dir)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_manifest_round(
    output_dir: Path,
    job_name: str,
    ref_name: str,
    round_idx: int,
    *,
    ref_sequence: Optional[str] = None,
) -> None:
    manifest = _load_manifest(output_dir)
    manifest.setdefault("job", job_name)
    manifest.setdefault("ref", ref_name)

    if ref_sequence and "ref_sequence" not in manifest:
        manifest["ref_sequence"] = ref_sequence
        manifest["ref_length"] = len(ref_sequence)

    if round_idx and not any(r.get("round") == round_idx for r in manifest["rounds"]):
        manifest["rounds"].append(
            {
                "round": round_idx,
                "variants": f"r{round_idx}_variants.jsonl",
                "elites": f"r{round_idx}_elites.jsonl",
                "norm_stats": f"norm_stats_r{round_idx}.json",
            }
        )
    _save_manifest(output_dir, manifest)


def _infer_metric_token(df: pd.DataFrame) -> str:
    try:
        if "metrics" in df.columns and not df["metrics"].isna().all():
            keys: Set[str] = set()
            for d in df["metrics"]:
                if isinstance(d, dict):
                    keys.update(list(d.keys()))
                    if len(keys) > 1:
                        return "objective"
            if len(keys) == 1:
                return next(iter(keys))
    except Exception:
        pass

    if "score_type" in df.columns and not df.empty:
        st = str(df["score_type"].iloc[0]).strip()
        if st:
            return st

    return "objective"


def write_round_files(
    *,
    variants: List[Dict],
    elites: List[Dict],
    output_dir: Path,
    job_name: str,
    ref_name: str,
    round_idx: int,
) -> None:
    v_path = output_dir / f"r{round_idx}_variants.jsonl"
    e_path = output_dir / f"r{round_idx}_elites.jsonl"
    _write_jsonl(v_path, variants, mode="w")
    _write_jsonl(e_path, elites, mode="w")
    _LOG.info(f"[{job_name}/{ref_name}] wrote: {v_path.name}, {e_path.name}")


def write_results(
    variants: List[Dict],
    elites: List[Dict],
    output_dir: Path,
    job_name: str,
    ref_name: str,
    plot_names: List[str] | None = None,
    *,
    write_csv: bool = False,
    ref_sequence: Optional[str] = None,
    metric_meta: Optional[List[Dict]] = None,  # NEW: to build evaluator subtitle
) -> None:
    """
    Persist optional CSVs and generate requested plots.
    If ref_sequence is provided, it will be passed to plots that accept it and
    also written into MANIFEST for future analysis-mode runs.
    """
    plot_names = plot_names or []
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_manifest_round(
        output_dir, job_name, ref_name, round_idx=0, ref_sequence=ref_sequence
    )

    df_all = _as_dataframe(variants)

    if write_csv:
        _write_csv(df_all, output_dir / f"{job_name}.csv")
        _write_csv(_as_dataframe(elites), output_dir / f"{job_name}_elites.csv")
        _LOG.info(
            f"[{job_name}/{ref_name}] CSV: wrote {job_name}.csv, {job_name}_elites.csv"
        )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_token = _infer_metric_token(df_all)
    safe_metric = metric_token.replace(" ", "_").lower()
    safe_ref = ref_name.replace(" ", "_")

    # compact evaluator subtitle like "placeholder" or "evaluatorA + evaluatorB"
    eval_names = []
    if metric_meta:
        try:
            eval_names = sorted(
                {str(m.get("evaluator", "")).strip() for m in metric_meta if m}
            )
        except Exception:
            eval_names = []
    subtitle = " + ".join([e for e in eval_names if e]) if eval_names else ""

    generated_plots: List[str] = []
    for name in plot_names:
        try:
            mod = importlib.import_module(f"dnadesign.permuter.plots.{name}")
        except ModuleNotFoundError:
            _LOG.warning(f"[{job_name}/{ref_name}] plot '{name}' not found; skipping")
            continue

        if not hasattr(mod, "plot"):
            _LOG.warning(
                f"[{job_name}/{ref_name}] module '{name}' has no plot(); skipping"
            )
            continue

        out_path = plots_dir / f"{safe_metric}_{safe_ref}_{name}.png"

        try:
            sig = inspect.signature(mod.plot)
            kwargs = dict(
                elite_df=_as_dataframe(elites),
                all_df=df_all,
                output_path=out_path,
                job_name=job_name,
            )
            if "ref_sequence" in sig.parameters:
                kwargs["ref_sequence"] = ref_sequence
            if "metric_id" in sig.parameters:
                kwargs["metric_id"] = metric_token
            if "evaluators" in sig.parameters:
                kwargs["evaluators"] = subtitle

            mod.plot(**kwargs)

            _LOG.info(f"[{job_name}/{ref_name}] plot saved → {out_path.name}")
            generated_plots.append(str(out_path.relative_to(output_dir)))
        except Exception as exc:
            _LOG.error(f"[{job_name}/{ref_name}] plot '{name}' failed: {exc}")

    manifest = _load_manifest(output_dir)
    manifest.setdefault("job", job_name)
    manifest.setdefault("ref", ref_name)
    manifest["plots"] = sorted(set(manifest.get("plots", [])).union(generated_plots))
    _save_manifest(output_dir, manifest)
