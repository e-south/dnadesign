"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/objective_meta.py

List objective metadata/diagnostics present for a campaign/round.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import typer
from pyarrow import dataset as ds

from ...utils import ExitCodes, print_stdout
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    store_from_cfg,
)


@cli_command(
    "objective-meta",
    help="List objective metadata and diagnostic keys for a campaign/round.",
)
def cmd_objective_meta(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="campaign.yaml or directory", envvar="OPAL_CONFIG"
    ),
    round: Optional[str] = typer.Option(
        None,
        "--round",
        "-r",
        help="Round selector: integer or 'latest' (default: latest)",
    ),
    json: bool = typer.Option(False, "--json/--human", help="Output as JSON"),
    profile: bool = typer.Option(
        True,
        "--profile/--no-profile",
        help="Profile candidate hue/size fields (dtype, coverage, range, suitability).",
    ),
) -> None:
    try:
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        base = Path(cfg.campaign.workdir) / "outputs"
        runs_file = base / "ledger.runs.parquet"  # current sink
        runs_dir = base / "ledger.runs"  # legacy sink (directory)
        pred_dir = base / "ledger.predictions"
        if not pred_dir.exists():
            raise typer.Exit(code=1)

        # Load runs metadata (file preferred; directory fallback)
        if runs_file.exists():
            rtab = pd.read_parquet(
                runs_file,
                columns=[
                    "run_id",
                    "as_of_round",
                    "objective__name",
                    "objective__params",
                    "objective__summary_stats",
                ],
            )
        elif runs_dir.exists():
            rd = ds.dataset(str(runs_dir))
            rtab = rd.to_table(
                columns=[
                    "run_id",
                    "as_of_round",
                    "objective__name",
                    "objective__params",
                    "objective__summary_stats",
                ]
            ).to_pandas()
        else:
            raise typer.Exit(code=1)
        if rtab.empty:
            raise typer.Exit(code=1)

        # Pick round (supports 'latest' or an integer)
        rsel = None if round is None else str(round).strip().lower()
        if rsel in (None, "", "latest"):
            as_of = int(rtab["as_of_round"].max())
        else:
            try:
                as_of = int(rsel)
            except ValueError as e:
                raise typer.BadParameter("Invalid --round: must be an integer or 'latest'") from e
        rsel = rtab[rtab["as_of_round"] == as_of]
        if rsel.empty:
            raise typer.Exit(code=1)
        # Use last (most recent) run_id at that round
        rsel = rsel.sort_values(["run_id"]).tail(1).iloc[0]

        # Row-level diagnostic schema from predictions
        pdset = ds.dataset(str(pred_dir))
        pred_schema = [f.name for f in pdset.schema]
        obj_diag_cols = sorted([c for c in pred_schema if c.startswith("obj__")])

        out: Dict[str, Any] = {
            "round": as_of,
            "objective": {
                "name": rsel["objective__name"],
                "params_keys": sorted(list((rsel["objective__params"] or {}).keys())),
                "summary_stats_keys": (
                    sorted(list((rsel["objective__summary_stats"] or {}).keys()))
                    if isinstance(rsel["objective__summary_stats"], dict)
                    else []
                ),
            },
            "row_level_diagnostics_columns": obj_diag_cols,
        }

        # ---- Optional profiling of columns for hue/size candicates ----
        if profile:
            run_id = str(rsel["run_id"])
            # Candidate columns: objective row-level diagnostics plus commonly useful fields
            extras = ["pred__y_obj_scalar", "sel__rank_competition", "sel__is_selected"]
            cand_cols = sorted(list(set(obj_diag_cols + [c for c in extras if c in pred_schema])))
            need = ["id"] + cand_cols + ["as_of_round", "run_id"]

            # Read only the selected run rows
            filt = (pc.field("run_id") == run_id) & (pc.field("as_of_round") == as_of)
            dfp = pdset.to_table(columns=need, filter=filt).to_pandas()
            if dfp.empty:
                raise RuntimeError("No prediction rows found for selected run_id/round.")

            # Universe size (for coverage vs. dataset)
            df_records = store.load()
            total_records = int(len(df_records))
            unique_ids_records = int(df_records["id"].astype(str).nunique())

            # Helper: profile a single numeric column
            def _profile_numeric(col: str) -> Dict[str, Any]:
                s = pd.to_numeric(dfp[col], errors="coerce")
                present_ids = dfp.loc[s.notna(), "id"].astype(str).nunique()
                pred_ids = dfp["id"].astype(str).nunique()
                cov_pred = (present_ids / pred_ids) if pred_ids > 0 else 0.0
                cov_vs_records = (present_ids / unique_ids_records) if unique_ids_records > 0 else 0.0
                finite = s[np.isfinite(s)]
                stats = {
                    "count": int(s.notna().sum()),
                    "finite_count": int(finite.size),
                    "non_finite_count": int(s.size - finite.size - s.isna().sum()),
                    "unique": int(pd.Series(finite).nunique()),
                    "min": (float(np.min(finite)) if finite.size else None),
                    "median": (float(np.median(finite)) if finite.size else None),
                    "max": (float(np.max(finite)) if finite.size else None),
                    "coverage_in_predictions": float(cov_pred),
                    "coverage_vs_records": float(cov_vs_records),
                    "bounded_0_1": bool(
                        finite.size > 0 and np.nanmin(finite) >= -1e-12 and np.nanmax(finite) <= 1.0 + 1e-12
                    ),
                    "non_negative": bool(finite.size > 0 and np.nanmin(finite) >= 0.0 - 1e-12),
                }
                # Suitability (assertive rules; easy to tweak)
                variable = stats["unique"] > 1
                hue_ok = variable and stats["finite_count"] > 0 and stats["coverage_in_predictions"] >= 0.8
                size_ok = variable and stats["finite_count"] > 0 and stats["coverage_in_predictions"] >= 0.8
                stats["hue_ok"] = bool(hue_ok)
                stats["size_ok"] = bool(size_ok)
                return stats

            # Prepare per-column profiles with dtype judgement
            prof_rows: List[Dict[str, Any]] = []
            hue_recs: List[Tuple[str, float]] = []
            size_recs: List[Tuple[str, float]] = []
            for c in cand_cols:
                dtype = str(dfp[c].dtype)
                is_bool = dtype.startswith("bool") or (set(pd.unique(dfp[c].dropna())) <= {True, False})
                is_num = pd.api.types.is_numeric_dtype(dfp[c]) or pd.api.types.is_bool_dtype(dfp[c])
                row: Dict[str, Any] = {"column": c, "dtype": dtype}
                if is_num:
                    pr = _profile_numeric(c)
                    row.update(pr)
                    # Rank recommendations by variance proxy (range) and coverage
                    rng = (pr["max"] - pr["min"]) if (pr["max"] is not None and pr["min"] is not None) else 0.0
                    score = float(rng) * pr["coverage_in_predictions"]
                    if pr.get("hue_ok") and not is_bool:
                        hue_recs.append((c, score))
                    if pr.get("size_ok"):
                        size_recs.append((c, score))
                else:
                    # Non-numeric: never suitable for numeric hue/size
                    row.update(
                        {
                            "count": int(dfp[c].notna().sum()),
                            "finite_count": 0,
                            "non_finite_count": 0,
                            "unique": int(dfp[c].dropna().nunique()),
                            "min": None,
                            "median": None,
                            "max": None,
                            "coverage_in_predictions": float(
                                dfp.loc[dfp[c].notna(), "id"].astype(str).nunique()
                                / max(1, dfp["id"].astype(str).nunique())
                            ),
                            "coverage_vs_records": float(
                                dfp.loc[dfp[c].notna(), "id"].astype(str).nunique() / max(1, unique_ids_records)
                            ),
                            "bounded_0_1": False,
                            "non_negative": False,
                            "hue_ok": False,
                            "size_ok": False,
                        }
                    )
                prof_rows.append(row)
            # Rank recommendations (best first)
            hue_recs.sort(key=lambda t: t[1], reverse=True)
            size_recs.sort(key=lambda t: t[1], reverse=True)
            out["profile"] = {
                "records_total": total_records,
                "records_unique_ids": unique_ids_records,
                "columns": prof_rows,
                "recommendations": {
                    "hue": [name for name, _ in hue_recs],
                    "size": [name for name, _ in size_recs],
                },
            }

        if json:
            json_out(out)
        else:
            print_stdout(f"[bold cyan]Round:[/] {out['round']}")
            print_stdout(f"[bold cyan]Objective:[/] {out['objective']['name']}")
            print_stdout("[bold]  params keys:[/] " + (", ".join(out["objective"]["params_keys"]) or "(none)"))
            print_stdout("[bold]  summary stats:[/] " + (", ".join(out["objective"]["summary_stats_keys"]) or "(none)"))
            print_stdout("[bold cyan]Row-level diagnostics:[/]")
            for c in out["row_level_diagnostics_columns"]:
                print_stdout(f"  • {c}")

            if profile:
                p = out.get("profile", {}) or {}
                cols = p.get("columns", [])
                rec_hue = p.get("recommendations", {}).get("hue", [])
                rec_size = p.get("recommendations", {}).get("size", [])
                if rec_hue:
                    print_stdout("\n[bold cyan]Recommended hue fields (best-first):[/]")
                    for n in rec_hue[:8]:
                        print_stdout(f"  • {n}")
                else:
                    print_stdout("\n[bold cyan]Recommended hue fields:[/] (none)")
                if rec_size:
                    print_stdout("\n[bold cyan]Recommended size fields (best-first):[/]")
                    for n in rec_size[:8]:
                        print_stdout(f"  • {n}")
                else:
                    print_stdout("\n[bold cyan]Recommended size fields:[/] (none)")

                # Compact column table (subset of stats)
                if cols:
                    print_stdout("\n[bold cyan]Column profiles (subset):[/]")
                    for r in cols:
                        name = r["column"]
                        dtype = r["dtype"]
                        covp = r["coverage_in_predictions"]
                        covd = r["coverage_vs_records"]
                        mn, md, mx = r["min"], r["median"], r["max"]
                        flags = []
                        if r.get("bounded_0_1"):
                            flags.append("0..1")
                        if r.get("non_negative") and not r.get("bounded_0_1"):
                            flags.append("≥0")
                        if r.get("hue_ok"):
                            flags.append("hue✓")
                        if r.get("size_ok"):
                            flags.append("size✓")
                        flag_str = (" [" + ", ".join(flags) + "]") if flags else ""
                        rng_str = (
                            "range: n/a" if mn is None or mx is None else f"range: {mn:.4g}..{mx:.4g}; median: {md:.4g}"
                        )
                        print_stdout(
                            f"  • {name:24s} dtype={dtype:8s} cov_pred={covp:6.2%} cov_ds={covd:6.2%}  {rng_str}{flag_str}"  # noqa
                        )

    except typer.Exit:
        raise
    except Exception as e:
        # Preserve pattern of other commands
        internal_error("objective-meta", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
