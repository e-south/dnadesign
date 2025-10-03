"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/cli/app.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich.traceback import install as rich_traceback

from ..algo.leiden import run as leiden_run
from ..algo.sweep import leiden_sweep
from ..io.detect import detect_context
from ..io.read import extract_X, load_table
from ..io.write import attach_usr, write_generic
from ..presets.loader import load_all as load_presets
from ..runs.index import add_or_update_index, list_runs
from ..runs.reuse import find_equivalent_fit
from ..runs.signatures import (
    AlgoSignature,
    InputSignature,
    UmapSignature,
    file_fingerprint,
    ids_hash,
)
from ..runs.store import (
    create_run_dir,
    runs_root,
    umap_dir,
    write_labels,
    write_run_meta,
    write_summary,
    write_umap_coords,
    write_umap_meta,
)
from ..umap.compute import compute as umap_compute
from ..umap.plot import scatter as umap_scatter
from ..util.checks import (
    ClusterError,
    assert_id_sequence_bijection,
    assert_no_duplicate_ids,
)
from ..util.meta import compact_meta
from ..util.slug import auto_run_name, slugify
from ..util.warnings import configure as configure_warnings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Cluster CLI — fit, UMAP, analyses, run store, presets.",
)
console = Console()


@app.callback(invoke_without_command=False)
def _global_opts(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False, "--debug", help="Show full rich tracebacks with locals."
    ),
):
    """Global flags & rich traceback config."""
    rich_traceback(show_locals=debug)
    ctx.obj = {"debug": debug}
    configure_warnings(verbose=debug)


# ----------------------------- Helpers -----------------------------
def _apply_dedupe(df: pd.DataFrame, key_col: str, policy: str) -> pd.DataFrame:
    return assert_no_duplicate_ids(df, key_col=key_col, policy=policy)


def _context_and_df(
    dataset: Optional[str],
    file: Optional[str],
    usr_root: Optional[str],
    columns: list[str] | None = None,
):
    ctx = detect_context(dataset, file, usr_root)
    df = load_table(ctx, columns=columns)
    return ctx, df


def _rows_ids(df: pd.DataFrame, key_col: str) -> list[str]:
    return list(map(str, df[key_col].tolist()))


def _source_clause(ctx: dict) -> dict:
    if ctx["kind"] == "usr":
        return {"kind": "usr", "dataset": ctx["dataset"]}
    if ctx["kind"] == "parquet":
        return {"kind": "parquet", "file": str(ctx["file"])}
    if ctx["kind"] == "csv":
        return {"kind": "csv", "file": str(ctx["file"])}
    return {"kind": ctx["kind"]}


def _collect_existing_meta_sig(df: pd.DataFrame, name: str) -> Optional[str]:
    col = f"cluster__{name}__meta"
    if col in df.columns:
        try:
            obj = json.loads(df[col].dropna().iloc[0])
            return obj.get("sig")
        except Exception:
            return None
    return None


def _apply_preset(kind: str, preset_name: Optional[str]) -> dict:
    """Return the preset params dict ({} if not found/None)."""
    if not preset_name:
        return {}
    presets = load_presets()
    p = presets.get(preset_name)
    if p is None:
        raise typer.BadParameter(f"Preset '{preset_name}' not found.")
    if p.kind != kind:
        console.print(
            f"[yellow]Warning[/yellow]: Preset '{preset_name}' is kind='{p.kind}', "
            f"but this command expects kind='{kind}'. Using overlapping keys only."
        )
    return p.params or {}


def _apply_plot_preset(preset_name: Optional[str]) -> dict:
    if not preset_name:
        return {}
    pres = load_presets().get(preset_name)
    return (pres.plot or {}) if pres else {}


# ----------------------------- Commands -----------------------------
@app.command(
    "fit",
    help="Run Leiden clustering on X, attach minimal columns, and catalog a fit run.",
)
def cmd_fit(
    ctx: typer.Context,
    dataset: Optional[str] = typer.Option(None, help="USR dataset name"),
    file: Optional[str] = typer.Option(None, help="Parquet/CSV path"),
    usr_root: Optional[str] = typer.Option(None, help="USR root directory"),
    name: Optional[str] = typer.Option(
        None, help="Run alias (slug). If omitted, auto-generated."
    ),
    key_col: str = typer.Option("id", help="Key column"),
    x_col: Optional[str] = typer.Option(
        None, help="Vector column (list<float> or JSON array string)"
    ),
    x_cols: Optional[str] = typer.Option(
        None, help="Comma-separated list of numeric columns"
    ),
    # allow presets to fill defaults; explicit flags still win because they’re non-None
    algo: str = typer.Option("leiden", help="Clustering algorithm", show_default=True),
    neighbors: Optional[int] = typer.Option(
        None, help="kNN neighbors (Leiden); falls back to preset or 15"
    ),
    resolution: Optional[float] = typer.Option(
        None, help="Leiden resolution; falls back to preset or 0.30"
    ),
    scale: Optional[bool] = typer.Option(
        None, help="Scale X before neighbors (Leiden); falls back to preset or False"
    ),
    metric: Optional[str] = typer.Option(
        None, help='Distance metric (Leiden/UMAP); falls back to preset or "euclidean"'
    ),
    random_state: Optional[int] = typer.Option(
        None, help="Random seed; falls back to preset or 42"
    ),
    preset: Optional[str] = typer.Option(
        None, help="Preset name (kind: 'fit') to pre-fill parameters"
    ),
    silhouette: bool = typer.Option(
        False, help="Attach per-row silhouette quality as cluster__<NAME>__quality"
    ),
    full_silhouette: bool = typer.Option(
        False, help="Compute silhouette on all rows (default samples to ≤20k)"
    ),
    dedupe_policy: str = typer.Option(
        "error",
        help="Duplicate id policy: error|keep-first|keep-last",
        show_default=True,
    ),
    # Reuse
    reuse: str = typer.Option(
        "auto", help="Reuse policy: auto|require|never|reattach", show_default=True
    ),
    force: bool = typer.Option(
        False, help="Force recompute (ignore reuse cache)", show_default=True
    ),
    # Writing
    write: bool = typer.Option(False, help="Apply changes to the table"),
    yes: bool = typer.Option(
        False,
        "-y",
        "--allow-overwrite",
        help="Allow overwriting already-attached columns in USR/file writes",
    ),
    inplace: bool = typer.Option(
        False, help="Rewrite the input file in place (generic files only)"
    ),
    out: Optional[str] = typer.Option(None, help="Output file path for generic files"),
):
    if name:
        name = slugify(name)
    ictx, df = _context_and_df(dataset, file, usr_root)
    console.rule("[bold]cluster fit[/]")
    console.log(
        f"Input: kind={ictx['kind']} ref={ictx.get('dataset') or ictx.get('file')}"
    )
    # initial checks
    df = _apply_dedupe(df, key_col=key_col, policy=dedupe_policy)
    try:
        assert_id_sequence_bijection(df, id_col=key_col, seq_col="sequence")
    except ClusterError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)
    # Build X
    cols_needed = [key_col]
    if x_col:
        cols_needed.append(x_col)
    if x_cols:
        cols_needed.extend([c.strip() for c in x_cols.split(",")])
    # Reload with projection to speed up
    ictx, df = _context_and_df(
        dataset,
        file,
        usr_root,
        columns=list(
            dict.fromkeys(
                cols_needed + (["sequence"] if "sequence" in df.columns else [])
            )
        ),
    )
    df = _apply_dedupe(df, key_col=key_col, policy=dedupe_policy)
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        t_build = prog.add_task("Preparing X...", total=None)
        X = extract_X(
            df,
            x_col=x_col,
            x_cols=[c.strip() for c in x_cols.split(",")] if x_cols else None,
        )
        ids = _rows_ids(df, key_col)
        prog.update(t_build, completed=1)

    # Resolve params with preset -> defaults cascade
    p = _apply_preset("fit", preset)
    neighbors = neighbors if neighbors is not None else int(p.get("neighbors", 15))
    resolution = (
        resolution if resolution is not None else float(p.get("resolution", 0.30))
    )
    scale = bool(scale) if scale is not None else bool(p.get("scale", False))
    metric = metric if metric is not None else str(p.get("metric", "euclidean"))
    random_state = (
        random_state if random_state is not None else int(p.get("random_state", 42))
    )

    # Signatures
    source_ref = ictx["dataset"] if ictx["kind"] == "usr" else str(ictx["file"])
    inp = {
        "source_kind": ictx["kind"],
        "source_ref": source_ref,
        "key_col": key_col,
        "row_ids_hash": ids_hash(ids),
        "x_spec": {
            "mode": "single_col" if x_col else "multi_col",
            "cols": [x_col] if x_col else [c.strip() for c in x_cols.split(",")],
            "x_dim": int(X.shape[1]),
        },
        "fingerprint": file_fingerprint(ictx["file"]),
    }
    input_sig = InputSignature(**inp)
    input_hash = input_sig.hash()
    algo_sig = AlgoSignature(
        algo="leiden",
        params={
            "neighbors": neighbors,
            "resolution": resolution,
            "scale": scale,
            "metric": metric,
            "random_state": random_state,
        },
        libs={},
    ).hash()
    # Reuse logic
    if not force and reuse in ("auto", "require", "reattach"):
        hit = find_equivalent_fit(input_hash, algo_sig, root=None)
        if hit is not None:
            existing_sig = _collect_existing_meta_sig(
                df, name or hit.get("alias") or hit.get("run_slug")
            )
            if reuse in ("auto", "require") and existing_sig == algo_sig:
                console.print(
                    "[green]Reuse[/green]: matching fit already attached; nothing to do."
                )
                raise typer.Exit(code=0)
            if reuse in ("auto", "reattach") and write:
                # reattach labels from run store
                try:
                    labels_df = pd.read_parquet(hit["labels_path"])
                    attach_cols = pd.merge(
                        df[[key_col]].astype(str).rename(columns={key_col: "id"}),
                        labels_df,
                        on="id",
                        how="left",
                    )
                    attach_cols = attach_cols.rename(
                        columns={"cluster_label": f"cluster__{name or hit['alias']}"}
                    )
                    attach_cols[f"cluster__{name or hit['alias']}__meta"] = (
                        compact_meta(
                            "2.0.0",
                            "leiden",
                            x_col or "<multi>",
                            len(df),
                            {
                                "neighbors": neighbors,
                                "resolution": resolution,
                                "scale": scale,
                                "metric": metric,
                                "random_state": random_state,
                            },
                            _source_clause(ictx),
                            sig_hash=algo_sig,
                        )
                    )
                    if ictx["kind"] == "usr":
                        attach_usr(
                            ictx["usr_root"],
                            ictx["dataset"],
                            attach_cols,
                            allow_overwrite=yes,
                        )
                        console.print(
                            "[green]Reattached[/green] labels from cache to USR dataset."
                        )
                    else:
                        merged = df.merge(attach_cols, on=key_col, how="left")
                        write_generic(
                            ictx["file"],
                            merged,
                            inplace=inplace,
                            out=(Path(out) if out else None),
                            backup_suffix=".bak",
                        )
                        console.print(
                            "[green]Reattached[/green] labels from cache to file."
                        )
                    raise typer.Exit(code=0)
                except Exception as e:
                    if reuse == "require":
                        console.print(
                            f"[red]Reuse required but reattach failed:[/red] {e}"
                        )
                        raise typer.Exit(code=2)

    # Compute
    if algo != "leiden":
        raise typer.BadParameter("Only 'leiden' is supported in v2.0.")
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        t_fit = prog.add_task("Clustering (Leiden)...", total=None)
        labels = leiden_run(
            X,
            neighbors=neighbors,
            resolution=resolution,
            scale=scale,
            metric=metric,
            seed=random_state,
        )
        prog.update(t_fit, completed=1)

    # Optional: silhouette
    quality = None
    if silhouette:
        try:
            from sklearn.metrics import silhouette_samples
        except Exception:
            console.print(
                "[yellow]Silhouette requested but scikit-learn is missing. Skipping.[/yellow]"
            )
        else:
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as prog:
                t_sil = prog.add_task("Computing silhouette...", total=None)
                n = len(df)
                if n > 20000 and not full_silhouette:
                    rng = np.random.default_rng(random_state)
                    keep = rng.choice(np.arange(n), size=20000, replace=False)
                    svals = np.full(n, np.nan, dtype="float32")
                    svals[keep] = silhouette_samples(
                        X[keep], labels[keep], metric=metric
                    ).astype("float32")
                    quality = svals
                else:
                    quality = silhouette_samples(X, labels, metric=metric).astype(
                        "float32"
                    )
                prog.update(t_sil, completed=1)
    # Build attachments
    run_alias = name or auto_run_name("ldn", {"n": neighbors, "r": resolution})
    meta_json = compact_meta(
        "2.0.0",
        "leiden",
        x_col or "<multi>",
        len(df),
        {
            "neighbors": neighbors,
            "resolution": resolution,
            "scale": scale,
            "metric": metric,
            "random_state": random_state,
        },
        _source_clause(ictx),
        sig_hash=algo_sig,
    )
    attach_cols = pd.DataFrame(
        {
            "id": df[key_col].astype(str),
            f"cluster__{run_alias}": labels,
            f"cluster__{run_alias}__meta": meta_json,
        }
    )
    if quality is not None:
        attach_cols[f"cluster__{run_alias}__quality"] = quality

    # Run store bookkeeping
    root = runs_root()
    slug = run_alias
    run_dir = create_run_dir(root, slug)
    write_run_meta(
        run_dir,
        {
            "alias": run_alias,
            "slug": slug,
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "input_signature": inp,
            "algo_signature": {
                "algo": "leiden",
                "params": {
                    "neighbors": neighbors,
                    "resolution": resolution,
                    "scale": scale,
                    "metric": metric,
                    "random_state": random_state,
                },
                "libs": {},
            },
            "io": _source_clause(ictx),
            "x": {"col": x_col or "<multi>", "dim": int(X.shape[1])},
            "counts": {
                "n_rows": int(len(df)),
                "n_clusters": int(len(np.unique(labels))),
            },
            "attach": {"wrote_usr_columns": bool(ictx["kind"] == "usr")},
            "columns": [f"cluster__{run_alias}", f"cluster__{run_alias}__meta"],
        },
    )
    labels_path = write_labels(
        run_dir, pd.DataFrame({"id": df[key_col].astype(str), "cluster_label": labels})
    )
    size_counts = pd.Series(labels).value_counts().to_dict()
    write_summary(run_dir, {"cluster_sizes": size_counts})
    add_or_update_index(
        {
            "kind": "fit",
            "run_slug": slug,
            "alias": run_alias,
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "source_kind": ictx["kind"],
            "source_ref": source_ref,
            "x_col": x_col or "<multi>",
            "n_rows": int(len(df)),
            "n_clusters": int(len(np.unique(labels))),
            "algo": "leiden",
            "algo_params": {
                "neighbors": neighbors,
                "resolution": resolution,
                "scale": scale,
                "metric": metric,
                "random_state": random_state,
            },
            "input_sig_hash": input_hash,
            "labels_path": str(labels_path),
            "status": "complete",
            "umap_slug": None,
            "umap_params": None,
            "coords_path": None,
            "plot_paths": None,
        }
    )

    # Attach/write
    if not write:
        console.print(
            "[yellow]Dry-run[/yellow]: computed labels but did not write to the table. Use --write to apply."
        )
        console.print(f"Run stored under [bold]{run_dir}[/]")
        _print_fit_summary(labels, run_alias, size_counts)
        raise typer.Exit(code=0)
    if ictx["kind"] == "usr":
        try:
            attach_usr(
                ictx["usr_root"], ictx["dataset"], attach_cols, allow_overwrite=yes
            )
        except Exception as e:
            if "Columns already exist" in str(e) and not yes:
                console.print(
                    "[red]Columns already exist[/red]. Re-run with "
                    "`-y/--allow-overwrite` or choose a new --name."
                )
                raise typer.Exit(code=2)
            raise
        console.print(
            f"[green]Attached[/green] columns to USR dataset '{ictx['dataset']}'."
        )
    else:
        # merge with df
        merged = df.merge(attach_cols, on=key_col, how="left")
        write_generic(
            ictx["file"],
            merged,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
        )
        console.print("[green]Wrote[/green] updated file.")
    _print_fit_summary(labels, run_alias, size_counts)


def _print_fit_summary(labels: np.ndarray, name: str, size_counts: dict):
    tbl = Table(
        title=f"Fit summary — {name}", show_lines=False, header_style="bold cyan"
    )
    tbl.add_column("Cluster", justify="right")
    tbl.add_column("Count", justify="right")
    for cl, ct in sorted(size_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        tbl.add_row(str(cl), str(ct))
    console.print(tbl)


@app.command(
    "umap",
    help="Compute UMAP, save coords & plots under the fit run; optionally attach coords.",
)
def cmd_umap(
    ctx: typer.Context,
    dataset: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None),
    usr_root: Optional[str] = typer.Option(None),
    name: str = typer.Option(
        ..., help="Existing fit alias to associate UMAP with (uses same rows)."
    ),
    key_col: str = typer.Option("id"),
    x_col: Optional[str] = typer.Option(None),
    x_cols: Optional[str] = typer.Option(None),
    neighbors: Optional[int] = typer.Option(None, help="Falls back to preset or 15"),
    min_dist: Optional[float] = typer.Option(None, help="Falls back to preset or 0.10"),
    metric: Optional[str] = typer.Option(
        None, help='Falls back to preset or "euclidean"'
    ),
    random_state: Optional[int] = typer.Option(None, help="Falls back to preset or 42"),
    preset: Optional[str] = typer.Option(
        None, help="Preset (kind: 'umap' and optional 'plot')"
    ),
    color_by: List[str] = typer.Option(["cluster"], help="Hue specs (repeatable)."),
    highlight_ids: Optional[str] = typer.Option(
        None, help="CSV/Parquet file of ids to highlight (first column or 'id')."
    ),
    alpha: float = typer.Option(0.5),
    size: float = typer.Option(4.0),
    dims: str = typer.Option("12,12"),
    font_scale: Optional[float] = typer.Option(
        None,
        help="Scale all plot fonts (1.0 = default). Overrides preset.plot.font_scale if set.",
    ),
    opal_campaign: Optional[str] = typer.Option(
        None,
        help="Path to OPAL campaign dir or campaign name under dnadesign/opal/campaigns/",
    ),
    opal_as_of_round: Optional[int] = typer.Option(
        None, help="Filter OPAL predictions to this round"
    ),
    opal_fields: Optional[str] = typer.Option(
        None,
        help="Comma-separated OPAL prediction fields to join (e.g., pred__y_obj_scalar,obj__logic_fidelity,obj__effect_scaled).",  # noqa
    ),
    attach_coords: bool = typer.Option(False),
    out_plot: Optional[str] = typer.Option(None),
    write: bool = typer.Option(False),
    yes: bool = typer.Option(
        False,
        "-y",
        "--allow-overwrite",
        help="Allow overwriting attached coord columns",
    ),
    inplace: bool = typer.Option(False),
    out: Optional[str] = typer.Option(None),
):
    ictx, df = _context_and_df(dataset, file, usr_root)
    console.rule("[bold]cluster umap[/]")
    df = assert_no_duplicate_ids(df, key_col=key_col, policy="error")
    # Always index by id so plotting & coords/attach are consistent
    if df.index.name != key_col:
        df = df.set_index(key_col, drop=False)
    # Resolve UMAP params via preset → flags BEFORE computing
    p_umap = _apply_preset("umap", preset)
    neighbors = neighbors if neighbors is not None else int(p_umap.get("neighbors", 15))
    min_dist = min_dist if min_dist is not None else float(p_umap.get("min_dist", 0.10))
    metric = metric if metric is not None else str(p_umap.get("metric", "euclidean"))
    random_state = (
        random_state
        if random_state is not None
        else int(p_umap.get("random_state", 42))
    )

    # ---------- Preflight hues (strict, before any heavy compute) ----------
    # Resolve presets first so --preset plot.* can inject color_by
    p_plot = _apply_plot_preset(preset)
    if "color_by" in p_plot and color_by == ["cluster"]:
        color_by = list(p_plot["color_by"])
    # If df isn't indexed by id yet, do it now for consistent diagnostics
    if df.index.name != "id":
        df = df.set_index(key_col, drop=False)
    try:
        # This calls _ensure_numeric_series under the hood and will raise with detailed offenders
        from ..umap.plot import resolve_hue as _resolve_hue

        _resolve_hue(df, color_specs=color_by, name=name)
    except Exception as e:
        console.print(f"[red]Hue validation failed[/red]: {e}")
        raise typer.Exit(code=2)

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        t_build = prog.add_task("Preparing X...", total=None)
        X = extract_X(
            df,
            x_col=x_col,
            x_cols=[c.strip() for c in x_cols.split(",")] if x_cols else None,
        )
        prog.update(t_build, completed=1)
        t_umap = prog.add_task("Computing UMAP...", total=None)
        coords = umap_compute(
            X, neighbors=neighbors, min_dist=min_dist, metric=metric, seed=random_state
        )
        prog.update(t_umap, completed=1)
    # Prepare plot
    h = None
    if highlight_ids:
        p = Path(highlight_ids)
        tab = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        col = "id" if "id" in tab.columns else tab.columns[0]
        ids = set(map(str, tab[col].tolist()))
        # Ensure df index is id for convenience
        if df.index.name != "id":
            df = df.set_index(key_col, drop=False)
        missing = ids - set(map(str, df.index.tolist()))
        if missing:
            console.print(
                f"[yellow]Warning:[/yellow] {len(missing)} ids in highlight set were not found in the table."
            )
        h = {"ids": list(ids & set(map(str, df.index.tolist())))}
    # dims can be provided either via --dims "W,H" or preset.plot.dims: [W,H]
    if (
        "dims" in p_plot
        and isinstance(p_plot["dims"], (list, tuple))
        and len(p_plot["dims"]) == 2
    ):
        W, H = int(p_plot["dims"][0]), int(p_plot["dims"][1])
    else:
        W, H = [int(x) for x in dims.split(",")]
    alpha = float(p_plot.get("alpha", alpha))
    size = float(p_plot.get("size", size))
    # optional preset-provided color_by (only if user didn't change default)
    if "color_by" in p_plot and color_by == ["cluster"]:
        color_by = list(p_plot["color_by"])
    # font scale from preset unless CLI overrides
    p_font = float(p_plot.get("font_scale", 1.2))
    font_scale = float(font_scale) if font_scale is not None else p_font
    # legend from preset (defaults preserved if not provided)
    _legend = p_plot.get("legend", {})
    legend = {
        "ncol": int(_legend.get("ncol", 1)),
        "bbox": tuple(_legend.get("bbox", (1.05, 1))),
    }

    # Optional: attach OPAL metrics into df (no write-backs)
    if opal_campaign or opal_fields:
        from ..opal.join import join_fields as _opal_join

        # resolve fields
        fields = (
            [f.strip() for f in opal_fields.split(",")]
            if opal_fields
            else ["pred__y_obj_scalar", "obj__logic_fidelity", "obj__effect_scaled"]
        )
        # resolve campaign dir (path or name under dnadesign/opal/campaigns)
        camp = Path(opal_campaign) if opal_campaign else None
        if camp and not camp.exists():
            guess = (
                Path(__file__).resolve().parents[3]
                / "opal"
                / "campaigns"
                / str(opal_campaign)
            )
            if guess.exists():
                camp = guess
        if not camp or not camp.exists():
            raise typer.BadParameter(
                "OPAL campaign directory not found. Pass --opal-campaign as a path or known name."
            )
        df = _opal_join(
            df,
            campaign_dir=camp,
            run_selector="latest",
            fields=fields,
            as_of_round=opal_as_of_round,
        )
        # coverage warnings
        for c in fields:
            if c in df.columns:
                miss = float(df[c].isna().mean())
                if miss > 0:
                    console.print(
                        f"[yellow]Warning[/yellow]: joined '{c}' has {miss:.1%} missing values."
                    )
            else:
                console.print(
                    f"[red]Error[/red]: OPAL field '{c}' not present after join."
                )
                raise typer.Exit(code=2)
    # Decide output location (default: run store)
    root = runs_root()
    run_dir = root / (name or "unnamed")
    run_dir.mkdir(parents=True, exist_ok=True)
    umap_sig = UmapSignature(
        params={
            "neighbors": neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state,
        },
        libs={},
    ).hash()
    umap_slug = (
        f"umap_n{neighbors}_md{str(min_dist).replace('.','p')}_{metric}_{umap_sig[:6]}"
    )
    udir = umap_dir(run_dir, umap_slug)
    if out_plot is None:
        out_path = udir / "plots" / f"{name}.png"  # base; function appends .<label>.png
    else:
        out_path = Path(out_plot)
    umap_scatter(
        coords,
        df if df.index.name == "id" else df.set_index(key_col, drop=False),
        color_specs=color_by,
        name=name,
        highlight=h,
        alpha=alpha,
        size=size,
        dims=(W, H),
        legend=legend,
        out_path=out_path,
        font_scale=font_scale,
    )
    # Save coords & meta into run store
    coords_df = pd.DataFrame(
        {
            "id": df.index.astype(str),  # safe because we set index=id above
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        }
    )
    write_umap_coords(udir, coords_df)
    write_umap_meta(
        udir,
        {
            "slug": umap_slug,
            "alias": name,
            "params": {
                "neighbors": neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
            },
        },
    )
    add_or_update_index(
        {
            "kind": "umap",
            "run_slug": name,
            "alias": name,
            "created_utc": pd.Timestamp.utcnow().isoformat(),
            "source_kind": ictx["kind"],
            "source_ref": ictx.get("dataset") or str(ictx.get("file")),
            "x_col": x_col or "<multi>",
            "n_rows": int(len(df)),
            "n_clusters": None,
            "algo": None,
            "algo_params": None,
            "input_sig_hash": None,
            "labels_path": None,
            "status": "complete",
            "umap_slug": umap_slug,
            "umap_params": {
                "neighbors": neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
            },
            "coords_path": str(udir / "coords.parquet"),
            "plot_paths": str(udir / "plots"),
        }
    )
    # Optionally attach coords
    if attach_coords:
        cols = pd.DataFrame(
            {
                "id": df[key_col].astype(str),
                f"cluster__{name}__umap_x": coords[:, 0],
                f"cluster__{name}__umap_y": coords[:, 1],
            }
        )
        if not write:
            typer.echo("Dry-run: computed UMAP coords. Use --write to attach.")
            raise typer.Exit(code=0)
        if ictx["kind"] == "usr":
            try:
                attach_usr(ictx["usr_root"], ictx["dataset"], cols, allow_overwrite=yes)
            except Exception as e:
                if "Columns already exist" in str(e) and not yes:
                    console.print(
                        "[red]Columns already exist[/red] for UMAP coords. "
                        "Re-run with `-y/--allow-overwrite`."
                    )
                    raise typer.Exit(code=2)
                raise
            console.print("[green]Attached[/green] UMAP coords to USR dataset.")
        else:
            merged = df.merge(cols, on=key_col, how="left")
            write_generic(
                ictx["file"],
                merged,
                inplace=inplace,
                out=(Path(out) if out else None),
                backup_suffix=".bak",
            )
            console.print("[green]Wrote[/green] updated file with UMAP coords.")


@app.command(
    "sweep",
    help="Leiden resolution sweep with replicates; saves Parquet+PNG and suggested resolution.",
)
def cmd_sweep(
    dataset: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None),
    usr_root: Optional[str] = typer.Option(None),
    key_col: str = typer.Option("id"),
    x_col: Optional[str] = typer.Option(None),
    x_cols: Optional[str] = typer.Option(None),
    neighbors: int = typer.Option(15),
    res_min: float = typer.Option(0.05),
    res_max: float = typer.Option(1.0),
    step: float = typer.Option(0.05),
    replicates: int = typer.Option(5),
    seeds: str = typer.Option("1,2,3,4,5"),
    out_dir: str = typer.Option(...),
):
    ctx, df = _context_and_df(dataset, file, usr_root)
    df = assert_no_duplicate_ids(df, key_col=key_col, policy="error")
    X = extract_X(
        df,
        x_col=x_col,
        x_cols=[c.strip() for c in x_cols.split(",")] if x_cols else None,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    s = [int(x.strip()) for x in seeds.split(",")] if seeds else []
    leiden_sweep(
        X,
        neighbors=neighbors,
        res_min=res_min,
        res_max=res_max,
        step=step,
        seeds=s,
        out_dir=out_path,
    )
    console.print(f"[green]Saved sweep results[/green] to {out_path}")


@app.command(
    "analyze",
    help="Run composition/diversity/differential analyses on an existing cluster__<NAME>.",
)
def cmd_analyze(
    dataset: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None),
    usr_root: Optional[str] = typer.Option(None),
    cluster_col: str = typer.Option(...),
    group_by: str = typer.Option("source"),
    out_dir: Optional[str] = typer.Option(
        None, help="If omitted, defaults to batch_results/<FIT>/analysis/<group_by>/"
    ),
    composition: bool = typer.Option(False),
    diversity: bool = typer.Option(False),
    difffeat: bool = typer.Option(False),
    plots: bool = typer.Option(False),
    numeric: Optional[str] = typer.Option(
        None,
        help="Comma-separated numeric columns to summarize/plot per cluster (e.g., infer__...__ll_mean,opal__...__latest_pred_scalar,obj__logic_fidelity)",  # noqa
    ),
    numeric_plots: bool = typer.Option(
        True, help="Whether to render plots for --numeric"
    ),
    font_scale: float = typer.Option(1.2, help="Font scale for analysis plots"),
    opal_campaign: Optional[str] = typer.Option(
        None, help="Optional: OPAL campaign dir or name to join metrics"
    ),
    opal_as_of_round: Optional[int] = typer.Option(
        None, help="Optional: round filter for OPAL join"
    ),
    opal_fields: Optional[str] = typer.Option(
        None, help="If set, join these OPAL fields before analysis"
    ),
):
    from ..analysis.composition import composition as comp_fn
    from ..analysis.differential import differential as diff_fn
    from ..analysis.diversity import diversity as div_fn
    from ..analysis.numeric_per_cluster import summarize_numeric_by_cluster

    ctx, df = _context_and_df(dataset, file, usr_root)
    # Decide output directory **without** ever doing Path(None)
    if out_dir is None and cluster_col.startswith("cluster__"):
        fit_name = cluster_col.split("__", 1)[1]
        out = runs_root() / fit_name / "analysis" / group_by
    else:
        out = Path(out_dir or "./analysis")
    out.mkdir(parents=True, exist_ok=True)

    # Optional: OPAL join before analysis (no writes)
    if opal_campaign or opal_fields:
        from ..opal.join import join_fields as _opal_join

        fields = (
            [f.strip() for f in opal_fields.split(",")]
            if opal_fields
            else ["pred__y_obj_scalar", "obj__logic_fidelity", "obj__effect_scaled"]
        )
        camp = Path(opal_campaign) if opal_campaign else None
        if camp and not camp.exists():
            guess = (
                Path(__file__).resolve().parents[3]
                / "opal"
                / "campaigns"
                / str(opal_campaign)
            )
            if guess.exists():
                camp = guess
        if not camp or not camp.exists():
            raise typer.BadParameter(
                "OPAL campaign directory not found. Pass --opal-campaign as a path or known name."
            )
        df = _opal_join(
            df,
            campaign_dir=camp,
            run_selector="latest",
            fields=fields,
            as_of_round=opal_as_of_round,
        )
        # explicit checks + warn about missing proportions (assertive but informative)
        for c in fields:
            if c not in df.columns:
                raise typer.BadParameter(f"Joined OPAL field '{c}' missing after join.")
            miss = float(df[c].isna().mean())
            if miss > 0:
                console.print(
                    f"[yellow]Warning[/yellow]: joined '{c}' has {miss:.1%} missing values."
                )

    if composition:
        comp_fn(
            df, cluster_col=cluster_col, group_by=group_by, out_dir=out, plots=plots
        )
    if diversity:
        div_fn(df, cluster_col=cluster_col, group_by=group_by, out_dir=out, plots=plots)
    if difffeat:
        diff_fn(df, cluster_col=cluster_col, group_by=group_by, out_dir=out)
    if numeric:
        cols = [c.strip() for c in numeric.split(",") if c.strip()]
        summarize_numeric_by_cluster(
            df,
            cluster_col=cluster_col,
            numeric_cols=cols,
            out_dir=out,
            plots=numeric_plots,
            font_scale=font_scale,
        )
    console.print(f"[green]Analyses complete[/green]. Outputs at {out}")


@app.command("intra-sim")
def cmd_intra_sim(
    dataset: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None),
    usr_root: Optional[str] = typer.Option(None),
    cluster_col: str = typer.Option(...),
    out_col: str = typer.Option(...),
    match: int = typer.Option(2),
    mismatch: int = typer.Option(-1),
    gap_open: int = typer.Option(10),
    gap_extend: int = typer.Option(1),
    max_per_cluster: int = typer.Option(2000),
    sample_if_larger: bool = typer.Option(True),
    write: bool = typer.Option(False),
    yes: bool = typer.Option(False, "-y", "--allow-overwrite"),
    inplace: bool = typer.Option(False),
    out: Optional[str] = typer.Option(None),
):
    from ..analysis.intra_similarity import intra_cluster_similarity

    ictx, df = _context_and_df(dataset, file, usr_root)
    if cluster_col not in df.columns:
        raise typer.BadParameter(f"Cluster column '{cluster_col}' not found.")
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        t = prog.add_task("Computing intra-cluster similarity...", total=None)
        s = intra_cluster_similarity(
            df,
            cluster_col=cluster_col,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
            max_per_cluster=max_per_cluster,
            sample_if_larger=sample_if_larger,
        )
        prog.update(t, completed=1)
    if not write:
        console.print(
            "[yellow]Dry-run[/yellow]: computed intra-sim but did not write. Use --write to attach."
        )

        raise typer.Exit(code=0)
    cols = pd.DataFrame({"id": df["id"].astype(str), out_col: s})
    if ictx["kind"] == "usr":
        try:
            attach_usr(ictx["usr_root"], ictx["dataset"], cols, allow_overwrite=yes)
        except Exception as e:
            if "Columns already exist" in str(e) and not yes:
                console.print(
                    "[red]Columns already exist[/red]. Re-run with "
                    "`-y/--allow-overwrite`."
                )
                raise typer.Exit(code=2)
            raise
        console.print("[green]Attached[/green] intra-sim to USR dataset.")
    else:
        merged = df.merge(cols, on="id", how="left")
        write_generic(
            ictx["file"],
            merged,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
        )
        console.print("[green]Wrote[/green] updated file with intra-sim column.")


runs_app = typer.Typer(help="Run store utilities")
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list():
    df = list_runs()
    if df.empty:
        console.print("No runs recorded.")
        return
    tbl = Table(title="Recorded runs", show_lines=False, header_style="bold cyan")
    keep = [
        "kind",
        "run_slug",
        "alias",
        "created_utc",
        "source_kind",
        "x_col",
        "n_rows",
        "n_clusters",
        "algo",
        "umap_slug",
    ]
    for k in keep:
        if k in df.columns:
            tbl.add_column(k)
    for _, row in df.iterrows():
        tbl.add_row(*[str(row.get(k, "")) for k in keep if k in df.columns])
    console.print(tbl)


presets_app = typer.Typer(help="Presets utilities")
app.add_typer(presets_app, name="presets")


@presets_app.command("list")
def presets_list():
    pres = load_presets()
    for k in sorted(pres.keys()):
        typer.echo(f"{k} -> kind={pres[k].kind}")


@presets_app.command("show")
def presets_show(name: str):
    pres = load_presets()
    if name not in pres:
        raise typer.BadParameter(f"Preset '{name}' not found.")
    typer.echo(json.dumps(pres[name].dict(), indent=2))


def main():
    app()


if __name__ == "__main__":
    main()
