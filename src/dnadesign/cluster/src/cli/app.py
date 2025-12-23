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
from typing import Callable, List, Optional

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
from ..io.read import extract_X, load_table, peek_columns
from ..io.write import attach_usr, drop_usr_columns, write_generic
from ..jobs.loader import load_job_file
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
    append_records_md,
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
    help="Cluster CLI — fit, UMAP, analyses, jobs, presets. Results live under ./results/",
)
console = Console()


@app.callback(invoke_without_command=False)
def _global_opts(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show full rich tracebacks with locals."),
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


def _safe_merge_on(df: pd.DataFrame, right: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Read-only merge helper that normalizes BOTH sides:
    if either dataframe has an index level named `key_col`, make it a plain column.
    Not for write-back; use _attach_columns_schema_preserving() instead.
    """
    left = df.reset_index(drop=True) if df.index.name == key_col else df
    if right.index.name == key_col and key_col in right.columns:
        right = right.reset_index(drop=True)
    elif right.index.name == key_col and key_col not in right.columns:
        right = right.reset_index()
    if key_col not in left.columns or key_col not in right.columns:
        raise KeyError(f"Both sides must contain '{key_col}' for merge.")
    return left.merge(right, on=key_col, how="left")


def _attach_columns_schema_preserving(
    full_df: pd.DataFrame,
    cols_df: pd.DataFrame,
    key_col: str,
    *,
    allow_overwrite: bool,
) -> pd.DataFrame:
    """
    Update/create columns from cols_df into full_df by aligning on `key_col` only.
    No merges (so no _x/_y suffixes), no column loss. Assertive:
      - duplicate ids on the right → error
      - overwriting existing columns without permission → error
    """
    if key_col not in full_df.columns:
        raise KeyError(f"Left table is missing key column '{key_col}'.")
    left = full_df.reset_index(drop=True) if full_df.index.name == key_col else full_df
    right = cols_df
    if right.index.name == key_col and key_col in right.columns:
        right = right.reset_index(drop=True)
    elif right.index.name == key_col and key_col not in right.columns:
        right = right.reset_index()
    if key_col not in right.columns:
        raise KeyError(f"Right table is missing key column '{key_col}'.")
    if right[key_col].duplicated().any():
        dupes = right.loc[right[key_col].duplicated(), key_col].astype(str).head(8).tolist()
        raise RuntimeError(f"Right table has duplicate '{key_col}' values (e.g., {dupes}).")
    # Try to align dtype of the join key without mutating full_df permanently
    try:
        right = right.copy()
        right[key_col] = right[key_col].astype(left[key_col].dtype)
    except Exception:
        left = left.copy()
        left[key_col] = left[key_col].astype(str)
        right[key_col] = right[key_col].astype(str)
    li = left.set_index(key_col, drop=False)
    ri = right.set_index(key_col, drop=False)
    to_attach = [c for c in ri.columns if c != key_col]
    if not to_attach:
        return left
    existing = [c for c in to_attach if c in li.columns]
    if existing and not allow_overwrite:
        raise RuntimeError(
            "Columns already exist: "
            + ", ".join(existing[:8])
            + (" ..." if len(existing) > 8 else "")
            + ". Re-run with `-y/--allow-overwrite` or use a new --name."
        )
    for c in to_attach:
        li[c] = ri[c].reindex(li.index).values
    return li.reset_index(drop=True) if full_df.index.name != key_col else li


def _normalize_for_key(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Ensure there is no 'label-or-level' ambiguity for `key_col`.
    - If index is named `key_col` and a same-named column exists, drop the index name.
    - If index is named `key_col` and the column does not exist, materialize it.
    Always return a frame whose *index is not named key_col* and that *has* key_col column.
    """
    out = df
    if out.index.name == key_col and key_col in out.columns:
        out = out.reset_index(drop=True)
    elif out.index.name == key_col and key_col not in out.columns:
        out = out.reset_index()
    if key_col not in out.columns:
        raise KeyError(f"Left table must contain '{key_col}' column.")
    return out


def _attach_by_key_update(left: pd.DataFrame, right: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Schema-preserving attach:
      • For overlapping columns (same name in left & right, excluding key), UPDATE values aligned by key.
      • For new columns (only in right), LEFT JOIN them.
    Never produces _x/_y suffixes, never drops left columns, and avoids 'id'-ambiguity.
    """
    L = _normalize_for_key(left.copy(), key_col)
    R = _normalize_for_key(right.copy(), key_col)
    # Align types for the key (string is safest / most consistent in our codebase)
    L[key_col] = L[key_col].astype(str)
    R[key_col] = R[key_col].astype(str)
    Li = L.set_index(key_col, drop=False)
    Ri = R.set_index(key_col, drop=False)
    # 1) Update overlapping columns in place
    overlap = [c for c in Ri.columns if c != key_col and c in Li.columns]
    if overlap:
        Li.update(Ri[overlap])
    # 2) Join brand‑new columns
    new_cols = [c for c in Ri.columns if c != key_col and c not in Li.columns]
    if new_cols:
        Li = Li.join(Ri[new_cols], how="left")
    # Return with a regular RangeIndex (avoid future label/level ambiguity)
    return Li.reset_index(drop=True)


def _assert_preserve_columns(before: list[str], after: list[str]) -> None:
    """Assert that *no* original top‑level column would be dropped."""
    missing = [c for c in before if c not in after]
    if missing:
        raise RuntimeError(
            "Refusing to write: detected potential column drop.\n"
            "Columns that would be lost: " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")
        )


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


def _apply_job_params(job_path: Optional[str], expected_command: str) -> dict:
    """
    Load a job YAML and return its params dict. Assert 'command' matches the CLI command.
    """
    if not job_path:
        return {}
    job = load_job_file(job_path)
    cmd = job.get("command", "").strip().lower()
    if cmd and cmd != expected_command:
        raise typer.BadParameter(f"Job '{job_path}' has command='{cmd}' but this subcommand is '{expected_command}'.")
    params = job.get("params", {}) or {}
    if not isinstance(params, dict):
        raise typer.BadParameter(f"Job '{job_path}': 'params' must be a mapping.")
    return params


def _assert_no_algo_overlap_with_preset(kind: str, job_params: dict, preset_name: Optional[str]) -> None:
    if not preset_name:
        return
    # Disallow common algo keys per kind when a preset is provided.
    fit_keys = {"neighbors", "resolution", "scale", "metric", "random_state", "algo"}
    umap_keys = {"neighbors", "min_dist", "metric", "random_state"}
    banned = fit_keys if kind == "fit" else (umap_keys if kind == "umap" else set())
    overlap = sorted(k for k in job_params.keys() if k in banned)
    if overlap:
        import typer

        raise typer.BadParameter(
            f"Job provides {overlap} but also references a preset. "
            f"Move algorithm knobs into the preset or pass via CLI flags."
        )


def _apply_job_plot(job_path: Optional[str], expected_command: str) -> dict:
    """
    Load a job YAML and return its *plot* mapping ({} if absent). Validates 'command'.
    """
    if not job_path:
        return {}
    job = load_job_file(job_path)
    cmd = (job.get("command", "") or "").strip().lower()
    if cmd and cmd != expected_command:
        raise typer.BadParameter(f"Job '{job_path}' has command='{cmd}' but this subcommand is '{expected_command}'.")
    plot = job.get("plot", {}) or {}
    if not isinstance(plot, dict):
        raise typer.BadParameter(f"Job '{job_path}': 'plot' must be a mapping.")
    return plot


def _resolve_color_by(cli_val, jp_params, jp_plot_cfg, preset_plot_cfg):
    """
    Resolve final color_by with precedence:
    CLI > job.plot > job.params > preset.plot > ["cluster"].
    If the user passed --color-by, respect it unless it is the bare default ["cluster"].
    """
    if cli_val and not (len(cli_val) == 1 and cli_val[0] == "cluster"):
        return list(cli_val)
    if isinstance(jp_plot_cfg.get("color_by"), (list, tuple)):
        return list(jp_plot_cfg["color_by"])
    if isinstance(jp_params.get("color_by"), (list, tuple)):
        return list(jp_params["color_by"])
    if isinstance(preset_plot_cfg.get("color_by"), (list, tuple)):
        return list(preset_plot_cfg["color_by"])
    return ["cluster"]


def _load_highlight_ids_from_file(
    path_str: str,
    df: pd.DataFrame,
    key_col: str,
    warn_fn: Optional[Callable[[str], None]] = None,
    groupby_col: Optional[str] = None,
) -> dict:
    """
    Read a CSV/Parquet of ids for highlighting and intersect with the dataset rows.
    Returns a dict with:
      - "ids": List[str] present in the dataset
      - optional "labels": Dict[id -> category string] if groupby_col is provided
      - optional "by": the groupby column name (echoed back for legend title)
      - optional "categories": List[str] of discovered categories (sorted)
    This is a pure helper with no global side-effects.
    """
    p = Path(path_str)
    if not p.exists():
        raise typer.BadParameter(f"--highlight path not found: {p}")
    tab = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    col = "id" if "id" in tab.columns else tab.columns[0]
    raw_ids = set(map(str, tab[col].tolist()))
    # Ensure df is indexed by id for a deterministic intersection (do not mutate caller)
    left = df if df.index.name == key_col else df.set_index(key_col, drop=False)
    present = raw_ids & set(map(str, left.index.astype(str).tolist()))
    missing = raw_ids - present
    if missing and warn_fn:
        warn_fn(f"{len(missing)} id(s) in highlight were not found in the dataset. They will be ignored.")
    out = {"ids": list(present)}
    if groupby_col is not None:
        if groupby_col not in tab.columns:
            raise typer.BadParameter(f"--highlight-hue-col='{groupby_col}' not found in {p.name}.")
        # Build id -> category mapping (treat as categorical; integers become strings)
        sub = tab[[col, groupby_col]].copy()
        sub[col] = sub[col].astype(str)
        sub[groupby_col] = sub[groupby_col].astype(str)
        labels = {rid: cat for rid, cat in zip(sub[col].tolist(), sub[groupby_col].tolist()) if rid in present}
        cats = sorted(set(labels.values()))
        out.update({"labels": labels, "by": groupby_col, "categories": cats})
    return out


# ----------------------------- Commands -----------------------------
@app.command(
    "fit",
    help="Run Leiden clustering on X, attach minimal columns, and catalog a fit run.",
)
def cmd_fit(
    ctx: typer.Context,
    job: Optional[str] = typer.Option(None, help="Path to a job YAML for 'fit'."),
    dataset: Optional[str] = typer.Option(None, help="USR dataset name"),
    file: Optional[str] = typer.Option(None, help="Parquet/CSV path"),
    usr_root: Optional[str] = typer.Option(None, help="USR root directory"),
    name: Optional[str] = typer.Option(None, help="Run alias (slug). If omitted, auto-generated."),
    key_col: str = typer.Option("id", help="Key column"),
    x_col: Optional[str] = typer.Option(None, help="Vector column (list<float> or JSON array string)"),
    x_cols: Optional[str] = typer.Option(None, help="Comma-separated list of numeric columns"),
    # allow presets to fill defaults; explicit flags still win because they’re non-None
    algo: str = typer.Option("leiden", help="Clustering algorithm", show_default=True),
    neighbors: Optional[int] = typer.Option(None, help="kNN neighbors (Leiden); falls back to preset or 15"),
    resolution: Optional[float] = typer.Option(None, help="Leiden resolution; falls back to preset or 0.30"),
    scale: Optional[bool] = typer.Option(None, help="Scale X before neighbors (Leiden); falls back to preset or False"),
    metric: Optional[str] = typer.Option(
        None, help='Distance metric (Leiden/UMAP); falls back to preset or "euclidean"'
    ),
    random_state: Optional[int] = typer.Option(None, help="Random seed; falls back to preset or 42"),
    preset: Optional[str] = typer.Option(None, help="Preset name (kind: 'fit') to pre-fill parameters"),
    silhouette: bool = typer.Option(False, help="Attach per-row silhouette quality as cluster__<NAME>__quality"),
    full_silhouette: bool = typer.Option(False, help="Compute silhouette on all rows (default samples to ≤20k)"),
    dedupe_policy: str = typer.Option(
        "error",
        help="Duplicate id policy: error|keep-first|keep-last",
        show_default=True,
    ),
    # Reuse
    reuse: str = typer.Option("auto", help="Reuse policy: auto|require|never|reattach", show_default=True),
    force: bool = typer.Option(False, help="Force recompute (ignore reuse cache)", show_default=True),
    # Writing
    write: bool = typer.Option(False, help="Apply changes to the table"),
    yes: bool = typer.Option(
        False,
        "-y",
        "--allow-overwrite",
        help="Allow overwriting already-attached columns in USR/file writes",
    ),
    inplace: bool = typer.Option(False, help="Rewrite the input file in place (generic files only)"),
    out: Optional[str] = typer.Option(None, help="Output file path for generic files"),
):
    # Apply job params first (flags still override)
    jp = _apply_job_params(job, expected_command="fit")
    dataset = dataset or jp.get("dataset")
    file = file or jp.get("file")
    usr_root = usr_root or jp.get("usr_root")
    name = name or jp.get("name")
    key_col = key_col or jp.get("key_col", "id")
    x_col = x_col or jp.get("x_col")
    if x_col:
        x_col = str(x_col).strip()
    x_cols = x_cols or jp.get("x_cols")
    algo = algo or jp.get("algo", "leiden")
    neighbors = neighbors if neighbors is not None else jp.get("neighbors")
    resolution = resolution if resolution is not None else jp.get("resolution")
    scale = scale if scale is not None else jp.get("scale")
    metric = metric or jp.get("metric")
    random_state = random_state if random_state is not None else jp.get("random_state")
    preset = preset or jp.get("preset")
    _assert_no_algo_overlap_with_preset("fit", jp, preset)
    silhouette = bool(silhouette or jp.get("silhouette", False))
    full_silhouette = bool(full_silhouette or jp.get("full_silhouette", False))
    dedupe_policy = jp.get("dedupe_policy", dedupe_policy)
    reuse = jp.get("reuse", reuse)
    force = bool(force or jp.get("force", False))
    write = bool(write or jp.get("write", False))
    yes = bool(yes or jp.get("allow_overwrite", False))
    inplace = bool(inplace or jp.get("inplace", False))
    out = out or jp.get("out")
    if name:
        name = slugify(name)
    ictx, df_full = _context_and_df(dataset, file, usr_root)
    console.rule("[bold]cluster fit[/]")
    console.log(f"Input: kind={ictx['kind']} ref={ictx.get('dataset') or ictx.get('file')}")
    # initial checks
    df = _apply_dedupe(df_full, key_col=key_col, policy=dedupe_policy)
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
        columns=list(dict.fromkeys(cols_needed + (["sequence"] if "sequence" in df.columns else []))),
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
    resolution = resolution if resolution is not None else float(p.get("resolution", 0.30))
    scale = bool(scale) if scale is not None else bool(p.get("scale", False))
    metric = metric if metric is not None else str(p.get("metric", "euclidean"))
    random_state = random_state if random_state is not None else int(p.get("random_state", 42))

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
            existing_sig = _collect_existing_meta_sig(df, name or hit.get("alias") or hit.get("run_slug"))
            if reuse in ("auto", "require") and existing_sig == algo_sig:
                console.print("[green]Reuse[/green]: matching fit already attached; nothing to do.")
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
                    attach_cols = attach_cols.rename(columns={"cluster_label": f"cluster__{name or hit['alias']}"})
                    attach_cols[f"cluster__{name or hit['alias']}__meta"] = compact_meta(
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
                    if ictx["kind"] == "usr":
                        attach_usr(
                            ictx["usr_root"],
                            ictx["dataset"],
                            attach_cols,
                            allow_overwrite=yes,
                        )
                        console.print("[green]Reattached[/green] labels from cache to USR dataset.")
                    else:
                        full_df = load_table(ictx)
                        merged = _attach_columns_schema_preserving(full_df, attach_cols, key_col, allow_overwrite=yes)
                        _assert_preserve_columns(list(full_df.columns), list(merged.columns))
                        write_generic(
                            ictx["file"],
                            merged,
                            inplace=inplace,
                            out=(Path(out) if out else None),
                            backup_suffix=".bak",
                        )
                        console.print("[green]Reattached[/green] labels from cache to file.")
                    raise typer.Exit(code=0)
                except Exception as e:
                    if reuse == "require":
                        console.print(f"[red]Reuse required but reattach failed:[/red] {e}")
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
            console.print("[yellow]Silhouette requested but scikit-learn is missing. Skipping.[/yellow]")
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
                    svals[keep] = silhouette_samples(X[keep], labels[keep], metric=metric).astype("float32")
                    quality = svals
                else:
                    quality = silhouette_samples(X, labels, metric=metric).astype("float32")
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
    labels_path = write_labels(run_dir, pd.DataFrame({"id": df[key_col].astype(str), "cluster_label": labels}))
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
        console.print("[yellow]Dry-run[/yellow]: computed labels but did not write to the table. Use --write to apply.")
        console.print(f"Run recorded under [bold]{run_dir}[/] (results root).")
        _print_fit_summary(labels, run_alias, size_counts)
        raise typer.Exit(code=0)
    if ictx["kind"] == "usr":
        try:
            attach_usr(ictx["usr_root"], ictx["dataset"], attach_cols, allow_overwrite=yes)
        except Exception as e:
            if "Columns already exist" in str(e) and not yes:
                console.print(
                    "[red]Columns already exist[/red]. Re-run with `-y/--allow-overwrite` or choose a new --name."
                )
                raise typer.Exit(code=2)
            raise
        console.print(f"[green]Attached[/green] columns to USR dataset '{ictx['dataset']}'.")
    else:
        merged = _attach_columns_schema_preserving(df_full, attach_cols, key_col, allow_overwrite=yes)
        write_generic(
            ictx["file"],
            merged,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
        )
        console.print("[green]Wrote[/green] updated file.")
    _print_fit_summary(labels, run_alias, size_counts)
    # ---- Records sink (Markdown) ----
    try:
        effective = {
            "command": "fit",
            "job": job or None,
            "preset": preset or None,
            "resolved": {
                "name": run_alias,
                "algo": "leiden",
                "neighbors": neighbors,
                "resolution": resolution,
                "scale": bool(scale),
                "metric": metric,
                "random_state": random_state,
            },
        }
        md = f"## cluster fit — {run_alias}\n\n```json\n{json.dumps(effective, indent=2, sort_keys=True)}\n```"
        append_records_md(run_dir, md)
    except Exception:
        pass


def _print_fit_summary(labels: np.ndarray, name: str, size_counts: dict):
    tbl = Table(title=f"Fit summary — {name}", show_lines=False, header_style="bold cyan")
    tbl.add_column("Cluster", justify="right")
    tbl.add_column("Count", justify="right")
    for cl, ct in sorted(size_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        tbl.add_row(str(cl), str(ct))
    console.print(tbl)


@app.command(
    "delete-columns",
    help="Delete cluster__*-namespaced columns from a dataset/file with a safety preview and confirmation.",
)
def cmd_delete_columns(
    dataset: Optional[str] = typer.Option(None, help="USR dataset name"),
    file: Optional[str] = typer.Option(None, help="Parquet/CSV path"),
    usr_root: Optional[str] = typer.Option(None, help="USR root directory"),
    # Scope selection (choose exactly one of --all, --name, --column)
    all_: bool = typer.Option(False, "--all", help="Delete ALL cluster__* columns"),
    name: List[str] = typer.Option(
        [],
        "--name",
        help="Delete columns for this fit alias (repeatable). Matches cluster__<name> and cluster__<name>__*",
    ),
    column: List[str] = typer.Option(
        [],
        "--column",
        help="Delete this fully-qualified column (repeatable). Must start with cluster__",
    ),
    # Execution controls
    write: bool = typer.Option(False, help="Apply changes (default is dry-run)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip interactive confirmation"),
    inplace: bool = typer.Option(
        False,
        help="For generic files: rewrite the input file in place (backs up to .bak)",
    ),
    out: Optional[str] = typer.Option(None, help="For generic files: write to this output path instead of --inplace"),
):
    # Resolve context
    ictx, _ = _context_and_df(dataset, file, usr_root, columns=None)
    console.rule("[bold]cluster delete-columns[/]")
    console.log(f"Input: kind={ictx['kind']} ref={ictx.get('dataset') or ictx.get('file')}")

    # ---------- Build deletion set ----------
    if sum(bool(x) for x in [all_, bool(name), bool(column)]) != 1:
        raise typer.BadParameter("Choose exactly one: --all OR --name ... OR --column ...")

    cols = peek_columns(ictx)
    # Always work with **top‑level** names (peek_columns already returns them for Parquet).
    # If callers pass dotted leaf paths via --column, normalize them below.
    cluster_cols = [c for c in cols if c.startswith("cluster__")]

    if all_:
        to_delete = cluster_cols
        reason = "all cluster__*"
    elif name:
        name = [slugify(n) for n in name]
        prefixes = [f"cluster__{n}" for n in name]
        to_delete = [c for c in cluster_cols if any(c == p or c.startswith(p + "__") for p in prefixes)]
        reason = "name=" + ",".join(name)
    else:
        # Normalize any dotted leaf paths to their top‑level parent
        normalized_requested = [c.split(".", 1)[0] for c in column]
        bad = [c for c in normalized_requested if not c.startswith("cluster__")]
        if bad:
            raise typer.BadParameter("Only 'cluster__*' columns can be deleted; offending: " + ", ".join(bad[:6]))
        # Only delete columns that actually exist
        to_delete = [c for c in normalized_requested if c in cluster_cols]
        missing = [c for c in normalized_requested if c not in cols]
        if missing:
            console.print(
                "[yellow]Note[/yellow]: the following columns were not found and will be ignored: "
                + ", ".join(missing[:8])
                + (" ..." if len(missing) > 8 else "")
            )
        reason = "explicit columns"

    if not to_delete:
        console.print("[green]Nothing to delete[/green]: no matching cluster__ columns found.")
        raise typer.Exit(code=0)

    # ---------- Preview ----------
    tbl = Table(
        title=f"Columns to delete ({len(to_delete)}) — scope: {reason}",
        header_style="bold cyan",
    )
    tbl.add_column("Column")
    for c in sorted(to_delete):
        tbl.add_row(c)
    console.print(tbl)

    # ---------- Confirmation ----------
    if not yes:
        if not typer.confirm(f"Are you sure you want to permanently delete {len(to_delete)} column(s)?"):
            console.print("Aborted by user.")
            raise typer.Exit(code=1)

    if not write:
        console.print("[yellow]Dry-run[/yellow]: no changes applied. Re-run with --write to proceed.")
        raise typer.Exit(code=0)

    # ---------- Execute ----------
    if ictx["kind"] == "usr":
        drop_usr_columns(ictx["usr_root"], ictx["dataset"], to_delete)
        console.print(f"[green]Removed[/green] {len(to_delete)} column(s) from USR dataset '{ictx['dataset']}'.")
    else:
        # Generic files: load, drop, write back (with backup if --inplace)
        df = pd.read_parquet(ictx["file"]) if ictx["kind"] == "parquet" else pd.read_csv(ictx["file"])
        missing_at_exec = [c for c in to_delete if c not in df.columns]
        if missing_at_exec:
            console.print(
                "[yellow]Note[/yellow]: some columns disappeared during load and were skipped: "
                + ", ".join(missing_at_exec[:8])
                + (" ..." if len(missing_at_exec) > 8 else "")
            )
        kept = [c for c in to_delete if c in df.columns]
        if kept:
            df = df.drop(columns=kept)
        else:
            console.print("[green]Nothing left to delete[/green].")
            raise typer.Exit(code=0)
        write_generic(
            ictx["file"],
            df,
            inplace=inplace,
            out=(Path(out) if out else None),
            backup_suffix=".bak",
        )
        console.print(f"[green]Wrote[/green] updated file ({'inplace' if inplace else 'out=' + str(out)}).")

    # ---------- Recap ----------
    recap = Table(title="Deleted columns recap", header_style="bold cyan")
    recap.add_column("Count", justify="right")
    recap.add_column("Preview")
    preview = ", ".join(sorted(to_delete)[:6]) + (" ..." if len(to_delete) > 6 else "")
    recap.add_row(str(len(to_delete)), preview)
    console.print(recap)


@app.command(
    "umap",
    help="Compute UMAP, save coords & plots under the fit run; optionally attach coords.",
)
def cmd_umap(
    _ctx: typer.Context,
    job: Optional[str] = typer.Option(None, help="Path to a job YAML for 'umap'."),
    dataset: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None),
    usr_root: Optional[str] = typer.Option(None),
    name: Optional[str] = typer.Option(None, help="Existing fit alias to associate UMAP with (uses same rows)."),
    key_col: str = typer.Option("id"),
    x_col: Optional[str] = typer.Option(None),
    x_cols: Optional[str] = typer.Option(None),
    neighbors: Optional[int] = typer.Option(None, help="Falls back to preset or 15"),
    min_dist: Optional[float] = typer.Option(None, help="Falls back to preset or 0.10"),
    metric: Optional[str] = typer.Option(None, help='Falls back to preset or "euclidean"'),
    random_state: Optional[int] = typer.Option(None, help="Falls back to preset or 42"),
    preset: Optional[str] = typer.Option(None, help="Preset (kind: 'umap' and optional 'plot')"),
    color_by: List[str] = typer.Option(["cluster"], help="Hue specs (repeatable). Includes 'highlight'."),
    highlight: Optional[str] = typer.Option(None, help="CSV/Parquet with ids to highlight (first column or 'id')."),
    highlight_topn: Optional[int] = typer.Option(
        None,
        help="Highlight Top-N rows from the primary table by ranking a numeric column (use with --highlight-topn-col).",
    ),
    highlight_topn_col: Optional[str] = typer.Option(
        None,
        help="Numeric column to rank for --highlight-topn (e.g., 'permuter__metric__llr_mean').",
    ),
    highlight_topn_asc: bool = typer.Option(
        False,
        "--highlight-topn-asc",
        help="If set, select the smallest N values (ascending) instead of largest.",
    ),
    highlight_hue_col: Optional[str] = typer.Option(
        None,
        help="Optional. If set, color highlights categorically by this column from the --highlight file "
        "(e.g., 'observed_round'). Integers are treated as categories.",
    ),
    alpha: Optional[float] = typer.Option(None, help="Point alpha (overrides job/preset)."),
    size: Optional[float] = typer.Option(None, help="Point size (overrides job/preset)."),
    dims: Optional[str] = typer.Option(None, help="Figure size 'W,H' (overrides job/preset)."),
    font_scale: Optional[float] = typer.Option(
        None,
        help="Scale all plot fonts (1.0 = default). Overrides preset.plot.font_scale if set.",
    ),
    opal_campaign: Optional[str] = typer.Option(
        None,
        help="Path to OPAL campaign dir or campaign name under dnadesign/opal/campaigns/",
    ),
    opal_run: Optional[str] = typer.Option(
        None,
        help="OPAL run selector: 'latest', 'round:<n>', or 'run_id:<rid>' "
        "(mutually exclusive with --opal-as-of-round).",
    ),
    opal_as_of_round: Optional[int] = typer.Option(None, help="Filter OPAL predictions to this round"),
    opal_fields: Optional[str] = typer.Option(
        None,
        help="Comma-separated OPAL prediction fields to join (e.g., pred__y_obj_scalar,obj__logic_fidelity,obj__effect_scaled).",  # noqa
    ),
    derive_ratio: List[str] = typer.Option(
        [],
        help="Repeatable. Define a derived ratio column: '<new_col>:<numerator_col>:<denominator_col>'.",
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
    # Job params (flags win)
    jp = _apply_job_params(job, expected_command="umap")
    jp_plot = _apply_job_plot(job, expected_command="umap")
    dataset = dataset or jp.get("dataset")
    file = file or jp.get("file")
    usr_root = usr_root or jp.get("usr_root")
    name = name or jp.get("name")
    if not name:
        raise typer.BadParameter("UMAP requires a fit alias. Provide --name or set params.name in the job YAML.")
    key_col = key_col or jp.get("key_col", "id")
    x_col = x_col or jp.get("x_col")
    if x_col:
        x_col = str(x_col).strip()
    x_cols = x_cols or jp.get("x_cols")
    neighbors = neighbors if neighbors is not None else jp.get("neighbors")
    min_dist = min_dist if min_dist is not None else jp.get("min_dist")
    metric = metric or jp.get("metric")
    random_state = random_state if random_state is not None else jp.get("random_state")
    preset = preset or jp.get("preset")
    _assert_no_algo_overlap_with_preset("umap", jp, preset)
    if color_by == ["cluster"] and isinstance(jp.get("color_by"), (list, tuple)):
        color_by = list(jp["color_by"])
    highlight = highlight or jp.get("highlight")
    highlight_hue_col = highlight_hue_col or jp.get("highlight_hue_col")
    # BUGFIX: read highlight_topn knobs from job params
    highlight_topn = highlight_topn if highlight_topn is not None else jp.get("highlight_topn")
    highlight_topn_col = highlight_topn_col or jp.get("highlight_topn_col")
    highlight_topn_asc = bool(highlight_topn_asc or jp.get("highlight_topn_asc", False))
    alpha = alpha if alpha is not None else jp.get("alpha")
    size = size if size is not None else jp.get("size")
    dims = dims if dims is not None else jp.get("dims")
    font_scale = font_scale if font_scale is not None else jp.get("font_scale")
    opal_campaign = opal_campaign or jp.get("opal_campaign")
    opal_run = opal_run or jp.get("opal_run")
    opal_as_of_round = opal_as_of_round or jp.get("opal_as_of_round")
    opal_fields = opal_fields or (
        ",".join(jp["opal_fields"]) if isinstance(jp.get("opal_fields"), (list, tuple)) else jp.get("opal_fields")
    )
    attach_coords = bool(attach_coords or jp.get("attach_coords", False))
    out_plot = out_plot or jp.get("out_plot")
    write = bool(write or jp.get("write", False))
    yes = bool(yes or jp.get("allow_overwrite", False))
    inplace = bool(inplace or jp.get("inplace", False))
    out = out or jp.get("out")
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
    random_state = random_state if random_state is not None else int(p_umap.get("random_state", 42))

    # ---------- Plot preset & OPAL preflight (join before hue validation) ----------
    # Resolve presets first so --preset plot.* can inject color_by
    p_plot = _apply_plot_preset(preset)

    color_by = _resolve_color_by(color_by, jp, jp_plot, p_plot)
    wants_highlight = bool(highlight or highlight_topn)
    if wants_highlight and "highlight" not in color_by:
        color_by = [*color_by, "highlight"]

    # Incorporate job.plot now; precedence: preset.plot -> job.plot -> CLI flags
    # Defaults for plotting (used if neither preset nor job nor CLI provides)
    _plot_defaults = {
        "alpha": 0.5,
        "size": 4.0,
        "dims": [12, 12],
        "font_scale": 1.2,
        "legend": {"ncol": 1, "bbox": (1.02, 1.0), "max_items": 40, "frameon": False},
        # New: explicit highlight style block (tunable via preset.plot.highlight or job.plot.highlight)
        "highlight": {
            "overlay": True,
            "size_multiplier": 1.6,  # used only if 'size' is not provided
            "alpha": 0.95,
            "facecolor": "none",
            "edgecolor": "red",
            "linewidth": 0.9,
            "marker": "o",
            "legend": False,
        },
    }
    # Start with preset.plot, overlay job.plot, overlay CLI values (if provided)
    merged_plot = {**_plot_defaults, **p_plot, **jp_plot}
    if alpha is not None:
        merged_plot["alpha"] = float(alpha)
    if size is not None:
        merged_plot["size"] = float(size)
    if dims is not None:
        merged_plot["dims"] = dims
    if font_scale is not None:
        merged_plot["font_scale"] = float(font_scale)

    # ---- Highlight ids: assert early and load once (used for validation and plotting) ----
    # If 'highlight' hue requested but neither mode is provided, fail fast.
    if "highlight" in color_by and not wants_highlight:
        console.print(
            "[red]Error[/red]: hue 'highlight' was requested (via preset/job), "
            "but neither --highlight <file> nor --highlight-topn was provided."
        )
        raise typer.Exit(code=2)
    h = None
    if highlight and highlight_topn:
        raise typer.BadParameter("Use either --highlight (file) OR --highlight-topn, not both.")
    if highlight:
        h = _load_highlight_ids_from_file(
            highlight,
            df,
            key_col,
            warn_fn=lambda m: console.print(f"[yellow]Warning:[/yellow] {m}"),
            groupby_col=highlight_hue_col,
        )
    elif highlight_topn is not None:
        if not highlight_topn_col:
            raise typer.BadParameter("--highlight-topn requires --highlight-topn-col.")
        if highlight_hue_col:
            console.print("[yellow]Note[/yellow]: --highlight-hue-col is ignored when using --highlight-topn.")
        if highlight_topn <= 0:
            raise typer.BadParameter("--highlight-topn must be a positive integer.")
        if highlight_topn_col not in df.columns:
            raise typer.BadParameter(f"--highlight-topn-col '{highlight_topn_col}' not found in the table.")
        # Strict numeric; we drop non-finite with a concise log (explicit behavior)
        try:
            s = pd.to_numeric(df[highlight_topn_col], errors="raise")
        except Exception as e:
            raise typer.BadParameter(f"--highlight-topn-col '{highlight_topn_col}' is not numeric: {e}")
        arr = s.to_numpy(dtype="float64", copy=False)
        nonfinite = ~np.isfinite(arr)
        if nonfinite.any():
            n_bad = int(nonfinite.sum())
            console.print(
                f"[yellow]Note[/yellow]: excluding {n_bad} non-finite row(s) from Top-N selection of '{highlight_topn_col}'."  # noqa
            )
            s = s[~nonfinite]
        # Rank and pick top/bottom N
        order = s.sort_values(ascending=bool(highlight_topn_asc))
        take = int(min(len(order), int(highlight_topn)))
        chosen_idx = order.iloc[:take].index
        # Map back to id strings; df is indexed by id (we enforced earlier), but keep robust:
        ids = (
            pd.Index(chosen_idx).astype(str).tolist()
            if df.index.name == key_col
            else df.loc[chosen_idx, key_col].astype(str).tolist()
        )
        h = {"ids": ids}
    # Derive which OPAL fields are actually needed from the hue specs,
    # and union with any explicit --opal-fields
    opal_needed_fields: set[str] = set()
    for spec in color_by:
        if spec.startswith(("numeric:", "categorical:")):
            col = spec.split(":", 1)[1]
            if col.startswith(("obj__", "pred__", "sel__")) and col not in df.columns:
                opal_needed_fields.add(col)
    if opal_fields:
        opal_needed_fields |= {c.strip() for c in opal_fields.split(",") if c.strip()}

    # If OPAL columns are requested by hues and not present, join them now
    if opal_needed_fields:
        if not opal_campaign:
            console.print(
                "[red]Error[/red]: The selected hues require OPAL predictions "
                f"(missing {', '.join(sorted(opal_needed_fields))}).\n"
                "Provide --opal-campaign (path or known name) and *either* "
                "--opal-run latest|round:<n>|run_id:<rid> or --opal-as-of-round <n>."
            )
            raise typer.Exit(code=2)
        if opal_run and opal_as_of_round is not None:
            raise typer.BadParameter("Use only one of --opal-run or --opal-as-of-round, not both.")
        from ..opal.join import (
            join_fields as _opal_join,
        )
        from ..opal.join import (
            list_available_fields as _opal_list,
        )
        from ..opal.join import (
            resolve_campaign_dir as _resolve_campaign_dir,
        )

        # Resolve campaign directory deterministically
        try:
            camp = _resolve_campaign_dir(opal_campaign)
        except FileNotFoundError as e:
            raise typer.BadParameter(str(e))
        # Always index by id so diagnostics stay consistent
        if df.index.name != key_col:
            df = df.set_index(key_col, drop=False)
        # Join only the columns we need
        df = _opal_join(
            df,
            campaign_dir=camp,
            run_selector=(opal_run or "latest"),
            fields=sorted(opal_needed_fields),
            as_of_round=opal_as_of_round,
            log_fn=lambda m: console.log(m),
        )
        # Assert post-join coverage and show discoverable alternatives if something is missing
        missing_after = [c for c in opal_needed_fields if c not in df.columns]
        if missing_after:
            avail = _opal_list(camp, run_selector=(opal_run or "latest"), as_of_round=opal_as_of_round)[:60]
            console.print(
                "[red]Error[/red]: OPAL join did not provide the requested column(s): " + ", ".join(missing_after)
            )
            console.print(
                "Available OPAL columns for this run/round include: "
                + ", ".join(avail)
                + (" ..." if len(avail) == 60 else "")
            )
            raise typer.Exit(code=2)

    # ---------- Preflight hue validation (after OPAL join) ----------
    # read derive_ratio from job params if not passed via CLI
    if not derive_ratio and jp.get("derive_ratio"):
        derive_ratio = (
            list(jp["derive_ratio"]) if isinstance(jp["derive_ratio"], (list, tuple)) else [str(jp["derive_ratio"])]
        )

    # Optional derived ratio columns: NUM / DEN (strict checks; explicit behavior)
    derived_cols: list[str] = []
    if derive_ratio:
        for spec in derive_ratio:
            parts = [p.strip() for p in spec.split(":", 2)]
            if len(parts) != 3 or any(not p for p in parts):
                raise typer.BadParameter(
                    f"--derive-ratio expects '<new_col>:<numerator_col>:<denominator_col>'; got '{spec}'."
                )
            new_col, num_col, den_col = parts
            for c in (num_col, den_col):
                if c not in df.columns:
                    raise typer.BadParameter(f"--derive-ratio: column '{c}' not found.")
            try:
                num = pd.to_numeric(df[num_col], errors="raise")
                den = pd.to_numeric(df[den_col], errors="raise")
            except Exception as e:
                raise typer.BadParameter(f"--derive-ratio: numeric coercion failed: {e}")
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = num / den
            nf = ~np.isfinite(ratio.to_numpy(dtype="float64", copy=False))
            if nf.any():
                console.print(
                    f"[yellow]Note[/yellow]: derived '{new_col}' has {int(nf.sum())} non-finite value(s) "
                    f"(NaN/Inf). These rows will be skipped for numeric hues."
                )
            df[new_col] = ratio.astype(float)
            derived_cols.append(new_col)
    if df.index.name != "id":
        df = df.set_index(key_col, drop=False)
    try:
        from ..umap.plot import resolve_hue as _resolve_hue

        _resolve_hue(
            df,
            color_specs=color_by,
            name=name,
            missing_policy="drop_and_log",
            log_fn=lambda m: console.print(f"[yellow]Note[/yellow]: {m}"),
            highlight=h,
        )
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
        coords = umap_compute(X, neighbors=neighbors, min_dist=min_dist, metric=metric, seed=random_state)
        prog.update(t_umap, completed=1)
    # Prepare plot
    # Figure dims: accept "W,H" string or [W,H] list
    _dims = merged_plot.get("dims", [12, 12])
    if isinstance(_dims, str):
        W, H = [int(x) for x in _dims.split(",")]
    else:
        W, H = int(_dims[0]), int(_dims[1])
    alpha = float(merged_plot["alpha"])
    size = float(merged_plot["size"])
    font_scale = float(merged_plot.get("font_scale", 1.2))
    # legend (preset < job < CLI handled via merged_plot already)
    _legend = dict(merged_plot.get("legend", {}))
    # Preserve known keys and sanitize types; allow presets to fully control legend behavior
    legend = {}
    if "ncol" in _legend:
        legend["ncol"] = int(_legend["ncol"])
    if "bbox" in _legend:
        bbox = _legend["bbox"]
        legend["bbox"] = tuple(bbox[:2]) if isinstance(bbox, (list, tuple)) else (1.05, 1.0)
    else:
        legend["bbox"] = (1.05, 1.0)
    if "max_items" in _legend:
        legend["max_items"] = int(_legend["max_items"])
    if "frameon" in _legend:
        legend["frameon"] = bool(_legend["frameon"])

    # Normalize highlight style: deep-merge defaults -> preset.plot.highlight -> job.plot.highlight
    def _deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    highlight_style = _deep_merge(_plot_defaults.get("highlight", {}), p_plot.get("highlight", {}))
    highlight_style = _deep_merge(highlight_style, jp_plot.get("highlight", {}))
    # If the job/preset provides a categorical palette for highlight categories, pass it through.
    # Accepted forms: a mapping {category: color}, or a palette name (string) resolved in plotting.
    if "palette" in (p_plot.get("highlight", {}) or {}):
        highlight_style["palette"] = p_plot["highlight"]["palette"]
    if "highlight" in jp_plot and "palette" in (jp_plot["highlight"] or {}):
        highlight_style["palette"] = jp_plot["highlight"]["palette"]
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
    udir = umap_dir(run_dir, "<flat>")
    out_path = Path(out_plot) if out_plot else (udir / f"{name}.png")  # base; .<label>.png appended
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
        # scatter() defaults to 'drop_and_log'; we omit log_fn here to avoid duplicate logs
        overlay_highlight=True,
        highlight_style=highlight_style,
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
            "alias": name,
            "params": {
                "neighbors": neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
            },
            "sig": umap_sig,
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
            "umap_slug": "flat",
            "umap_params": {
                "neighbors": neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
            },
            "coords_path": str(udir / "coords.parquet"),
            "plot_paths": str(udir),
        }
    )
    # Persist artifacts (coords and/or derived columns) if requested
    to_attach = {"id": df[key_col].astype(str)}
    if attach_coords:
        to_attach.update(
            {
                f"cluster__{name}__umap_x": coords[:, 0],
                f"cluster__{name}__umap_y": coords[:, 1],
            }
        )
    if derived_cols:
        # For generic files, attach derived columns under their plain names (e.g., 'epistasis').
        # For USR datasets, attach under the cluster namespace to satisfy the USR API.
        if ictx["kind"] == "usr":
            for c in derived_cols:
                to_attach[f"cluster__{name}__{c}"] = df[c].astype(float).to_numpy()
        else:
            for c in derived_cols:
                to_attach[c] = df[c].astype(float).to_numpy()

    if (attach_coords or derived_cols) and write:
        cols = pd.DataFrame(to_attach)
        if ictx["kind"] == "usr":
            try:
                attach_usr(ictx["usr_root"], ictx["dataset"], cols, allow_overwrite=yes)
            except Exception as e:
                if "Columns already exist" in str(e) and not yes:
                    console.print(
                        "[red]Columns already exist[/red] for attachment. Re-run with `-y/--allow-overwrite`."
                    )
                    raise typer.Exit(code=2)
                raise
            console.print("[green]Attached[/green] columns to USR dataset.")
        else:
            full_df = load_table(ictx)
            merged = _attach_columns_schema_preserving(full_df, cols, "id", allow_overwrite=yes)
            write_generic(
                ictx["file"],
                merged,
                inplace=inplace,
                out=(Path(out) if out else None),
                backup_suffix=".bak",
            )
            console.print("[green]Wrote[/green] updated file with attachments.")
    elif (attach_coords or derived_cols) and not write:
        typer.echo("Dry-run: computed artifacts. Use --write to attach.")
        raise typer.Exit(code=0)

    # Small UX: tell users where the PNGs went and how many were rendered
    try:
        console.print(f"[green]Saved[/green] {len(color_by)} UMAP PNG(s) to {udir}")
    except Exception:
        pass

    # ---- Records sink (Markdown) ----
    try:
        run_dir = runs_root() / (name or "unnamed")
        payload = {
            "command": "umap",
            "job": job or None,
            "preset": preset or None,
            "resolved": {
                "name": name,
                "neighbors": neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state,
                "plot": {
                    "alpha": alpha,
                    "size": size,
                    "dims": [W, H],
                    "font_scale": font_scale,
                    "legend": legend,
                    "color_by": color_by,
                    "highlight": highlight_style,
                },
            },
        }
        md = f"## cluster umap — {name}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```"
        append_records_md(run_dir, md)
    except Exception:
        pass


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
    _, df = _context_and_df(dataset, file, usr_root)
    df = assert_no_duplicate_ids(df, key_col=key_col, policy="error")
    X = extract_X(
        df,
        x_col=x_col,
        x_cols=[c.strip() for c in x_cols.split(",")] if x_cols else None,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # If --seeds isn't provided (empty string), derive seeds from --replicates
    if isinstance(seeds, str) and seeds.strip():
        s = [int(x.strip()) for x in seeds.split(",")]
    else:
        s = list(range(1, int(replicates) + 1))
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
    job: Optional[str] = typer.Option(None, help="Path to a job YAML for 'analyze'."),
    cluster_col: Optional[str] = typer.Option(None, help="e.g., cluster__perm_v1"),
    group_by: str = typer.Option("source"),
    preset: Optional[str] = typer.Option(None, help="Preset name (kind: 'analysis') to pre-fill parameters"),
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
    numeric_plots: bool = typer.Option(True, help="Whether to render plots for --numeric"),
    font_scale: Optional[float] = typer.Option(None, help="Font scale for analysis plots (overrides job/preset)."),
    opal_campaign: Optional[str] = typer.Option(None, help="Optional: OPAL campaign dir or name to join metrics"),
    opal_as_of_round: Optional[int] = typer.Option(None, help="Optional: round filter for OPAL join"),
    opal_fields: Optional[str] = typer.Option(None, help="If set, join these OPAL fields before analysis"),
):
    from ..analysis.composition import composition as comp_fn
    from ..analysis.differential import differential as diff_fn
    from ..analysis.diversity import diversity as div_fn
    from ..analysis.numeric_per_cluster import summarize_numeric_by_cluster

    # Job params
    jp = _apply_job_params(job, expected_command="analyze")
    jp_plot = _apply_job_plot(job, expected_command="analyze")
    dataset = dataset or jp.get("dataset")
    file = file or jp.get("file")
    usr_root = usr_root or jp.get("usr_root")
    cluster_col = cluster_col or jp.get("cluster_col")
    if not cluster_col:
        raise typer.BadParameter(
            "Missing --cluster-col and job.params.cluster_col.\n"
            "Provide --cluster-col cluster__<NAME> or set it in the job YAML."
        )
    group_by = group_by or jp.get("group_by", "source")
    preset = preset or jp.get("preset")
    out_dir = out_dir or jp.get("out_dir")
    composition = bool(composition or jp.get("composition", False))
    diversity = bool(diversity or jp.get("diversity", False))
    difffeat = bool(difffeat or jp.get("difffeat", False))
    plots = bool(plots or jp.get("plots", False))
    if not numeric and jp.get("numeric"):
        numeric = ",".join(jp["numeric"]) if isinstance(jp["numeric"], (list, tuple)) else str(jp["numeric"])
    numeric_plots = bool(jp.get("numeric_plots", numeric_plots))
    font_scale = (
        float(jp.get("font_scale", font_scale))
        if font_scale is not None
        else (float(jp_plot.get("font_scale")) if jp_plot.get("font_scale") is not None else None)
    )
    opal_campaign = opal_campaign or jp.get("opal_campaign")
    opal_as_of_round = opal_as_of_round or jp.get("opal_as_of_round")
    if not opal_fields and jp.get("opal_fields"):
        opal_fields = (
            ",".join(jp["opal_fields"]) if isinstance(jp["opal_fields"], (list, tuple)) else str(jp["opal_fields"])
        )

    _, df = _context_and_df(dataset, file, usr_root)
    console.rule("[bold]cluster analyze[/]")

    # ---------- Apply preset (if any) ----------
    p = _apply_preset("analysis", preset)
    # Scalars/flags from preset only fill when user didn't override
    if p:
        group_by_from_preset = p.get("group_by")
        if group_by_from_preset:
            # allow either string or list in YAML
            group_bys = [group_by_from_preset] if isinstance(group_by_from_preset, str) else list(group_by_from_preset)
        else:
            group_bys = [group_by]
        composition = composition or bool(p.get("composition", False))
        diversity = diversity or bool(p.get("diversity", False))
        difffeat = difffeat or bool(p.get("difffeat", False))
        plots = plots or bool(p.get("plots", False))
        # numeric can come as a list in YAML
        if not numeric and p.get("numeric"):
            numeric = ",".join(p["numeric"]) if isinstance(p["numeric"], (list, tuple)) else str(p["numeric"])
        # font_scale + missing policy
        if font_scale is None:
            # precedence: preset.plot.font_scale -> job.plot.font_scale -> CLI
            # jp_plot already considered above; take preset only if still None
            font_scale = float(p.get("font_scale")) if p.get("font_scale") is not None else None
        # If still None, use default 1.2
        if font_scale is None:
            font_scale = 1.2
        numeric_missing_policy = str(p.get("missing_policy", "error"))
        # OPAL knobs can be provided via preset
        opal_campaign = opal_campaign or p.get("opal_campaign")
        opal_as_of_round = opal_as_of_round or p.get("opal_as_of_round")
        if not opal_fields and p.get("opal_fields"):
            opal_fields = (
                ",".join(p["opal_fields"]) if isinstance(p["opal_fields"], (list, tuple)) else str(p["opal_fields"])
            )
    else:
        group_bys = [group_by]
        numeric_missing_policy = "error"

    if font_scale is None:
        font_scale = 1.2

    # Decide *root* output directory (flattened layout; no per-group_by subdirs)
    if out_dir is None and cluster_col.startswith("cluster__"):
        fit_name = cluster_col.split("__", 1)[1]
        out_root = runs_root() / fit_name / "analysis"
    else:
        out_root = Path(out_dir or "./analysis")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- Optional OPAL join, driven by numeric metrics and/or explicit fields ----------
    # If numeric metrics include obj__/pred__/sel__ columns and they are not present,
    # auto-join them (mirrors UMAP behavior).
    needed_from_numeric: set[str] = set()
    if numeric:
        for c in [x.strip() for x in numeric.split(",") if x.strip()]:
            if c.startswith(("obj__", "pred__", "sel__")) and c not in df.columns:
                needed_from_numeric.add(c)
    explicit_fields = {f.strip() for f in opal_fields.split(",")} if opal_fields else set()
    required_fields = needed_from_numeric | explicit_fields

    if required_fields:
        if not opal_campaign:
            raise typer.BadParameter(
                "Analysis requires OPAL metrics "
                f"({', '.join(sorted(required_fields))}) but --opal-campaign is not set. "
                "Pass --opal-campaign <name|path> and optionally --opal-as-of-round <n>."
            )
        from ..opal.join import join_fields as _opal_join
        from ..opal.join import resolve_campaign_dir as _resolve_campaign_dir

        try:
            camp = _resolve_campaign_dir(opal_campaign)
        except FileNotFoundError as e:
            raise typer.BadParameter(str(e))
        df = _opal_join(
            df,
            campaign_dir=camp,
            run_selector="latest",
            fields=sorted(required_fields),
            as_of_round=opal_as_of_round,
            log_fn=lambda m: console.log(m),
        )
        # Assert coverage + warn if any nulls remain
        for c in required_fields:
            if c not in df.columns:
                raise typer.BadParameter(f"Joined OPAL field '{c}' missing after join.")
            miss = float(df[c].isna().mean())
            # Only warn when there are actual missings; keep logs compact
            if miss > 0.0:
                console.print(f"[yellow]Warning[/yellow]: joined '{c}' has {miss:.1%} missing values.")
        if required_fields:
            console.log("Joined OPAL fields: " + ", ".join(sorted(required_fields)))

    # Run numeric once (not per group_by) to avoid duplication
    if numeric:
        cols = [c.strip() for c in numeric.split(",") if c.strip()]
    else:
        cols = []
    # If the mut-count column exists, include it (coerced to numeric) for the violins
    if "permuter__mut_count" in df.columns and "permuter__mut_count" not in cols:
        cols.append("permuter__mut_count")
        summarize_numeric_by_cluster(
            df,
            cluster_col=cluster_col,
            numeric_cols=cols,
            out_dir=out_root,
            plots=numeric_plots,
            font_scale=font_scale,
            missing_policy=numeric_missing_policy,
            log_fn=lambda m: console.print(f"[yellow]Note[/yellow]: {m}"),
        )
        console.log("Numeric summaries/plots written.")

    # Run group_by‑dependent analyses; write into the *same* root with filenames namespaced by __by_<group>
    for gb in group_bys:
        if composition:
            comp_fn(df, cluster_col=cluster_col, group_by=gb, out_dir=out_root, plots=plots)
        if diversity:
            div_fn(df, cluster_col=cluster_col, group_by=gb, out_dir=out_root, plots=plots)
        if difffeat:
            diff_fn(df, cluster_col=cluster_col, group_by=gb, out_dir=out_root)
        console.log(f"Completed group_by='{gb}'.")
    console.print(f"[green]Analyses complete[/green]. Outputs at {out_root}")
    # ---- Records sink (Markdown) ----
    try:
        fit_name = cluster_col.split("__", 1)[1] if cluster_col.startswith("cluster__") else "analysis"
        run_dir = runs_root() / fit_name
        payload = {
            "command": "analyze",
            "job": job or None,
            "preset": preset or None,
            "resolved": {
                "cluster_col": cluster_col,
                "group_by": group_bys,
                "plots": bool(plots),
                "font_scale": float(font_scale),
            },
        }
        md = f"## cluster analyze — {fit_name}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```"
        append_records_md(run_dir, md)
    except Exception:
        pass


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
        console.print("[yellow]Dry-run[/yellow]: computed intra-sim but did not write. Use --write to attach.")

        raise typer.Exit(code=0)
    cols = pd.DataFrame({"id": df["id"].astype(str), out_col: s})
    if ictx["kind"] == "usr":
        try:
            attach_usr(ictx["usr_root"], ictx["dataset"], cols, allow_overwrite=yes)
        except Exception as e:
            if "Columns already exist" in str(e) and not yes:
                console.print("[red]Columns already exist[/red]. Re-run with `-y/--allow-overwrite`.")
                raise typer.Exit(code=2)
            raise
        console.print("[green]Attached[/green] intra-sim to USR dataset.")
    else:
        merged = _attach_columns_schema_preserving(df, cols, "id", allow_overwrite=yes)
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
    if not pres:
        console.print("No presets found.")
        return
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
