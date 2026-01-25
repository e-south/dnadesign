"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/viz/plotting.py

Plots:
- compression_ratio         : histogram
- tf_usage                  : usage per TF (stacked by length | TFBS | totals)
- tfbs_usage                : counts by promoter pair (skips unmapped/None)
- plan_counts               : stacked by day with UP-DN label
- tf_coverage               : 1-nt bars, solid edges, tunable palette/edges
- tfbs_length_density       : KDEs filled by default
- tfbs_positional_frequency : line plot of TFBS positional frequency
- tfbs_positional_histogram : overlaid histogram of TFBS positions

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from ..core.artifacts.pool import POOL_MODE_TFBS, TFBSPoolArtifact, load_pool_artifact
from .plot_registry import PLOT_SPECS

# Embed TrueType fonts for clean text in vector exports
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

_console = Console()
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# ---------------------- Small helpers ----------------------


def _dg(col: str) -> str:
    return col if col.startswith("densegen__") else f"densegen__{col}"


def _style(style: Optional[dict]) -> dict:
    s = (style or {}).copy()
    s.setdefault("seaborn_style", True)
    s.setdefault("despine", True)
    s.setdefault("legend_frame", False)
    s.setdefault("figsize", (8, 4))
    s.setdefault("palette", "okabe_ito")
    s.setdefault("font_size", 13)
    return s


def _apply_style(ax, style: dict):
    if style.get("seaborn_style", True):
        applied = False
        for name in ("seaborn-v0_8-ticks", "seaborn-ticks"):
            try:
                plt.style.use(name)
                applied = True
                break
            except Exception:
                continue
        if not applied:
            raise ValueError(
                "seaborn_style is true but no seaborn style is available. "
                "Install matplotlib styles that include seaborn or set seaborn_style: false."
            )
    if style.get("despine", True):
        if "top" in ax.spines:
            ax.spines["top"].set_visible(False)
        if "right" in ax.spines:
            ax.spines["right"].set_visible(False)
    fs = float(style.get("font_size", 13))
    ax.tick_params(axis="both", labelsize=float(style.get("tick_size", fs * 0.9)))
    ax.xaxis.label.set_size(float(style.get("label_size", fs)))
    ax.yaxis.label.set_size(float(style.get("label_size", fs)))
    ax.title.set_size(float(style.get("title_size", fs * 1.1)))
    lg = ax.get_legend()
    if lg is not None:
        lg.set_frame_on(bool(style.get("legend_frame", False)))


def _fig_ax(style: dict):
    w, h = style.get("figsize", (8, 4))
    return plt.subplots(figsize=(float(w), float(h)))


def _safe_filename(text: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(text).strip())
    return cleaned or "densegen"


def _plot_manifest_path(out_dir: Path) -> Path:
    return out_dir / "plot_manifest.json"


def _load_plot_manifest(out_dir: Path) -> dict:
    path = _plot_manifest_path(out_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_plot_manifest(
    out_dir: Path,
    *,
    entries: list[dict],
    run_root: Path,
    cfg_path: Path,
    source: str,
) -> None:
    existing = _load_plot_manifest(out_dir)
    merged: dict[str, dict] = {}
    for item in existing.get("plots", []):
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        if (out_dir / rel_path).exists():
            merged[rel_path] = item
    for item in entries:
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        merged[rel_path] = item
    payload = {
        "schema_version": "1.0",
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "plots": sorted(merged.values(), key=lambda x: (x.get("name", ""), x.get("path", ""))),
    }
    _plot_manifest_path(out_dir).write_text(json.dumps(payload, indent=2, sort_keys=True))


# color utils
try:
    from matplotlib.colors import is_color_like as _mpl_is_color_like
except Exception:
    _mpl_is_color_like = None


def _is_color_like(x) -> bool:
    if _mpl_is_color_like is not None:
        try:
            return bool(_mpl_is_color_like(x))
        except Exception:
            pass
    try:
        to_rgba(x)
        return True
    except Exception:
        return False


def _with_alpha(color, alpha: float):
    r, g, b, _ = to_rgba(color)
    return (r, g, b, float(alpha))


def _darker(color, factor: float = 0.85):
    r, g, b, _ = to_rgba(color)
    f = max(0.0, min(1.0, float(factor)))
    return (r * f, g * f, b * f, 1.0)


def _palette(style: dict, n: int, *, no_repeat: bool = False):
    pal = style.get("palette", "okabe_ito")
    if isinstance(pal, str):
        key = pal.lower().replace("-", "_")
        if key in {"okabe_ito", "okabeito", "colorblind"}:
            base = [
                "#000000",
                "#E69F00",
                "#56B4E9",
                "#009E73",
                "#F0E442",
                "#0072B2",
                "#D55E00",
                "#CC79A7",
            ]
            if n <= len(base):
                return base[:n]
            if no_repeat:
                raise ValueError(
                    f"Need {n} unique colors; okabe_ito has {len(base)}. Provide a longer palette or reduce categories."
                )
            return [base[i % len(base)] for i in range(n)]
        if _is_color_like(pal):  # single color
            if no_repeat and n > 1:
                raise ValueError(f"Single color '{pal}' cannot provide {n} unique colors.")
            return [pal] * n
        try:  # colormap name
            cmap = plt.get_cmap(pal)
            return [cmap(i / max(1, n - 1)) for i in range(n)]
        except Exception:
            raise ValueError(f"Unknown palette or colormap name: {pal!r}")
    if isinstance(pal, (list, tuple)):
        base = list(pal)
        if len(base) >= n:
            return base[:n]
        if no_repeat:
            raise ValueError(f"Need {n} unique colors; got {len(base)} in explicit list.")
        return [base[i % len(base)] for i in range(n)]
    raise ValueError(f"Invalid palette type: {type(pal).__name__}")


# parsing helpers
def _as_py(val):
    if hasattr(val, "as_py"):
        return val.as_py()
    return val


def _ensure_list_of_dicts(v) -> list[dict]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        raise ValueError("Expected list of dicts, got null/NaN.")
    v = _as_py(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        out = []
        for item in list(v):
            item = _as_py(item)
            if not isinstance(item, dict):
                raise ValueError("Expected list of dicts; found non-dict entries.")
            out.append(item)
        return out
    raise ValueError(f"Expected list of dicts; got {type(v).__name__}.")


def _ensure_list_of_strs(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        raise ValueError("Expected list of strings, got null/NaN.")
    v = _as_py(v)
    if isinstance(v, (list, tuple, set, np.ndarray)):
        out = [str(_as_py(x)).strip() for x in list(v)]
        if any(not s for s in out):
            raise ValueError("Expected non-empty strings; found empty value.")
        return out
    raise ValueError(f"Expected list of strings; got {type(v).__name__}.")


def _ensure_tf_counts(v) -> list[tuple[str, int]]:
    counts = []
    for item in _ensure_list_of_dicts(v):
        tf = str(item.get("tf") or "").strip()
        if not tf:
            raise ValueError("used_tf_counts entries must include non-empty 'tf'")
        count = item.get("count")
        if not isinstance(count, (int, np.integer)):
            raise ValueError("used_tf_counts entries must include integer 'count'")
        counts.append((tf, int(count)))
    return counts


# promoter scanning (top-strand only)
def _valid_dna_string(s: str) -> bool:
    return isinstance(s, str) and bool(s.strip()) and set(s.upper()) <= {"A", "C", "G", "T"}


def _scan_all_occurrences(seq: str, motif: str) -> list[int]:
    hits, i = [], seq.find(motif, 0)
    while i != -1:
        hits.append(i)
        i = seq.find(motif, i + 1)
    return hits


def _scan_motif_coverage_top_strand(seqs: Iterable[str], motifs: Iterable[str], L: int) -> np.ndarray:
    arr = np.zeros(L, dtype=float)
    motifs = [m.strip().upper() for m in motifs if _valid_dna_string(m)]
    if not motifs:
        return arr
    for s in (str(x).upper() for x in seqs if isinstance(x, str)):
        for m in motifs:
            for pos in _scan_all_occurrences(s, m):
                arr[pos : min(L, pos + len(m))] += 1.0
    return arr


def _extract_promoter_site_motifs_from_cfg(cfg: dict) -> Dict[str, set[str]]:
    """Collect upstream/downstream motifs across all plan items; returns {'35 site': {...}, '10 site': {...}}."""
    ups, dns = set(), set()
    for item in (cfg.get("generation", {}) or {}).get("plan", []) or []:
        pc = ((item or {}).get("fixed_elements", {}) or {}).get("promoter_constraints")
        pcs = [pc] if isinstance(pc, dict) else (pc or [])
        for p in (x for x in pcs if isinstance(x, dict)):
            up, dn = p.get("upstream"), p.get("downstream")
            if _valid_dna_string(str(up or "")) and str(up).lower() != "none":
                ups.add(str(up).strip().upper())
            if _valid_dna_string(str(dn or "")) and str(dn).lower() != "none":
                dns.add(str(dn).strip().upper())
    return {"35 site": ups, "10 site": dns}


def _plan_to_pair_label_map(cfg: dict) -> dict[str, str]:
    out, gen = {}, cfg.get("generation", {}) or {}
    for item in gen.get("plan", []) or []:
        nm = str(item.get("name") or "").strip()
        pc = (item.get("fixed_elements") or {}).get("promoter_constraints")
        if isinstance(pc, list) and pc:
            pc = pc[0]
        if not (nm and isinstance(pc, dict)):
            continue
        up, dn = pc.get("upstream"), pc.get("downstream")
        if _valid_dna_string(str(up or "")) and _valid_dna_string(str(dn or "")):
            out[nm] = f"{str(up).strip().upper()}–{str(dn).strip().upper()}"
    return out


def _ensure_out_dir(plots_cfg, cfg_path: Path, run_root: Path) -> Path:
    out_dir = plots_cfg.out_dir if plots_cfg else "outputs/plots"
    out = resolve_outputs_scoped_path(cfg_path, run_root, out_dir, label="plots.out_dir")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_stage_a_pools(run_root: Path) -> tuple[TFBSPoolArtifact, dict[str, pd.DataFrame]]:
    pools_dir = run_root / "outputs" / "pools"
    artifact = load_pool_artifact(pools_dir)
    pools: dict[str, pd.DataFrame] = {}
    for entry in artifact.inputs.values():
        if entry.pool_mode != POOL_MODE_TFBS:
            continue
        pool_path = pools_dir / entry.pool_path
        if not pool_path.exists():
            raise FileNotFoundError(f"Stage-A pool not found: {pool_path}")
        pools[entry.name] = pd.read_parquet(pool_path)
    if not pools:
        raise ValueError("No TFBS pools available for Stage-A plots.")
    return artifact, pools


# ---------------------- Plots ----------------------


def plot_compression_ratio(df: pd.DataFrame, out_path: Path, *, bins: int = 30, style: Optional[dict] = None) -> None:
    col = _dg("compression_ratio")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    style = _style(style)
    fig, ax = _fig_ax(style)
    ax.hist(vals, bins=bins)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Count")
    ax.set_title("DenseGen: Compression Ratio")
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tf_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    mode: str = "stack_lengths",  # stack_lengths | stack_tfbs | totals
    max_segments: Optional[int] = 20,
    top_k_tfs: Optional[int] = None,
    exclude_tfbs: Optional[Iterable[str]] = None,
    style: Optional[dict] = None,
) -> None:
    style = _style(style)
    fig, ax = _fig_ax(style)
    no_repeat = bool(style.get("palette_no_repeat", False))

    if mode == "totals":
        col = _dg("used_tf_counts")
        ser = df[col].dropna()
        counts: Dict[str, int] = {}
        for row in ser:
            for tf, n in _ensure_tf_counts(row):
                counts[tf] = counts.get(tf, 0) + int(n)
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if top_k_tfs:
            items = items[:top_k_tfs]
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        ax.bar(labels, values, color=_palette(style, len(labels), no_repeat=no_repeat))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel("Total placements")
        ax.set_title("TF usage (totals)")
        _apply_style(ax, style)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    det_col = _dg("used_tfbs_detail")
    excl = {str(x).strip().upper() for x in (exclude_tfbs or []) if str(x).strip()}

    if mode == "stack_lengths":
        by_tf_len: Dict[str, Dict[int, int]] = {}
        for row in df[det_col].dropna():
            for d in _ensure_list_of_dicts(row):
                tf = str(d.get("tf") or "").strip()
                tfbs = str(d.get("tfbs") or "").strip().upper()
                if not (tf and tfbs) or tfbs in excl:
                    continue
                by_tf_len.setdefault(tf, {}).setdefault(len(tfbs), 0)
                by_tf_len[tf][len(tfbs)] += 1
        tf_totals = sorted(
            ((tf, sum(v.values())) for tf, v in by_tf_len.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if top_k_tfs:
            tf_totals = tf_totals[:top_k_tfs]
        tf_order = [tf for tf, _ in tf_totals]
        len_global: Dict[int, int] = {}
        for v in by_tf_len.values():
            for L, n in v.items():
                len_global[L] = len_global.get(L, 0) + n
        segs_sorted = sorted(len_global.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_segments and len(segs_sorted) > max_segments:
            chosen = [L for L, _ in segs_sorted[:max_segments]]
            remaining = {L for L, _ in segs_sorted[max_segments:]}
            segments = chosen + ["OTHER"]
        else:
            remaining, segments = set(), [L for L, _ in segs_sorted]
        x = np.arange(len(tf_order))
        bottom = np.zeros_like(x, dtype=float)
        # Sequential colormap by increasing motif length; OTHER → neutral gray
        length_bins = [int(s) for s in segments if s != "OTHER"]
        length_bins_sorted = sorted(set(length_bins))
        cmap_name = style.get("length_cmap", "cividis")
        cmap = plt.get_cmap(cmap_name)

        def _seq_color(i, n):
            return cmap(i / max(1, n - 1))

        seq_colors = {L: _seq_color(i, len(length_bins_sorted)) for i, L in enumerate(length_bins_sorted)}
        seg_to_color = {seg: (seq_colors[int(seg)] if seg != "OTHER" else "#B0B0B0") for seg in segments}
        for seg in segments:
            heights = []
            for tf in tf_order:
                val = (
                    sum(n for L, n in by_tf_len.get(tf, {}).items() if L in remaining)
                    if seg == "OTHER"
                    else int(by_tf_len.get(tf, {}).get(int(seg), 0))
                )
                heights.append(float(val))
            ax.bar(
                x,
                heights,
                bottom=bottom,
                color=seg_to_color[seg],
                linewidth=0,
                label=f"len={seg}" if seg != "OTHER" else "OTHER",
            )
            bottom += np.array(heights, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels(tf_order)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel("Occurrences")
        ax.set_title("TF usage (stacked by TFBS length)")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=bool(style.get("legend_frame", False)),
        )
        _apply_style(ax, style)
        fig.tight_layout(rect=[0, 0, 0.82, 1.0])
        fig.savefig(out_path)
        plt.close(fig)
        return

    if mode == "stack_tfbs":
        by_tf_tfbs: Dict[str, Dict[str, int]] = {}
        for row in df[det_col].dropna():
            for d in _ensure_list_of_dicts(row):
                tf = str(d.get("tf") or "").strip()
                tfbs = str(d.get("tfbs") or "").strip().upper()
                if not (tf and tfbs) or tfbs in excl:
                    continue
                by_tf_tfbs.setdefault(tf, {}).setdefault(tfbs, 0)
                by_tf_tfbs[tf][tfbs] += 1
        tf_totals = sorted(
            ((tf, sum(v.values())) for tf, v in by_tf_tfbs.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if top_k_tfs:
            tf_totals = tf_totals[:top_k_tfs]
        tf_order = [tf for tf, _ in tf_totals]
        tfbs_global: Dict[str, int] = {}
        for v in by_tf_tfbs.values():
            for s, n in v.items():
                tfbs_global[s] = tfbs_global.get(s, 0) + n
        segs_sorted = sorted(tfbs_global.items(), key=lambda kv: kv[1], reverse=True)
        if max_segments and len(segs_sorted) > max_segments:
            chosen = [s for s, _ in segs_sorted[:max_segments]]
            remaining = {s for s, _ in segs_sorted[max_segments:]}
            segments = chosen + ["OTHER"]
        else:
            remaining, segments = set(), [s for s, _ in segs_sorted]
        x = np.arange(len(tf_order))
        bottom = np.zeros_like(x, dtype=float)
        colors = _palette(style, len(segments), no_repeat=no_repeat)
        seg_to_color = {segments[i]: colors[i] for i in range(len(segments))}
        for seg in segments:
            heights = []
            for tf in tf_order:
                inner = by_tf_tfbs.get(tf, {})
                val = sum(n for s, n in inner.items() if s in remaining) if seg == "OTHER" else int(inner.get(seg, 0))
                heights.append(float(val))
            ax.bar(
                x,
                heights,
                bottom=bottom,
                color=seg_to_color[seg],
                linewidth=0,
                label=seg,
            )
            bottom += np.array(heights, dtype=float)
        ax.set_xticks(x)
        ax.set_xticklabels(tf_order)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel("Occurrences")
        ax.set_title("TF usage (stacked by TFBS)")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=bool(style.get("legend_frame", False)),
        )
        _apply_style(ax, style)
        fig.tight_layout(rect=[0, 0, 0.82, 1.0])
        fig.savefig(out_path)
        plt.close(fig)
        return

    raise ValueError("tf_usage.mode must be one of: stack_lengths, stack_tfbs, totals")


def plot_pad_gc(df: pd.DataFrame, out_path: Path, *, style: Optional[dict] = None) -> None:
    used_col, gc_col, b_col = (
        _dg("pad_used"),
        _dg("pad_gc_actual"),
        _dg("pad_bases"),
    )
    mask = df[used_col] == True  # noqa: E712
    x = pd.to_numeric(df.loc[mask, gc_col], errors="coerce")
    y = pd.to_numeric(df.loc[mask, b_col], errors="coerce")
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    style = _style(style)
    fig, ax = _fig_ax(style)
    ax.scatter(x.values, y.values, alpha=0.35, s=12)
    ax.set_xlabel("Pad GC fraction")
    ax.set_ylabel("Pad bases")
    ax.set_title("Pad: bases vs GC fraction (padded only)")
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_plan_counts(
    df: pd.DataFrame,
    out_path: Path,
    *,
    style: Optional[dict] = None,
    stacked_by_day: bool = True,
    created_at_col: str = "densegen__created_at",
    cfg: Optional[dict] = None,
) -> None:
    plan_col = _dg("plan")
    if cfg is None:
        raise ValueError("plan_counts requires cfg for UP–DN labels.")
    df = df[df[plan_col].notna()].copy()
    df = df[df[plan_col].astype(str).str.strip().str.lower() != "none"].copy()
    style = _style(style)
    fig, ax = _fig_ax(style)
    pmap = _plan_to_pair_label_map(cfg)

    df_local = df[[plan_col] + ([created_at_col] if stacked_by_day else [])].copy()
    df_local["_plan"] = df_local[plan_col].astype(str)

    if stacked_by_day:
        dt = pd.to_datetime(df_local[created_at_col], errors="coerce")
        df_local = df_local[dt.notna()].copy()
        df_local["_day"] = dt[dt.notna()].dt.floor("D")
        pivot = df_local.groupby(["_plan", "_day"]).size().unstack(fill_value=0).sort_index(axis=1)
        days = list(pivot.columns)
        x = np.arange(len(pivot.index))
        bottom = np.zeros(len(x), dtype=float)
        cols = _palette(style, len(days))
        for i, d in enumerate(days):
            vals = pivot[d].to_numpy(dtype=float)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=cols[i],
                linewidth=0,
                label=str(getattr(d, "date", lambda: d)()),
            )
            bottom += vals
        ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
        labels = pivot.index.tolist()
    else:
        counts = df_local["_plan"].value_counts()
        x = np.arange(len(counts.index))
        ax.bar(x, counts.values, linewidth=0)
        labels = counts.index.tolist()

    xticks = [f"{p}\n{pmap[p]}" if p in pmap and pmap[p] else p for p in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_ylabel("Appearances")
    ax.set_title("Counts per plan (σ70 UP–DN; stacked by day)")
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tf_coverage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    top_k: int = 20,
    normalize: bool = True,
    stacked: bool = False,
    include_promoter_sites: bool = True,  # default on
    cfg: Optional[dict] = None,  # required when include_promoter_sites=True
    alpha: float = 0.35,
    edge_alpha: float = 0.85,
    edge_width: float = 0.6,
    edge_color: Optional[str] = "auto",
    promoter_edge_alpha: Optional[float] = None,
    promoter_edge_style: str = "solid",
    promoter_on_top: bool = True,
    promoter_zorder: float = 4.0,
    edge_on: bool = True,
    edge_dark_factor: float = 0.80,
    style: Optional[dict] = None,
) -> None:
    """
    Adds two distinct categories scanned from config:
    - '35 site'  (upstream)
    - '10 site'  (downstream)
    Top-strand exact matches only (no reverse complement).
    """
    det_col = _dg("used_tfbs_detail")
    if "length" in df.columns:
        L = int(pd.to_numeric(df["length"], errors="coerce").dropna().max())
    elif _dg("sequence_length") in df.columns:
        L = int(pd.to_numeric(df[_dg("sequence_length")], errors="coerce").dropna().max())
    elif "sequence" in df.columns:
        L = int(df["sequence"].astype(str).map(len).max())
    else:
        raise ValueError("Cannot infer sequence length for tf_coverage.")
    details = df[det_col].dropna()

    # accumulate coverage per TF
    coverages: Dict[str, np.ndarray] = {}
    n_seqs = len(details)
    for row in details:
        for d in _ensure_list_of_dicts(row):
            tf = str(d.get("tf") or "").strip()
            tfbs = str(d.get("tfbs") or "")
            if not tf or not tfbs:
                continue
            start = int(float(d.get("offset", 0)))
            span = len(tfbs)
            if span <= 0:
                continue
            start = max(0, min(start, L))
            end = min(L, start + span)
            if end <= start:
                continue
            coverages.setdefault(tf, np.zeros(L, dtype=float))[start:end] += 1.0

    # upstream/downstream overlays from cfg (top strand only)
    if include_promoter_sites:
        if cfg is None:
            raise ValueError("tf_coverage(include_promoter_sites=True) requires cfg.")
        if "sequence" not in df.columns:
            raise ValueError("tf_coverage(include_promoter_sites=True) requires 'sequence'.")
        seqs = df["sequence"].astype(str).tolist()
        prom = _extract_promoter_site_motifs_from_cfg(cfg)  # {'35 site': {...}, '10 site': {...}}
        for label, motif_set in prom.items():
            if motif_set:
                arr = _scan_motif_coverage_top_strand(seqs, motif_set, L)
                coverages[label] = coverages.get(label, np.zeros(L, dtype=float)) + arr
        n_seqs = max(n_seqs, len(seqs))

    if not coverages:
        raise ValueError("No coverage tracks available.")

    # top_k by total coverage
    order = sorted(coverages.items(), key=lambda kv: kv[1].sum(), reverse=True)[:top_k]
    names = [k for k, _ in order]
    style = _style(style)
    fig, ax = _fig_ax(style)

    # base colors, then override promoter categories
    colors = _palette(style, len(order))
    name_to_color = {names[i]: colors[i] for i in range(len(names))}
    default_promoter_colors = {"35 site": "#D55E00", "10 site": "#0072B2"}

    def _prom_color(label: str, default: str):
        pc = style.get("promoter_colors") or {}
        if label in pc:
            return pc[label]
        return default

    for key, default in default_promoter_colors.items():
        if key in name_to_color:
            name_to_color[key] = _prom_color(key, default)

    xs = np.arange(L)
    base = np.zeros(L, dtype=float)
    prom_keys = {"35 site", "10 site"}
    main = [(k, v) for k, v in order if k not in prom_keys]
    prom = [(k, v) for k, v in order if k in prom_keys]
    if not promoter_on_top:
        main, prom = (prom + main, [])

    lw = float(edge_width) if (edge_on and edge_width and edge_alpha) else 0.0

    def _edge_rgba(face_color, a):
        if lw == 0.0:
            return "none"
        base_c = (
            _darker(face_color, float(edge_dark_factor))
            if (edge_color is None or str(edge_color).lower() == "auto")
            else edge_color
        )
        return _with_alpha(base_c, float(a))

    # main bars
    for tf, arr in main:
        y = arr.astype(float)
        if normalize and n_seqs > 0:
            y = y / float(n_seqs)
        face = name_to_color[tf]
        ax.bar(
            xs,
            y,
            bottom=(base if stacked else None),
            width=1.0,
            align="edge",
            color=_with_alpha(face, alpha),
            edgecolor=_edge_rgba(face, edge_alpha),
            linewidth=lw,
            linestyle="solid",
            label=tf,
            zorder=2.0,
        )
        if stacked:
            base += y

    # promoter overlays
    if prom:
        pa = max(edge_alpha, 0.95) if promoter_edge_alpha is None else float(promoter_edge_alpha)
        for tf, arr in prom:
            y = arr.astype(float)
            if normalize and n_seqs > 0:
                y = y / float(n_seqs)
            face = name_to_color[tf]
            ax.bar(
                xs,
                y,
                bottom=(base if stacked else None),
                width=1.0,
                align="edge",
                color=_with_alpha(face, alpha),
                edgecolor=_edge_rgba(face, pa),
                linewidth=lw,
                linestyle=promoter_edge_style,
                label=tf,
                zorder=float(promoter_zorder),
            )
            if stacked:
                base += y

    ax.set_xlabel("Nucleotide Position")
    ax.set_ylabel("Coverage" + (" (fraction)" if normalize else " (count)"))
    ax.set_title("TFBS coverage along the sequence")
    ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _kde_gaussian(x: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.zeros_like(grid)
    std = np.std(x) or 1.0
    if not bandwidth or not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.06 * std * (n ** (-1 / 5))
        if bandwidth <= 0:
            bandwidth = max(0.1, std * 0.2)
    z = (grid[:, None] - x[None, :]) / bandwidth
    dens = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return dens.sum(axis=1) / (n * bandwidth)


def _ensure_list(val) -> list:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return list(parsed)
            except Exception:
                return []
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return []


def plot_tfbs_length_density(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: Optional[pd.DataFrame] = None,
    bins: int | str = "auto",
    kde: bool = True,
    kde_bandwidth: Optional[float] = None,
    kde_points: int = 256,
    fill_alpha: float = 0.35,
    include_promoter_sites: bool = False,  # retained for forward-compat; no RC scanning here
    promoter_site_motifs: Optional[Dict[str, set[str]]] = None,  # optional explicit overlay sets
    style: Optional[dict] = None,
) -> None:
    by_tf: Dict[str, list[int]] = {}
    if attempts_df is None or attempts_df.empty:
        raise ValueError(
            "outputs/tables/attempts.parquet is required for tfbs_length_density. "
            "Run `dense run -c <config.yaml>` to generate attempts."
        )
    if attempts_df is not None and not attempts_df.empty:
        seen_libs: set[tuple[str, int]] = set()
        for _, row in attempts_df.iterrows():
            lib_hash = str(row.get("sampling_library_hash") or "")
            lib_idx = int(row.get("sampling_library_index") or 0)
            key = (lib_hash, lib_idx)
            if key in seen_libs:
                continue
            seen_libs.add(key)
            tfbs_list = _ensure_list(row.get("library_tfbs"))
            tf_list = _ensure_list(row.get("library_tfs"))
            for idx, tfbs in enumerate(tfbs_list):
                tf = str(tf_list[idx]) if idx < len(tf_list) else ""
                if tf and tfbs:
                    by_tf.setdefault(tf, []).append(len(str(tfbs)))
    # Optional: if explicit promoter_site_motifs provided, include their motif lengths
    if include_promoter_sites and promoter_site_motifs:
        for label, motif_set in (promoter_site_motifs or {}).items():
            lengths: list[int] = []
            for m in motif_set or []:
                m = str(m).strip().upper()
                if not _valid_dna_string(m):
                    continue
                lengths.append(len(m))
            if lengths:
                by_tf.setdefault(label, []).extend(lengths)

    if not by_tf:
        raise ValueError("No TFBS lengths available.")
    style = _style(style)
    fig, ax = _fig_ax(style)
    tfs = sorted(by_tf.keys())
    cols = _palette(style, len(tfs))
    for tf, color in zip(tfs, cols):
        vals = np.array(by_tf[tf], dtype=float)
        if vals.size == 0:
            continue
        if kde:
            xmin, xmax = float(vals.min()), float(vals.max())
            pad = max(1.0, 0.05 * (xmax - xmin if xmax > xmin else 2.0))
            grid = np.linspace(xmin - pad, xmax + pad, max(64, int(kde_points)))
            dens = _kde_gaussian(vals, grid, bandwidth=kde_bandwidth)
            if float(fill_alpha) > 0:
                ax.fill_between(
                    grid,
                    0,
                    dens,
                    color=color,
                    alpha=float(fill_alpha),
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(grid, dens, linewidth=2, color=color, label=tf, zorder=2)
        else:
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=tf,
            )
    ax.set_xlabel("Motif length (nt)")
    ax.set_ylabel("Density")
    ax.set_title("Library TFBS length distribution")
    ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tfbs_positional_frequency(
    df: pd.DataFrame,
    out_path: Path,
    *,
    top_k: int = 10,
    normalize: bool = True,
    include_promoter_sites: bool = True,
    cfg: Optional[dict] = None,
    style: Optional[dict] = None,
) -> None:
    det_col = _dg("used_tfbs_detail")
    if "length" in df.columns:
        L = int(pd.to_numeric(df["length"], errors="coerce").dropna().max())
    elif _dg("sequence_length") in df.columns:
        L = int(pd.to_numeric(df[_dg("sequence_length")], errors="coerce").dropna().max())
    elif "sequence" in df.columns:
        L = int(df["sequence"].astype(str).map(len).max())
    else:
        raise ValueError("Cannot infer sequence length for tfbs_positional_frequency.")
    details = df[det_col].dropna()
    coverages: Dict[str, np.ndarray] = {}
    n_seqs = len(details)
    for row in details:
        for d in _ensure_list_of_dicts(row):
            tf = str(d.get("tf") or "").strip()
            tfbs = str(d.get("tfbs") or "")
            if not tf or not tfbs:
                continue
            start = int(float(d.get("offset", 0)))
            span = len(tfbs)
            if span <= 0:
                continue
            start = max(0, min(start, L))
            end = min(L, start + span)
            if end <= start:
                continue
            coverages.setdefault(tf, np.zeros(L, dtype=float))[start:end] += 1.0

    if include_promoter_sites:
        if cfg is None:
            raise ValueError("tfbs_positional_frequency(include_promoter_sites=True) requires cfg.")
        if "sequence" not in df.columns:
            raise ValueError("tfbs_positional_frequency(include_promoter_sites=True) requires 'sequence'.")
        seqs = df["sequence"].astype(str).tolist()
        prom = _extract_promoter_site_motifs_from_cfg(cfg)
        for label, motif_set in prom.items():
            if motif_set:
                arr = _scan_motif_coverage_top_strand(seqs, motif_set, L)
                coverages[label] = coverages.get(label, np.zeros(L, dtype=float)) + arr
        n_seqs = max(n_seqs, len(seqs))

    if not coverages:
        raise ValueError("No positional TFBS coverage available.")

    order = sorted(coverages.items(), key=lambda kv: kv[1].sum(), reverse=True)[: max(1, int(top_k))]
    style = _style(style)
    fig, ax = _fig_ax(style)
    colors = _palette(style, len(order))
    xs = np.arange(L)
    for (tf, arr), color in zip(order, colors):
        y = arr.astype(float)
        if normalize and n_seqs > 0:
            y = y / float(n_seqs)
        ax.plot(xs, y, linewidth=2.0, label=tf, color=color)
    ax.set_xlabel("Position (nt)")
    ax.set_ylabel("Frequency" + (" (normalized)" if normalize else ""))
    ax.set_title("TFBS positional frequency")
    ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tfbs_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    max_sites: Optional[int] = None,  # applies to TFBS strings (pairs always included)
    couple_sigma_pairs: bool = True,  # when True -> combined: pairs + TFBS strings
    exclude_tfbs: Optional[Iterable[str]] = None,
    cfg: Optional[dict] = None,
    style: Optional[dict] = None,
) -> None:
    """
    Plot only unique TFBS strings (ranked).
    Colors: TFBS bars are colored by dominant TF. The strings ACCGCG and TATAAT
    are excluded by default.
    """
    style = _style(style)
    fig, ax = _fig_ax(style)
    used_col, det_col = _dg("used_tfbs"), _dg("used_tfbs_detail")
    excl: set[str] = set(str(x).strip().upper() for x in (exclude_tfbs or []) if str(x).strip())
    if exclude_tfbs is None:
        excl.add("G")  # default drop lone 'G'
    # Always exclude promoter elements from this plot
    excl.update({"ACCGCG", "TATAAT"})

    if couple_sigma_pairs:
        # Show only unique TFBS strings (no UP–DN pair bars)
        counts_tfbs: Dict[str, int] = {}
        tf_for_tfbs: Dict[str, str] = {}
        if used_col in df.columns:
            for row in df[used_col].dropna():
                for s in _ensure_list_of_strs(row):
                    tf = s.split(":", 1)[0].strip() if ":" in s else ""
                    tfbs = (s.split(":", 1)[1] if ":" in s else s).strip().upper()
                    if not tfbs or tfbs in excl:
                        continue
                    counts_tfbs[tfbs] = counts_tfbs.get(tfbs, 0) + 1
                    if tf and tfbs not in tf_for_tfbs:
                        tf_for_tfbs[tfbs] = tf
        elif det_col in df.columns:
            for row in df[det_col].dropna():
                for d in _ensure_list_of_dicts(row):
                    tf = str(d.get("tf") or "").strip()
                    tfbs = str(d.get("tfbs") or "").strip().upper()
                    if not tfbs or tfbs in excl:
                        continue
                    counts_tfbs[tfbs] = counts_tfbs.get(tfbs, 0) + 1
                    if tf and tfbs not in tf_for_tfbs:
                        tf_for_tfbs[tfbs] = tf
        else:
            raise KeyError("missing TFBS columns")

        if not counts_tfbs:
            raise ValueError("No TFBS usage after filtering.")
        ranked_tfbs = sorted(counts_tfbs.items(), key=lambda kv: kv[1], reverse=True)
        if max_sites:
            ranked_tfbs = ranked_tfbs[:max_sites]
        labels = [s for s, _ in ranked_tfbs]
        values = [v for _, v in ranked_tfbs]
        present_tfs = sorted({tf_for_tfbs.get(s, "") for s in labels if tf_for_tfbs.get(s, "")})
        tf_colors = {tf: c for tf, c in zip(present_tfs, _palette(style, len(present_tfs)))}
        colors = [tf_colors.get(tf_for_tfbs.get(s, ""), "#BBBBBB") for s in labels]
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelrotation=90)
        if present_tfs:
            ax.legend(
                handles=[Patch(facecolor=tf_colors[tf], label=tf) for tf in present_tfs],
                loc="best",
                frameon=bool(style.get("legend_frame", False)),
            )
        ax.set_ylabel("Occurrences")
        ax.set_title("TFBS usage: unique binding sites (pairs excluded)")
        _apply_style(ax, style)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    # If couple_sigma_pairs=False -> classic per-TFBS ranking colored by TF
    counts: Dict[str, int] = {}
    tf_for_tfbs: Dict[str, str] = {}
    if used_col in df.columns:
        for row in df[used_col].dropna():
            for s in _ensure_list_of_strs(row):
                tf = s.split(":", 1)[0].strip() if ":" in s else ""
                tfbs = (s.split(":", 1)[1] if ":" in s else s).strip().upper()
                if not tfbs or tfbs in excl:
                    continue
                counts[tfbs] = counts.get(tfbs, 0) + 1
                if tf:
                    tf_for_tfbs.setdefault(tfbs, tf)
    elif det_col in df.columns:
        for row in df[det_col].dropna():
            for d in _ensure_list_of_dicts(row):
                tf = str(d.get("tf") or "").strip()
                tfbs = str(d.get("tfbs") or "").strip().upper()
                if not tfbs or tfbs in excl:
                    continue
                counts[tfbs] = counts.get(tfbs, 0) + 1
                if tf:
                    tf_for_tfbs.setdefault(tfbs, tf)
    else:
        raise KeyError("missing TFBS columns")
    if not counts:
        raise ValueError("No TFBS usage after filtering.")
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if max_sites:
        ranked = ranked[:max_sites]
    labels = [k for k, _ in ranked]
    values = [v for _, v in ranked]
    present_tfs = sorted({tf_for_tfbs.get(b, "") for b in labels if tf_for_tfbs.get(b, "")})
    tf_colors = {tf: c for tf, c in zip(present_tfs, _palette(style, len(present_tfs)))}
    bar_colors = [tf_colors.get(tf_for_tfbs.get(b, ""), "#BBBBBB") for b in labels]
    ax.bar(labels, values, color=bar_colors)
    if present_tfs:
        ax.legend(
            handles=[Patch(facecolor=tf_colors[tf], label=tf) for tf in present_tfs],
            loc="best",
            frameon=bool(style.get("legend_frame", False)),
        )
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylabel("Occurrences")
    ax.set_title("TFBS usage by TF (ranked)")
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _normalized_entropy(counts: dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for val in counts.values():
        p = float(val) / total
        if p <= 0:
            continue
        ent -= p * np.log(p)
    return float(ent / np.log(len(counts)))


def plot_diversity_health(
    df: pd.DataFrame,
    out_path: Path,
    *,
    created_at_col: Optional[str] = None,
    normalize: bool = True,
    show_unique_tfs: bool = True,
    show_unique_tfbs: bool = True,
    show_entropy: bool = True,
    style: Optional[dict] = None,
) -> None:
    det_col = _dg("used_tfbs_detail")
    if det_col not in df.columns:
        raise ValueError("diversity_health requires used_tfbs_detail.")
    created_col = created_at_col or _dg("created_at")
    ordered = df
    if created_col in df.columns:
        ordered = df.sort_values(created_col, kind="mergesort")

    counts_tf: dict[str, int] = {}
    counts_tfbs: dict[str, int] = {}
    uniq_tf: list[float] = []
    uniq_tfbs: list[float] = []
    entropy: list[float] = []

    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, float):
            return bool(np.isnan(value))
        return False

    for row in ordered[det_col]:
        if _is_missing(row):
            continue
        for entry in _ensure_list_of_dicts(row):
            tf = str(entry.get("tf") or "").strip()
            tfbs = str(entry.get("tfbs") or "").strip()
            if tf:
                counts_tf[tf] = counts_tf.get(tf, 0) + 1
            if tfbs:
                counts_tfbs[tfbs] = counts_tfbs.get(tfbs, 0) + 1
        uniq_tf.append(float(len(counts_tf)))
        uniq_tfbs.append(float(len(counts_tfbs)))
        entropy.append(_normalized_entropy(counts_tfbs))

    if not uniq_tfbs:
        raise ValueError("No TFBS placements found for diversity_health.")

    if normalize:
        tf_denom = max(1.0, uniq_tf[-1] if uniq_tf else 1.0)
        tfbs_denom = max(1.0, uniq_tfbs[-1])
        uniq_tf = [val / tf_denom for val in uniq_tf]
        uniq_tfbs = [val / tfbs_denom for val in uniq_tfbs]

    style = _style(style)
    fig, ax = _fig_ax(style)
    x = np.arange(1, len(uniq_tfbs) + 1)
    colors = _palette(style, 3)
    handles = []
    if show_unique_tfs:
        handles.append(ax.plot(x, uniq_tf, label="unique TFs", color=colors[0], linewidth=2)[0])
    if show_unique_tfbs:
        handles.append(ax.plot(x, uniq_tfbs, label="unique TFBS", color=colors[1], linewidth=2)[0])
    ax.set_xlabel("Sequence index")
    ax.set_ylabel("Coverage (normalized)" if normalize else "Unique count")
    ax.set_title("Diversity health")

    if show_entropy:
        ax2 = ax.twinx()
        handles.append(
            ax2.plot(
                x,
                entropy,
                label="TFBS entropy",
                color=colors[2],
                linewidth=2,
                linestyle="--",
            )[0]
        )
        ax2.set_ylabel("Normalized entropy")
    ax.legend(handles=handles, loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tfbs_positional_histogram(
    df: pd.DataFrame,
    out_path: Path,
    *,
    top_k: int = 10,
    normalize: bool = True,
    include_promoter_sites: bool = True,
    alpha: float = 0.35,
    edge_alpha: float = 0.8,
    edge_width: float = 0.5,
    cfg: Optional[dict] = None,
    style: Optional[dict] = None,
) -> None:
    det_col = _dg("used_tfbs_detail")
    if "length" in df.columns:
        L = int(pd.to_numeric(df["length"], errors="coerce").dropna().max())
    elif _dg("sequence_length") in df.columns:
        L = int(pd.to_numeric(df[_dg("sequence_length")], errors="coerce").dropna().max())
    elif "sequence" in df.columns:
        L = int(df["sequence"].astype(str).map(len).max())
    else:
        raise ValueError("Cannot infer sequence length for tfbs_positional_histogram.")
    details = df[det_col].dropna()
    coverages: Dict[str, np.ndarray] = {}
    n_seqs = len(details)
    for row in details:
        for d in _ensure_list_of_dicts(row):
            tf = str(d.get("tf") or "").strip()
            tfbs = str(d.get("tfbs") or "").strip()
            label = tf or tfbs
            if not label:
                continue
            start = int(float(d.get("offset", 0)))
            span = int(d.get("length", len(tfbs)))
            if span <= 0:
                continue
            start = max(0, min(start, L))
            end = min(L, start + span)
            if end <= start:
                continue
            coverages.setdefault(label, np.zeros(L, dtype=float))[start:end] += 1.0

    if include_promoter_sites:
        if cfg is None:
            raise ValueError("tfbs_positional_histogram(include_promoter_sites=True) requires cfg.")
        if "sequence" not in df.columns:
            raise ValueError("tfbs_positional_histogram(include_promoter_sites=True) requires 'sequence'.")
        seqs = df["sequence"].astype(str).tolist()
        prom = _extract_promoter_site_motifs_from_cfg(cfg)
        for label, motif_set in prom.items():
            if motif_set:
                arr = _scan_motif_coverage_top_strand(seqs, motif_set, L)
                coverages[label] = coverages.get(label, np.zeros(L, dtype=float)) + arr
        n_seqs = max(n_seqs, len(seqs))

    if not coverages:
        raise ValueError("No positional TFBS coverage available.")

    order = sorted(coverages.items(), key=lambda kv: kv[1].sum(), reverse=True)[: max(1, int(top_k))]
    style = _style(style)
    fig, ax = _fig_ax(style)
    colors = _palette(style, len(order))
    xs = np.arange(L)
    for (label, arr), color in zip(order, colors):
        y = arr.astype(float)
        if normalize and n_seqs > 0:
            y = y / float(n_seqs)
        edge = _with_alpha(_darker(color), float(edge_alpha))
        ax.bar(
            xs,
            y,
            width=1.0,
            alpha=float(alpha),
            color=color,
            edgecolor=edge,
            linewidth=float(edge_width),
            label=label,
        )
    ax.set_xlabel("Position (nt)")
    ax.set_ylabel("Frequency" + (" (normalized)" if normalize else ""))
    ax.set_title("TFBS positional histogram")
    ax.legend(loc="best", frameon=bool(style.get("legend_frame", False)))
    _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stage_a_strata_overview(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if pools is None or pool_manifest is None:
        raise ValueError("Stage-A plots require pool manifests; run stage-a build-pool first.")
    raw_style = style or {}
    style = _style(raw_style)
    if "figsize" not in raw_style:
        style["figsize"] = (11, 4)
    paths: list[Path] = []
    for input_name, pool_df in pools.items():
        entry = pool_manifest.entry_for(input_name)
        sampling = entry.stage_a_sampling
        if sampling is None:
            raise ValueError(f"Stage-A sampling metadata missing for input '{input_name}'.")
        if sampling.get("backend") != "fimo":
            raise ValueError(f"Stage-A strata overview requires FIMO sampling (input '{input_name}').")
        pvalue_strata = sampling.get("pvalue_strata") or []
        retain_depth = sampling.get("retain_depth")
        eligible_bins = sampling.get("eligible_bins") or []
        retained_bins = sampling.get("retained_bins") or []
        if not pvalue_strata:
            raise ValueError(f"Stage-A sampling missing pvalue_strata for input '{input_name}'.")
        if retain_depth is None:
            raise ValueError(f"Stage-A sampling missing retain_depth for input '{input_name}'.")
        if not eligible_bins:
            raise ValueError(f"Stage-A sampling missing eligible bin counts for input '{input_name}'.")
        if not retained_bins:
            raise ValueError(f"Stage-A sampling missing retained bin counts for input '{input_name}'.")
        if int(retain_depth) < 1 or int(retain_depth) > len(pvalue_strata):
            raise ValueError(f"Stage-A sampling retain_depth out of range for input '{input_name}'.")

        regulators = [str(row.get("regulator")) for row in eligible_bins]
        eligible_by_reg = {str(row.get("regulator")): list(row.get("counts") or []) for row in eligible_bins}
        retained_by_reg = {str(row.get("regulator")): list(row.get("counts") or []) for row in retained_bins}
        for reg, counts in eligible_by_reg.items():
            if len(counts) != len(pvalue_strata):
                raise ValueError(f"Eligible bin counts length mismatch for '{input_name}' ({reg}).")
        for reg, counts in retained_by_reg.items():
            if len(counts) != len(pvalue_strata):
                raise ValueError(f"Retained bin counts length mismatch for '{input_name}' ({reg}).")

        colors = _palette(style, len(regulators), no_repeat=bool(style.get("palette_no_repeat", False)))
        color_by_reg = {reg: colors[idx] for idx, reg in enumerate(regulators)}

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=style["figsize"])

        xs = [float(edge) for edge in pvalue_strata]
        for reg in regulators:
            counts = eligible_by_reg.get(reg, [0 for _ in pvalue_strata])
            ax_left.plot(xs, counts, marker="o", color=color_by_reg[reg], label=reg)
        ax_left.set_xscale("log")
        ax_left.invert_xaxis()
        ax_left.set_xticks(xs)
        ax_left.set_xticklabels([f"{edge:.0e}" for edge in pvalue_strata])
        ax_left.set_xlabel("p-value (lower is better)")
        ax_left.set_ylabel("eligible count")
        retain_edge = float(pvalue_strata[int(retain_depth) - 1])
        ax_left.axvline(retain_edge, linestyle="--", color="black", linewidth=1)
        ax_left.legend(loc="best", frameon=bool(style.get("legend_frame", False)))

        lengths_by_reg: dict[str, list[int]] = {}
        if "tf" not in pool_df.columns or "tfbs" not in pool_df.columns:
            raise ValueError(f"Stage-A pool missing tf/tfbs columns for input '{input_name}'.")
        for reg, seq in pool_df[["tf", "tfbs"]].itertuples(index=False):
            reg_label = str(reg)
            lengths_by_reg.setdefault(reg_label, []).append(len(str(seq)))
        all_lengths = [val for vals in lengths_by_reg.values() for val in vals]
        if not all_lengths:
            ax_right.text(0.5, 0.5, "No retained sequences", ha="center", va="center", transform=ax_right.transAxes)
        else:
            min_len = min(all_lengths)
            max_len = max(all_lengths)
            bins = np.arange(min_len, max_len + 2) - 0.5
            for reg, lengths in lengths_by_reg.items():
                if not lengths:
                    continue
                ax_right.hist(
                    lengths,
                    bins=bins,
                    color=color_by_reg.get(reg, "#4c78a8"),
                    alpha=0.6,
                    edgecolor="white",
                    label=reg,
                )
            ax_right.set_xticks(list(range(min_len, max_len + 1)))
        ax_right.set_xlabel("TFBS length (nt)")
        ax_right.set_ylabel("retained count")

        fig.suptitle(f"Stage-A strata overview: {input_name}")
        _apply_style(ax_left, style)
        _apply_style(ax_right, style)
        fig.tight_layout()
        fname = f"{out_path.stem}__{_safe_filename(input_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


AVAILABLE_PLOTS: Dict[str, Dict[str, object]] = {}
for _name, _spec in PLOT_SPECS.items():
    _fn_name = _spec.get("fn")
    _fn = globals().get(str(_fn_name))
    if _fn is None:
        raise RuntimeError(f"Plot function '{_fn_name}' not found for '{_name}'.")
    AVAILABLE_PLOTS[_name] = {
        "fn": _fn,
        "description": _spec.get("description", ""),
        "requires": _spec.get("requires"),
    }


# ---------------------- Runner with unknown-option filter ----------------------

# Options explicitly supported by each plot; unknown options raise errors (strict).
_ALLOWED_OPTIONS = {
    "compression_ratio": {"bins"},
    "tf_usage": {"mode", "max_segments", "top_k_tfs", "exclude_tfbs", "length_cmap"},
    "tfbs_usage": {"max_sites", "couple_sigma_pairs", "exclude_tfbs"},
    "plan_counts": {"stacked_by_day", "created_at_col"},
    "tf_coverage": {
        "top_k",
        "normalize",
        "stacked",
        "include_promoter_sites",
        "alpha",
        "edge_alpha",
        "edge_width",
        "edge_color",
        "promoter_edge_alpha",
        "promoter_edge_style",
        "promoter_on_top",
        "promoter_zorder",
        "edge_on",
        "edge_dark_factor",
        # 'cfg' and 'style' are injected, not read from options
    },
    "tfbs_positional_frequency": {
        "top_k",
        "normalize",
        "include_promoter_sites",
        # 'cfg' and 'style' are injected, not read from options
    },
    "tfbs_positional_histogram": {
        "top_k",
        "normalize",
        "include_promoter_sites",
        "alpha",
        "edge_alpha",
        "edge_width",
        # 'cfg' and 'style' are injected, not read from options
    },
    "diversity_health": {
        "created_at_col",
        "normalize",
        "show_unique_tfs",
        "show_unique_tfbs",
        "show_entropy",
    },
    "tfbs_length_density": {
        "bins",
        "kde",
        "kde_bandwidth",
        "kde_points",
        "fill_alpha",
        "include_promoter_sites",
        "promoter_site_motifs",
    },
    "pad_gc": set(),
    "stage_a_strata_overview": set(),
}


def _filter_kwargs(name: str, kwargs: dict) -> dict:
    allowed = _ALLOWED_OPTIONS.get(name)
    if allowed is None:
        raise ValueError(f"Unknown plot name: {name}")
    unknown = [
        k
        for k in list(kwargs.keys())
        if k not in allowed and k not in {"dims", "palette", "palette_no_repeat", "style", "attempts_df"}
    ]
    if unknown:
        raise ValueError(f"Unknown options for plot '{name}': {unknown}")
    return kwargs


def _plot_required_columns(selected: Iterable[str], options: Dict[str, Dict[str, object]]) -> list[str]:
    cols: set[str] = set()
    for name in selected:
        raw = options.get(name, {}) if options else {}
        if name == "stage_a_strata_overview":
            continue
        if name == "compression_ratio":
            cols.add(_dg("compression_ratio"))
        elif name == "tf_usage":
            cols.update({_dg("used_tf_counts"), _dg("used_tfbs_detail")})
        elif name == "pad_gc":
            cols.update({_dg("pad_used"), _dg("pad_gc_actual"), _dg("pad_bases")})
        elif name == "plan_counts":
            cols.add(_dg("plan"))
            created_col = str(raw.get("created_at_col", _dg("created_at")))
            cols.add(created_col)
        elif name == "tf_coverage":
            cols.update({_dg("used_tfbs_detail"), _dg("length"), _dg("sequence_length")})
            include_promoter = bool(raw.get("include_promoter_sites", True))
            if include_promoter:
                cols.add("sequence")
        elif name == "tfbs_positional_frequency":
            cols.update({_dg("used_tfbs_detail"), _dg("length"), _dg("sequence_length")})
            include_promoter = bool(raw.get("include_promoter_sites", True))
            if include_promoter:
                cols.add("sequence")
        elif name == "tfbs_positional_histogram":
            cols.update({_dg("used_tfbs_detail"), _dg("length"), _dg("sequence_length")})
            include_promoter = bool(raw.get("include_promoter_sites", True))
            if include_promoter:
                cols.add("sequence")
        elif name == "diversity_health":
            cols.add(_dg("used_tfbs_detail"))
            created_col = str(raw.get("created_at_col", _dg("created_at")))
            cols.add(created_col)
        elif name == "tfbs_length_density":
            cols.add(_dg("used_tfbs_detail"))
        elif name == "tfbs_usage":
            cols.update({_dg("used_tfbs"), _dg("used_tfbs_detail")})
    return sorted(cols)


def _plot_required_sources(selected: Iterable[str]) -> set[str]:
    sources: set[str] = set()
    for name in selected:
        spec = AVAILABLE_PLOTS.get(name, {})
        requires = spec.get("requires")
        if requires:
            sources.update({str(item) for item in requires})
        else:
            sources.add("outputs")
    return sources


def run_plots_from_config(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    only: Optional[str] = None,
    source: str = "plot",
) -> None:
    plots_cfg = root_cfg.plots
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_dir = _ensure_out_dir(plots_cfg, cfg_path, run_root)
    plot_format = plots_cfg.format if plots_cfg and getattr(plots_cfg, "format", None) else "png"
    default_list = plots_cfg.default if (plots_cfg and plots_cfg.default) else list(AVAILABLE_PLOTS.keys())
    selected = [p.strip() for p in (only.split(",") if only else default_list)]
    options = plots_cfg.options if plots_cfg else {}
    global_style = plots_cfg.style if plots_cfg else {}
    required_sources = _plot_required_sources(selected)
    cols = _plot_required_columns(selected, options)
    max_rows = plots_cfg.sample_rows if plots_cfg else None
    df = pd.DataFrame()
    src_label = "none"
    row_count = 0
    if "outputs" in required_sources:
        df, src_label = load_records_from_config(root_cfg, cfg_path, columns=cols, max_rows=max_rows)
        row_count = len(df)
    pools: dict[str, pd.DataFrame] | None = None
    pool_manifest: TFBSPoolArtifact | None = None
    if "pools" in required_sources:
        pool_manifest, pools = _load_stage_a_pools(run_root)
        if row_count == 0:
            row_count = sum(len(pool_df) for pool_df in pools.values())
            src_label = f"pools:{run_root / 'outputs' / 'pools'}"

    _console.print(
        Panel.fit(
            f"DenseGen plotting • source: {src_label} • rows: {row_count:,}\nOutput: {out_dir}",
            border_style="blue",
        )
    )
    summary = Table("plot", "saved to", "status")
    errors: list[tuple[str, Exception]] = []
    manifest_entries: list[dict] = []

    for name in selected:
        if name not in AVAILABLE_PLOTS:
            raise ValueError(f"Unknown plot name requested: {name}")
        fn = AVAILABLE_PLOTS[name]["fn"]
        raw = (options.get(name, {}) or {}).copy()

        # absorb dims/palette into style
        dims = raw.pop("dims", None)
        style = {**global_style, **(raw.pop("style", {}) or {})}
        if dims:
            style["figsize"] = tuple(dims)
        pal_override = raw.pop("palette", None)
        if pal_override is not None:
            style["palette"] = pal_override
        if "palette_no_repeat" in raw:
            style["palette_no_repeat"] = bool(raw.pop("palette_no_repeat"))

        # drop unknown/retired options (e.g., promoter_scan_revcomp)
        kwargs = _filter_kwargs(name, raw)

        out_path = out_dir / f"{name}.{plot_format}"
        try:
            # pass cfg only to plots that need it
            if name in {
                "tfbs_usage",
                "plan_counts",
                "tf_coverage",
                "tfbs_positional_frequency",
                "tfbs_positional_histogram",
            }:
                result = fn(df, out_path, style=style, cfg=root_cfg.densegen.model_dump(), **kwargs)
            elif name == "tfbs_length_density":
                attempts_df = None
                attempts_path = run_root / "outputs" / "tables" / "attempts.parquet"
                if attempts_path.exists():
                    attempts_df = pd.read_parquet(attempts_path)
                result = fn(df, out_path, style=style, attempts_df=attempts_df, **kwargs)
            elif name == "stage_a_strata_overview":
                result = fn(df, out_path, style=style, pools=pools, pool_manifest=pool_manifest, **kwargs)
            else:
                result = fn(df, out_path, style=style, **kwargs)
            if result is None:
                paths = [out_path]
            elif isinstance(result, (list, tuple, set)):
                paths = [Path(p) for p in result]
            else:
                paths = [Path(result)]
            saved_label = str(paths[0]) if len(paths) == 1 else f"{paths[0]} (+{len(paths) - 1})"
            summary.add_row(name, saved_label, "[green]ok[/]")
            created_at = datetime.now(timezone.utc).isoformat()
            for path in paths:
                manifest_entries.append(
                    {
                        "name": name,
                        "path": str(path.relative_to(out_dir)),
                        "description": AVAILABLE_PLOTS[name]["description"],
                        "figsize": list(style.get("figsize", [])) if style.get("figsize") else None,
                        "generated_at": created_at,
                        "source": str(source),
                    }
                )
        except Exception as e:
            summary.add_row(name, "—", f"[red]failed[/] ({e})")
            errors.append((name, e))

    _console.print(summary)
    if errors:
        details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"{len(errors)} plot(s) failed: {details}")

    _write_plot_manifest(out_dir, entries=manifest_entries, run_root=run_root, cfg_path=cfg_path, source=source)
