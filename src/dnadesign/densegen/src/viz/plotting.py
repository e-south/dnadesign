"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/viz/plotting.py

Plots:
- placement_map   : 1-nt occupancy map across binding-site types
- tfbs_usage      : rank-frequency + distribution summary of TFBS usage
- run_health      : run outcomes, failures, and duplicate pressure
- stage_a_summary : Stage-A pool quality + yield + bias checks
- stage_b_summary : Stage-B feasibility + composition + utilization summary

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
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
from rich.console import Console

from ..adapters.outputs import load_records_from_config
from ..config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from ..core.artifacts.pool import POOL_MODE_TFBS, TFBSPoolArtifact, load_pool_artifact
from ..utils.plot_style import format_regulator_label, stage_a_rcparams
from ..utils.rich_style import make_panel, make_table
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


def _format_plot_path(path: Path, run_root: Path, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return str(path)


def _format_source_label(label: str, run_root: Path, absolute: bool) -> str:
    if ":" not in label:
        return label
    prefix, raw = label.split(":", 1)
    raw = raw.strip()
    if not raw:
        return label
    try:
        path = Path(raw)
    except Exception:
        return label
    return f"{prefix}:{_format_plot_path(path, run_root, absolute)}"


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


def _format_percent(value: float) -> str:
    if value >= 0.10:
        return f"{value * 100:.0f}%"
    if value >= 0.01:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.2f}%"


def _add_anchored_box(
    ax: mpl.axes.Axes,
    lines: list[str],
    *,
    loc: str = "upper right",
    fontsize: float = 9.0,
    alpha: float = 0.9,
    edgecolor: str | None = "#dddddd",
) -> AnchoredText | None:
    if not lines:
        return None
    text = "\n".join(lines)
    box = AnchoredText(text, loc=loc, prop={"size": fontsize}, frameon=True, pad=0.35, borderpad=0.4)
    box.patch.set_alpha(alpha)
    if edgecolor is None or str(edgecolor).lower() in {"none", "transparent"}:
        box.patch.set_edgecolor("none")
        box.patch.set_linewidth(0.0)
    else:
        box.patch.set_edgecolor(edgecolor)
    box.patch.set_facecolor("white")
    ax.add_artist(box)
    return box


def _draw_tier_markers(
    ax: mpl.axes.Axes,
    thresholds: list[tuple[str, float | None, str | None]],
    *,
    ymax_fraction: float = 0.58,
    label_mode: str = "box",
    loc: str = "upper right",
    fontsize: float | None = None,
) -> None:
    cleaned: list[tuple[str, float, str | None]] = []
    for label, value, label_text in thresholds:
        if value is None:
            continue
        cleaned.append((label, float(value), label_text))
    if not cleaned:
        return
    for _, value, _ in cleaned:
        ax.axvline(value, ymin=0.0, ymax=ymax_fraction, linestyle="--", linewidth=1, alpha=0.85, color="#222222")
    if label_mode == "box":
        lines = []
        for label, value, label_text in cleaned:
            value_text = f"{value:.2f}"
            if label_text:
                lines.append(f"{label}: {value_text} (n={label_text})")
            else:
                lines.append(f"{label}: {value_text}")
        y_min, y_max = ax.get_ylim()
        y_top = y_min + (y_max - y_min) * ymax_fraction
        for label, value, _ in cleaned:
            ax.scatter([value], [y_top], s=16, color="#222222", edgecolors="none", zorder=4)
            ax.annotate(
                label,
                (value, y_top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=(fontsize or float(ax.yaxis.label.get_size()) * 0.74) * 0.9,
                color="#222222",
            )
        box_size = fontsize or float(ax.yaxis.label.get_size()) * 0.74
        _add_anchored_box(ax, lines, loc=loc, fontsize=box_size, alpha=0.9, edgecolor="none")


def _shared_axis_cleanup(axes: list[mpl.axes.Axes]) -> None:
    if len(axes) < 2:
        return
    for ax in axes[:-1]:
        ax.label_outer()
        ax.tick_params(labelbottom=False)


def _shared_x_cleanup(axes: list[mpl.axes.Axes]) -> None:
    if len(axes) < 2:
        return
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)


def _pastelize_color(color: str, amount: float = 0.6) -> tuple[float, float, float, float]:
    base = to_rgba(color)
    return (
        base[0] + (1.0 - base[0]) * amount,
        base[1] + (1.0 - base[1]) * amount,
        base[2] + (1.0 - base[2]) * amount,
        base[3],
    )


def _stage_a_text_sizes(style: dict) -> dict[str, float]:
    font_size = float(style.get("font_size", 12.0))
    label_size = float(style.get("label_size", font_size))
    panel_title = float(style.get("title_size", font_size * 1.15))
    fig_title = float(style.get("fig_title_size", panel_title * 1.15))
    regulator_label = float(style.get("regulator_label_size", label_size * 0.95))
    sublabel = float(style.get("sublabel_size", label_size * 0.8))
    annotation = float(style.get("annotation_size", label_size * 0.72))
    return {
        "fig_title": fig_title,
        "panel_title": panel_title,
        "regulator_label": regulator_label,
        "sublabel": sublabel,
        "annotation": annotation,
    }


def _stage_a_regulator_colors(regulators: list[str], style: dict) -> dict[str, str]:
    base = _palette(style, max(len(regulators), 6), no_repeat=False)
    special = {"lexa": "#0072B2", "cpxr": "#009E73"}
    color_by_reg: dict[str, str] = {}
    used: set[str] = set()
    for reg in regulators:
        lowered = str(reg).strip().lower()
        if lowered.startswith("lexa"):
            color_by_reg[reg] = special["lexa"]
            used.add(special["lexa"])
        elif lowered.startswith("cpxr"):
            color_by_reg[reg] = special["cpxr"]
            used.add(special["cpxr"])
    available = [color for color in base if color not in used]
    if not available:
        available = list(base)
    idx = 0
    for reg in regulators:
        if reg in color_by_reg:
            continue
        color_by_reg[reg] = available[idx % len(available)]
        idx += 1
    return color_by_reg


def _plot_manifest_path(out_dir: Path) -> Path:
    return out_dir / "plot_manifest.json"


def _load_plot_manifest(out_dir: Path) -> dict:
    path = _plot_manifest_path(out_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_attempts(run_root: Path) -> pd.DataFrame:
    attempts_path = run_root / "outputs" / "tables" / "attempts.parquet"
    if not attempts_path.exists():
        raise ValueError(f"attempts.parquet not found: {attempts_path}")
    return pd.read_parquet(attempts_path)


def _load_run_metrics(run_root: Path) -> pd.DataFrame:
    metrics_path = run_root / "outputs" / "tables" / "run_metrics.parquet"
    if not metrics_path.exists():
        raise ValueError(f"run_metrics.parquet not found: {metrics_path}")
    return pd.read_parquet(metrics_path)


def _load_events(run_root: Path) -> pd.DataFrame:
    events_path = run_root / "outputs" / "meta" / "events.jsonl"
    if not events_path.exists():
        raise ValueError(f"events.jsonl not found: {events_path}")
    rows = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return pd.DataFrame(rows)


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
        if key in {"okabe_ito", "okabeito", "colorblind", "colorblind2", "colorblind_2"}:
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


def _plot_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    if isinstance(cfg.get("generation"), dict):
        return cfg
    nested = cfg.get("config")
    if isinstance(nested, dict):
        return nested
    return cfg


def _extract_promoter_site_motifs_from_cfg(cfg: dict) -> Dict[str, set[str]]:
    """Collect upstream/downstream motifs across all plan items; returns {'35 site': {...}, '10 site': {...}}."""
    cfg = _plot_config(cfg)
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


def _extract_fixed_element_ranges(cfg: dict, plan_name: str) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    cfg = _plot_config(cfg)
    gen = cfg.get("generation", {}) if cfg else {}
    for item in gen.get("plan", []) or []:
        name = str(item.get("name") or "").strip()
        if not name or name != plan_name:
            continue
        fixed = item.get("fixed_elements") or {}
        pcs = fixed.get("promoter_constraints")
        pcs = [pcs] if isinstance(pcs, dict) else (pcs or [])
        for p in (x for x in pcs if isinstance(x, dict)):
            upstream_pos = p.get("upstream_pos")
            downstream_pos = p.get("downstream_pos")
            if upstream_pos is not None:
                lo, hi = upstream_pos
                ranges.append(("promoter_upstream", int(lo), int(hi)))
            if downstream_pos is not None:
                lo, hi = downstream_pos
                ranges.append(("promoter_downstream", int(lo), int(hi)))
    return ranges


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


def _maybe_load_stage_a_pools(run_root: Path) -> tuple[TFBSPoolArtifact | None, dict[str, pd.DataFrame] | None]:
    pools_dir = run_root / "outputs" / "pools"
    if not pools_dir.exists():
        return None, None
    return _load_stage_a_pools(run_root)


def _load_composition(run_root: Path) -> pd.DataFrame:
    path = run_root / "outputs" / "tables" / "composition.parquet"
    if not path.exists():
        raise ValueError(f"composition.parquet not found: {path}")
    return pd.read_parquet(path)


def _maybe_load_composition(run_root: Path) -> pd.DataFrame | None:
    path = run_root / "outputs" / "tables" / "composition.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_libraries(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    libs_dir = run_root / "outputs" / "libraries"
    builds_path = libs_dir / "library_builds.parquet"
    members_path = libs_dir / "library_members.parquet"
    if not builds_path.exists():
        raise ValueError(f"library_builds.parquet not found: {builds_path}")
    if not members_path.exists():
        raise ValueError(f"library_members.parquet not found: {members_path}")
    return pd.read_parquet(builds_path), pd.read_parquet(members_path)


def _maybe_load_libraries(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    libs_dir = run_root / "outputs" / "libraries"
    builds_path = libs_dir / "library_builds.parquet"
    members_path = libs_dir / "library_members.parquet"
    if not builds_path.exists() or not members_path.exists():
        return None
    return pd.read_parquet(builds_path), pd.read_parquet(members_path)


def _load_effective_config(run_root: Path) -> dict:
    path = run_root / "outputs" / "meta" / "effective_config.json"
    if not path.exists():
        raise ValueError(f"effective_config.json not found: {path}")
    return json.loads(path.read_text())


def _sequence_length_from_cfg(cfg: dict) -> int:
    cfg = _plot_config(cfg)
    gen = cfg.get("generation") if cfg else None
    if not isinstance(gen, dict):
        raise ValueError("Plot config missing generation block.")
    length = gen.get("sequence_length")
    if length is None:
        raise ValueError("Plot config missing generation.sequence_length.")
    return int(length)


def _fixed_element_alias(label: str) -> str:
    text = str(label or "").lower()
    if "upstream" in text or "promoter_upstream" in text:
        return "-35"
    if "downstream" in text or "promoter_downstream" in text:
        return "-10"
    return str(label)


def _bin_attempts(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, num=int(bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def _axis_pixel_width(ax) -> float:
    fig = ax.figure
    width = fig.get_figwidth() * fig.dpi
    return max(1.0, width * ax.get_position().width)


def _resolution_bins(ax, n_points: int, *, min_bins: int = 25) -> int:
    if n_points <= 0:
        return int(min_bins)
    px = _axis_pixel_width(ax)
    return max(min_bins, min(int(px), int(n_points)))


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    seq = str(seq).upper()
    gc = sum(1 for ch in seq if ch in {"G", "C"})
    return float(gc) / float(len(seq))


def plot_placement_map(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    cfg: dict,
    style: Optional[dict] = None,
) -> list[Path]:
    if composition_df is None or composition_df.empty:
        raise ValueError("placement_map requires composition.parquet with placements.")
    required = {"solution_id", "input_name", "plan_name", "tf", "tfbs"}
    missing = required - set(composition_df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    start_col = "offset" if "offset" in composition_df.columns else None
    length_col = "length" if "length" in composition_df.columns else None
    end_col = "end" if "end" in composition_df.columns else None
    if start_col is None and end_col is None:
        raise ValueError("composition.parquet requires offset or end columns.")
    if length_col is None and end_col is None:
        raise ValueError("composition.parquet requires length or end columns.")
    style = _style(style)
    seq_len = _sequence_length_from_cfg(cfg)
    paths: list[Path] = []
    for (input_name, plan_name), sub in composition_df.groupby(["input_name", "plan_name"]):
        sub = sub.copy()
        n_solutions = sub["solution_id"].astype(str).nunique()
        if n_solutions <= 0:
            raise ValueError(f"placement_map has no solutions for {input_name}/{plan_name}.")
        fixed_ranges = _extract_fixed_element_ranges(cfg, str(plan_name))
        fixed_labels = [_fixed_element_alias(label) for label, _lo, _hi in fixed_ranges]
        tf_labels = sorted({str(tf) for tf in sub["tf"].astype(str).tolist() if str(tf).strip()})
        labels = tf_labels + [lab for lab in fixed_labels if lab not in tf_labels]
        if not labels:
            raise ValueError(f"placement_map has no binding-site labels for {input_name}/{plan_name}.")
        occupancy = {label: np.zeros(seq_len, dtype=float) for label in labels}
        for _sol_id, sol_df in sub.groupby("solution_id"):
            for label, group in sol_df.groupby("tf"):
                label = str(label)
                if label not in occupancy:
                    continue
                diff = np.zeros(seq_len + 1, dtype=float)
                for _, row in group.iterrows():
                    start = int(row.get(start_col) or 0)
                    if length_col is not None and row.get(length_col) is not None:
                        length = int(row.get(length_col))
                        end = start + length
                    else:
                        end = int(row.get(end_col) or 0)
                        length = end - start
                    if length <= 0:
                        continue
                    lo = max(0, min(start, seq_len))
                    hi = max(lo, min(end, seq_len))
                    if hi <= lo:
                        continue
                    diff[lo] += 1.0
                    diff[hi] -= 1.0
                covered = np.cumsum(diff[:-1]) > 0
                occupancy[label] += covered.astype(float)
        for label, lo, hi in fixed_ranges:
            alias = _fixed_element_alias(label)
            if alias not in occupancy:
                occupancy[alias] = np.zeros(seq_len, dtype=float)
                labels.append(alias)
            lo = max(0, min(int(lo), seq_len))
            hi = max(lo, min(int(hi), seq_len))
            if hi > lo:
                occupancy[alias][lo:hi] = float(n_solutions)
        mat = np.vstack([occupancy[label] for label in labels]) / float(n_solutions)
        fig, ax = _fig_ax(style)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        xticks = np.linspace(0, max(0, seq_len - 1), num=min(seq_len, 8), dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x + 1) for x in xticks])
        ax.set_xlabel("Position (nt)")
        ax.set_ylabel("Binding-site type")
        ax.set_title(f"Placement map — {input_name}/{plan_name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of solutions")
        _apply_style(ax, style)
        fig.tight_layout()
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_tfbs_usage(
    df: pd.DataFrame,
    out_path: Path,
    *,
    composition_df: pd.DataFrame,
    pools: dict[str, pd.DataFrame] | None = None,
    library_members_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if composition_df is None or composition_df.empty:
        raise ValueError("tfbs_usage requires composition.parquet with placements.")
    required = {"input_name", "plan_name", "tf", "tfbs"}
    missing = required - set(composition_df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    paths: list[Path] = []
    for (input_name, plan_name), sub in composition_df.groupby(["input_name", "plan_name"]):
        counts = sub.groupby(["tf", "tfbs"]).size().sort_values(ascending=False)
        if counts.empty:
            raise ValueError(f"tfbs_usage found no TFBS counts for {input_name}/{plan_name}.")
        ranks = np.arange(1, len(counts) + 1)
        values = counts.to_numpy(dtype=float)
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=style["figsize"])
        ax_left.plot(ranks, values, color="#4c78a8", linewidth=1.8)
        ax_left.set_xlabel("TFBS rank (by usage)")
        ax_left.set_ylabel("Usage count")
        ax_left.set_title("TFBS rank–frequency")

        ax_right.hist(values, bins="fd", color="#4c78a8", alpha=0.75)
        ax_right.set_xlabel("Usage count")
        ax_right.set_ylabel("Count")
        ax_right.set_title("TFBS usage distribution")
        ax_ecdf = ax_right.twinx()
        sorted_vals = np.sort(values)
        ax_ecdf.plot(sorted_vals, np.arange(1, len(sorted_vals) + 1) / len(sorted_vals), color="#f28e2b")
        ax_ecdf.set_ylabel("ECDF")

        annotations = []
        used_unique = int(len(counts))
        if pools and input_name in pools:
            pool_df = pools[input_name]
            if "tfbs_sequence" in pool_df.columns:
                tfbs_col = "tfbs_sequence"
            else:
                tfbs_col = "tfbs"
            pool_unique = pool_df.assign(tf=pool_df["tf"].astype(str), tfbs=pool_df[tfbs_col].astype(str))
            pool_unique = pool_unique.drop_duplicates(subset=["tf", "tfbs"])
            pool_count = int(len(pool_unique))
            if pool_count > 0:
                annotations.append(f"pool used ≥1: {used_unique}/{pool_count} ({used_unique / pool_count:.2%})")
        if library_members_df is not None and not library_members_df.empty:
            offered = library_members_df[
                (library_members_df["input_name"].astype(str) == str(input_name))
                & (library_members_df["plan_name"].astype(str) == str(plan_name))
            ]
            if not offered.empty:
                offered_unique = offered.drop_duplicates(subset=["tf", "tfbs"])
                offered_count = int(len(offered_unique))
                if offered_count > 0:
                    annotations.append(
                        f"offered used ≥1: {used_unique}/{offered_count} ({used_unique / offered_count:.2%})"
                    )
        if annotations:
            ax_right.text(0.98, 0.98, "\n".join(annotations), transform=ax_right.transAxes, ha="right", va="top")
        _apply_style(ax_left, style)
        _apply_style(ax_right, style)
        fig.tight_layout()
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_run_health(
    df: pd.DataFrame,
    out_path: Path,
    *,
    attempts_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    style: Optional[dict] = None,
) -> None:
    if attempts_df is None or attempts_df.empty:
        raise ValueError("run_health requires attempts.parquet.")
    required = {"attempt_index", "status", "reason", "plan_name"}
    missing = required - set(attempts_df.columns)
    if missing:
        raise ValueError(f"attempts.parquet missing required columns: {sorted(missing)}")
    style = _style(style)
    attempts_df = attempts_df.copy()
    attempts_df["attempt_index"] = pd.to_numeric(attempts_df["attempt_index"], errors="coerce").fillna(0).astype(int)
    attempts_df["created_at"] = pd.to_datetime(attempts_df.get("created_at"), errors="coerce")
    statuses = ["success", "duplicate", "failed"]
    plan_names = sorted({str(p) for p in attempts_df["plan_name"].astype(str).tolist() if str(p).strip()})
    show_plans = len(plan_names) > 1
    if show_plans:
        fig, axes = plt.subplots(2, 2, figsize=style["figsize"])
        ax_outcome, ax_dup, ax_fail, ax_plan = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=style["figsize"])
        ax_outcome, ax_dup, ax_fail = axes
        ax_plan = None

    bins = _resolution_bins(ax_outcome, len(attempts_df))
    edges, centers = _bin_attempts(attempts_df["attempt_index"].to_numpy(dtype=float), bins)
    if centers.size == 0:
        raise ValueError("run_health cannot bin attempts without attempt_index values.")
    counts_by_status: dict[str, np.ndarray] = {}
    for status in statuses:
        sub = attempts_df[attempts_df["status"].astype(str) == status]
        counts, _ = np.histogram(sub["attempt_index"].to_numpy(dtype=float), bins=edges)
        counts_by_status[status] = counts.astype(float)

    ax_outcome.stackplot(centers, [counts_by_status[s] for s in statuses], labels=statuses)
    ax_outcome.set_xlabel("Attempt index (binned)")
    ax_outcome.set_ylabel("Count")
    ax_outcome.set_title("Outcome mix")
    ax_outcome.legend(loc="upper right", frameon=bool(style.get("legend_frame", False)))

    totals = sum(counts_by_status.values())
    dup_rate = np.divide(counts_by_status.get("duplicate", np.zeros_like(totals)), np.where(totals > 0, totals, 1.0))
    ax_dup.plot(centers, dup_rate, color="#e15759")
    ax_dup.set_xlabel("Attempt index (binned)")
    ax_dup.set_ylabel("Duplicate rate")
    ax_dup.set_ylim(0.0, min(1.0, max(0.05, float(np.nanmax(dup_rate)) + 0.05)))
    ax_dup.set_title("Duplicate pressure")

    failed = attempts_df[attempts_df["status"].astype(str) == "failed"]
    if failed.empty:
        ax_fail.text(0.5, 0.5, "No failures", ha="center", va="center", transform=ax_fail.transAxes)
        ax_fail.set_axis_off()
    else:
        reason_counts = failed["reason"].astype(str).value_counts()
        if len(reason_counts) <= 8:
            positions = np.arange(len(reason_counts))
            ax_fail.bar(positions, reason_counts.values.tolist(), color="#4c78a8")
            ax_fail.set_xticks(positions)
            ax_fail.set_xticklabels(reason_counts.index.tolist(), rotation=45, ha="right")
            ax_fail.set_ylabel("Count")
            ax_fail.set_title("Failure reasons")
        else:
            ranks = np.arange(1, len(reason_counts) + 1)
            ax_fail.plot(ranks, reason_counts.values, color="#4c78a8", linewidth=1.6)
            ax_fail.set_xlabel("Reason rank")
            ax_fail.set_ylabel("Count")
            ax_fail.set_title("Failure rank–frequency")

    if show_plans and ax_plan is not None:
        for plan in plan_names:
            sub = attempts_df[(attempts_df["plan_name"].astype(str) == plan) & (attempts_df["status"] == "success")]
            counts, _ = np.histogram(sub["attempt_index"].to_numpy(dtype=float), bins=edges)
            cumulative = np.cumsum(counts)
            ax_plan.plot(centers, cumulative, label=plan)
        ax_plan.set_xlabel("Attempt index (binned)")
        ax_plan.set_ylabel("Cumulative successes")
        ax_plan.set_title("Plan progress")
        ax_plan.legend(loc="upper left", frameon=bool(style.get("legend_frame", False)))

    if events_df is not None and not events_df.empty and "created_at" in attempts_df.columns:
        event_times = pd.to_datetime(events_df.get("created_at"), errors="coerce").dropna()
        if not event_times.empty:
            attempt_times = attempts_df.dropna(subset=["created_at"]).sort_values("created_at")
            if not attempt_times.empty:
                idx_values = attempt_times["attempt_index"].to_numpy()
                time_values = attempt_times["created_at"].to_numpy()
                for evt_time in event_times.to_numpy():
                    insert = np.searchsorted(time_values, evt_time)
                    if insert <= 0:
                        event_idx = idx_values[0]
                    elif insert >= len(idx_values):
                        event_idx = idx_values[-1]
                    else:
                        before = time_values[insert - 1]
                        after = time_values[insert]
                        use_prev = (evt_time - before) <= (after - evt_time)
                        event_idx = idx_values[insert - 1] if use_prev else idx_values[insert]
                    ax_outcome.axvline(event_idx, color="#999999", linestyle="--", linewidth=1)

    for ax in [ax_outcome, ax_dup, ax_fail] + ([ax_plan] if ax_plan is not None else []):
        if ax is not None:
            _apply_style(ax, style)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_stage_a_strata_overview_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes, mpl.axes.Axes]:
    style = _style(style)
    style["seaborn_style"] = False
    rc = stage_a_rcparams(style)
    text_sizes = _stage_a_text_sizes(style)
    if sampling.get("backend") != "fimo":
        raise ValueError(f"Stage-A strata overview requires FIMO sampling (input '{input_name}').")
    eligible_score_hist = sampling.get("eligible_score_hist") or []
    if not eligible_score_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    if "regulator_id" in pool_df.columns:
        tf_col = "regulator_id"
    elif "tf" in pool_df.columns:
        tf_col = "tf"
    else:
        raise ValueError(f"Stage-A pool missing regulator_id or tf column for input '{input_name}'.")
    if "tfbs_sequence" in pool_df.columns:
        tfbs_col = "tfbs_sequence"
    elif "tfbs" in pool_df.columns:
        tfbs_col = "tfbs"
    else:
        raise ValueError(f"Stage-A pool missing tfbs_sequence or tfbs column for input '{input_name}'.")
    if "best_hit_score" not in pool_df.columns:
        raise ValueError(f"Stage-A pool missing best_hit_score for input '{input_name}'.")

    regulators = [str(row.get("regulator") or "") for row in eligible_score_hist]
    hist_by_reg: dict[str, tuple[list[float], list[int], float | None, float | None, float | None]] = {}
    global_scores: list[float] = []
    for row in eligible_score_hist:
        reg = str(row.get("regulator"))
        edges = [float(v) for v in (row.get("edges") or [])]
        counts = [int(v) for v in (row.get("counts") or [])]
        tier0_score = row.get("tier0_score")
        tier1_score = row.get("tier1_score")
        tier2_score = row.get("tier2_score")
        if edges and len(counts) != len(edges) - 1:
            raise ValueError(f"Eligible score histogram length mismatch for '{input_name}' ({reg}).")
        for val in edges:
            global_scores.append(float(val))
        for val in (tier0_score, tier1_score, tier2_score):
            if val is not None:
                global_scores.append(float(val))
        hist_by_reg[reg] = (
            edges,
            counts,
            float(tier0_score) if tier0_score is not None else None,
            float(tier1_score) if tier1_score is not None else None,
            float(tier2_score) if tier2_score is not None else None,
        )

    base_colors = _stage_a_regulator_colors(regulators, style)
    color_by_reg = {reg: _pastelize_color(color, amount=0.35) for reg, color in base_colors.items()}

    n_regs = max(1, len(regulators))
    fig_width = float(style.get("figsize", (11, 4))[0])
    fig_height = max(3.8, 1.35 * n_regs + 1.2)
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        header_height = min(0.95, fig_height * 0.18)
        body_height = max(1.0, fig_height - header_height)
        outer = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[header_height, body_height],
            hspace=0.05,
        )
        ax_header = fig.add_subplot(outer[0, 0])
        ax_header.set_axis_off()
        ax_header.set_label("header")
        ax_header.text(
            0.5,
            0.86,
            f"Stage-A pool tiers — {input_name}",
            ha="center",
            va="center",
            fontsize=text_sizes["fig_title"],
            color="#111111",
        )
        gs = outer[1].subgridspec(
            nrows=n_regs,
            ncols=2,
            width_ratios=[2.2, 1.1],
            hspace=0.28,
            wspace=0.28,
        )
        axes_left: list[mpl.axes.Axes] = []
        for idx in range(n_regs):
            ax = fig.add_subplot(gs[idx, 0], sharex=axes_left[0] if axes_left else None)
            axes_left.append(ax)
        ax_right = fig.add_subplot(gs[:, 1])

        retained_tiers = {}
        core_lengths: dict[str, int] = {}
        if "tfbs_core" in pool_df.columns:
            core_series = pool_df["tfbs_core"].astype(str)
            for reg, core in zip(pool_df[tf_col].astype(str).to_list(), core_series.to_list()):
                core_lengths.setdefault(reg, []).append(len(core))
            core_lengths = {reg: int(np.median(vals)) for reg, vals in core_lengths.items() if vals}
        if "tier" in pool_df.columns:
            tier_counts = pool_df.groupby([tf_col, "tier"], dropna=False).size().rename("count").reset_index()
            for _, row in tier_counts.iterrows():
                retained_tiers.setdefault(str(row[tf_col]), {})[int(row["tier"])] = int(row["count"])

        if not regulators:
            axes_left[0].text(0.5, 0.5, "No eligible hits", ha="center", va="center", transform=axes_left[0].transAxes)
        else:
            if global_scores:
                global_min = float(min(global_scores))
                global_max = float(max(global_scores))
                pad = max(0.25, (global_max - global_min) * 0.03) if global_max > global_min else 0.25
                global_min -= pad
                global_max += pad
            else:
                global_min, global_max = 0.0, 1.0
            for idx, reg in enumerate(regulators):
                ax = axes_left[idx]
                edges, counts, tier0_score, tier1_score, tier2_score = hist_by_reg.get(reg, ([], [], None, None, None))
                if not edges:
                    ax.text(0.5, 0.5, "No eligible hits", ha="center", va="center", transform=ax.transAxes)
                else:
                    counts_arr = np.asarray(counts, dtype=float)
                    max_count = float(counts_arr.max()) if counts_arr.size else 0.0
                    scale = max_count if max_count > 0 else 1.0
                    density = counts_arr / scale
                    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2.0
                    hue = color_by_reg.get(reg, "#4c78a8")
                    ax.fill_between(centers, 0.0, density, color=hue, alpha=0.28)
                    ax.plot(centers, density, color=hue, linewidth=1.2)
                    retained_vals = pd.to_numeric(
                        pool_df.loc[pool_df[tf_col].astype(str) == reg, "best_hit_score"],
                        errors="coerce",
                    ).dropna()
                    if not retained_vals.empty:
                        retained_counts, _ = np.histogram(retained_vals.to_numpy(dtype=float), bins=np.asarray(edges))
                        retained_arr = np.asarray(retained_counts, dtype=float)
                        if retained_arr.max() > 0:
                            retained_density = retained_arr / scale
                            ax.fill_between(centers, 0.0, retained_density, color=hue, alpha=0.5)
                    retained = retained_tiers.get(reg, {})
                    _draw_tier_markers(
                        ax,
                        [
                            ("0.1%", tier0_score, str(retained.get(0, 0))),
                            ("1%", tier1_score, str(retained.get(1, 0))),
                            ("9%", tier2_score, str(retained.get(2, 0))),
                        ],
                        ymax_fraction=0.58,
                        label_mode="box",
                        loc="lower right",
                        fontsize=text_sizes["annotation"] * 0.65,
                    )
                ax.set_yticks([])
                ax.set_ylim(0, 1.05)
                label = format_regulator_label(reg)
                core_len = core_lengths.get(reg)
                ax.set_ylabel("")
                ax.text(
                    -0.015,
                    0.64,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=text_sizes["regulator_label"],
                    color="#222222",
                    clip_on=False,
                )
                if core_len:
                    ax.text(
                        -0.015,
                        0.34,
                        f"(core {core_len} bp)",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=text_sizes["sublabel"],
                        color="#555555",
                        clip_on=False,
                    )
                ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            for ax in axes_left:
                ax.set_xlim(global_min, global_max)

        if axes_left:
            axes_left[0].set_title(
                "Eligible score tiers with retained overlays",
                fontsize=text_sizes["annotation"],
                color="#444444",
                pad=12,
                loc="center",
            )
            axes_left[-1].set_xlabel("FIMO log-odds score")
            axes_left[-1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            _shared_axis_cleanup(axes_left)

        lengths_by_reg: dict[str, list[int]] = {}
        for reg, seq in pool_df[[tf_col, tfbs_col]].itertuples(index=False):
            reg_label = str(reg)
            lengths_by_reg.setdefault(reg_label, []).append(len(str(seq)))
        all_lengths = [val for vals in lengths_by_reg.values() for val in vals]
        if not all_lengths:
            ax_right.text(0.5, 0.5, "No retained sequences", ha="center", va="center", transform=ax_right.transAxes)
        else:
            min_len = int(min(all_lengths))
            max_len = int(max(all_lengths))
            bins = np.arange(min_len - 0.5, max_len + 1.5, 1.0)
            for reg, lengths in lengths_by_reg.items():
                if not lengths:
                    continue
                hue = color_by_reg.get(reg, "#4c78a8")
                ax_right.hist(
                    lengths,
                    bins=bins,
                    density=False,
                    alpha=0.25,
                    color=hue,
                    edgecolor=hue,
                    linewidth=0.7,
                )
            span = max_len - min_len
            pad = max(5, int(round(span * 0.05))) if span > 0 else 5
            ax_right.set_xlim(min_len - pad, max_len + pad)
            ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
        ax_right.set_xlabel("TFBS length (nt)")
        ax_right.set_ylabel("Count")
        ax_right.set_title(
            "Retained TFBS length counts",
            fontsize=text_sizes["annotation"],
            color="#444444",
            pad=12,
        )

        legend_handles = [
            Patch(
                facecolor=color_by_reg[reg],
                edgecolor=color_by_reg[reg],
                label=format_regulator_label(reg),
                alpha=0.35,
            )
            for reg in regulators
        ]
        if legend_handles:
            ax_right.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=False,
                fontsize=text_sizes["annotation"] * 0.8,
            )

        for ax in axes_left + [ax_right]:
            _apply_style(ax, style)

    ax_left = axes_left[-1]
    return fig, ax_left, ax_right


def _build_stage_a_yield_bias_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], list[mpl.axes.Axes], mpl.axes.Axes]:
    style = _style(style)
    style["seaborn_style"] = False
    rc = stage_a_rcparams(style)
    text_sizes = _stage_a_text_sizes(style)
    eligible_hist = sampling.get("eligible_score_hist") or []
    if not eligible_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")

    regs = []
    stage_counts: list[list[int]] = []
    for row in eligible_hist:
        reg = str(row.get("regulator") or "")
        generated = row.get("generated")
        hit = row.get("candidates_with_hit")
        eligible = row.get("eligible")
        unique = row.get("eligible_unique")
        retained = row.get("retained")
        if generated in (None, 0):
            raise ValueError(f"Stage-A sampling missing generated counts for input '{input_name}'.")
        if any(val is None for val in (hit, eligible, unique, retained)):
            raise ValueError(f"Stage-A sampling missing yield counters for input '{input_name}'.")
        regs.append(reg)
        stage_counts.append([int(generated), int(hit), int(eligible), int(unique), int(retained)])

    if "tfbs_sequence" in pool_df.columns:
        tfbs_col = "tfbs_sequence"
    elif "tfbs" in pool_df.columns:
        tfbs_col = "tfbs"
    else:
        raise ValueError(f"Stage-A pool missing tfbs_sequence or tfbs for input '{input_name}'.")
    scores = pd.to_numeric(pool_df.get("best_hit_score"), errors="coerce")
    if scores is None or scores.dropna().empty:
        raise ValueError(f"Stage-A pool missing best_hit_score for input '{input_name}'.")
    seqs = pool_df[tfbs_col].astype(str)
    lengths = seqs.map(len)
    gc = seqs.map(_gc_fraction)
    if "regulator_id" in pool_df.columns:
        tf_col = "regulator_id"
    elif "tf" in pool_df.columns:
        tf_col = "tf"
    else:
        raise ValueError(f"Stage-A pool missing regulator_id or tf column for input '{input_name}'.")

    lengths = lengths.to_numpy(dtype=float)
    scores_arr = scores.to_numpy(dtype=float)
    gc_arr = gc.to_numpy(dtype=float)
    tf_vals = pool_df[tf_col].astype(str).to_numpy()
    base_keys = [f"{tf}|{seq}" for tf, seq in zip(tf_vals, seqs.to_list())]
    core_lengths: dict[str, int] = {}
    if "tfbs_core" in pool_df.columns:
        core_series = pool_df["tfbs_core"].astype(str)
        for reg, core in zip(tf_vals, core_series.to_list()):
            core_lengths.setdefault(str(reg), []).append(len(core))
        core_lengths = {reg: int(np.median(vals)) for reg, vals in core_lengths.items() if vals}

    def _stable_jitter(keys: list[str], width: float = 0.18) -> np.ndarray:
        import hashlib

        values = []
        for key in keys:
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
            values.append((bucket - 0.5) * width + 0.0)
        return np.asarray(values) if values else np.zeros((0,), dtype=float)

    jitter = _stable_jitter(base_keys)
    lengths_j = lengths + jitter

    fig_width = float(style.get("figsize", (11, 4.2))[0])
    base_height = float(style.get("figsize", (11, 4.2))[1])
    reg_order = [reg for reg in regs if reg]
    if not reg_order:
        reg_order = sorted({str(val) for val in tf_vals})
    n_regs = max(1, len(reg_order))
    fig_height = max(4.8, base_height, 1.75 * n_regs + 0.8)
    reg_colors = _stage_a_regulator_colors(reg_order, style)
    stage_labels = ["Generated", "Hit", "Eligible", "Unique", "Retained"]
    counts_by_reg = {reg: counts for reg, counts in zip(regs, stage_counts)}
    max_count = max((max(counts) for counts in stage_counts), default=0)
    subtitle_size = text_sizes["panel_title"] * 0.88
    tick_size = text_sizes["annotation"] * 0.65
    title_pad = 12
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        header_height = min(0.95, fig_height * 0.18)
        body_height = max(1.0, fig_height - header_height)
        outer = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[header_height, body_height],
            hspace=0.05,
        )
        ax_header = fig.add_subplot(outer[0, 0])
        ax_header.set_axis_off()
        ax_header.set_label("header")
        ax_header.text(
            0.5,
            0.74,
            f"Stage-A yield & bias — {input_name}",
            ha="center",
            va="center",
            fontsize=text_sizes["fig_title"],
            color="#111111",
        )
        body = outer[1].subgridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 0.05],
            wspace=0.12,
        )
        main = body[0, 0].subgridspec(
            nrows=n_regs,
            ncols=2,
            width_ratios=[1.0, 1.0],
            hspace=0.32,
            wspace=0.52,
        )
        axes_left: list[mpl.axes.Axes] = []
        axes_right: list[mpl.axes.Axes] = []
        for idx in range(n_regs):
            share_left = axes_left[0] if axes_left else None
            share_right = axes_right[0] if axes_right else None
            axes_left.append(fig.add_subplot(main[idx, 0], sharex=share_left, sharey=share_left))
            axes_right.append(fig.add_subplot(main[idx, 1], sharex=share_right, sharey=share_right))
        cbar_ax = fig.add_subplot(body[0, 1])

        if not reg_order:
            axes_left[0].text(0.5, 0.5, "No yield data", ha="center", va="center", transform=axes_left[0].transAxes)
            axes_left[0].set_axis_off()
        else:
            stage_labels = [label.capitalize() for label in stage_labels]
            x_positions = np.arange(len(stage_labels))
            offset = max(1.0, max_count * 0.03) if max_count else 1.0
            y_limit = max_count * 1.25 + offset if max_count else 1.0
            for idx, reg in enumerate(reg_order):
                ax = axes_left[idx]
                counts = counts_by_reg.get(reg)
                if not counts:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                else:
                    hue = reg_colors.get(reg, "#4c78a8")
                    ax.plot(x_positions, counts, color=hue, marker="o", linewidth=1.4, markersize=4)
                    ax.set_ylim(0.0, y_limit)
                    for step_idx, cur in enumerate(counts):
                        prev = counts[step_idx - 1] if step_idx > 0 else cur
                        frac = float(cur) / float(prev) if prev else 0.0
                        label = f"{cur:,}\n{_format_percent(frac)}"
                        ax.annotate(
                            label,
                            (x_positions[step_idx], cur),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=text_sizes["annotation"] * 0.75,
                            color="#222222",
                        )
                label = format_regulator_label(reg)
                ax.set_ylabel("")
                ax.text(
                    -0.18,
                    0.64,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=text_sizes["regulator_label"] * 0.84,
                    color="#222222",
                    clip_on=False,
                )
                core_len = core_lengths.get(str(reg))
                if core_len:
                    ax.text(
                        -0.18,
                        0.34,
                        f"(core {core_len} bp)",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=text_sizes["sublabel"] * 0.84,
                        color="#555555",
                        clip_on=False,
                    )
                ax.tick_params(axis="y", pad=1, labelsize=tick_size)
                ax.tick_params(axis="x", labelsize=tick_size)
                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
                ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, integer=True))

            axes_left[0].set_title("Stepwise sequence yield", fontsize=subtitle_size, pad=title_pad)
            axes_left[-1].set_xticks(x_positions)
            axes_left[-1].set_xticklabels(stage_labels)
            axes_left[-1].set_xlabel("Stage")
            for ax in axes_left:
                ax.set_xlim(-0.5, len(stage_labels) - 1 + 0.5)
            _shared_x_cleanup(axes_left)

        for idx, reg in enumerate(reg_order):
            ax = axes_right[idx]
            mask = tf_vals == reg
            if not np.any(mask):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.scatter(
                    lengths_j[mask],
                    scores_arr[mask],
                    c=gc_arr[mask],
                    cmap="viridis",
                    alpha=0.65,
                    s=12,
                    marker="o",
                    edgecolors="none",
                )
            ax.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        axes_right[0].set_title(
            "Retained sites: score vs length (GC color)",
            fontsize=subtitle_size,
            pad=title_pad,
        )
        axes_right[-1].set_xlabel("TFBS length (nt)")
        for ax in axes_right:
            ax.set_ylabel("Best-hit score")
            ax.tick_params(axis="y", labelsize=tick_size, labelleft=True)
            ax.tick_params(axis="x", labelsize=tick_size)
        x_min = float(np.nanmin(lengths)) if len(lengths) else 0.0
        x_max = float(np.nanmax(lengths)) if len(lengths) else 0.0
        if x_max > x_min:
            for ax in axes_right:
                ax.set_xlim(x_min - 1.0, x_max + 1.0)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("GC fraction", fontsize=text_sizes["annotation"] * 0.85)
        cbar.ax.tick_params(labelsize=tick_size)
        _shared_x_cleanup(axes_right)
        for ax in axes_left + axes_right:
            _apply_style(ax, style)
            ax.tick_params(axis="both", labelsize=tick_size)
    return fig, axes_left, axes_right, cbar_ax


def _build_stage_a_diversity_figure(
    *,
    input_name: str,
    pool_df: pd.DataFrame,
    sampling: dict,
    style: dict,
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes], list[mpl.axes.Axes]]:
    style = _style(style)
    style["seaborn_style"] = False
    rc = stage_a_rcparams(style)
    text_sizes = _stage_a_text_sizes(style)
    eligible_hist = sampling.get("eligible_score_hist") or []
    if not eligible_hist:
        raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
    regulators = [str(row.get("regulator") or "") for row in eligible_hist]
    reg_colors = _stage_a_regulator_colors(regulators, style)
    n_regs = max(1, len(regulators))
    fig_width = float(style.get("figsize", (11, 4))[0])
    fig_height = max(4.0, 1.5 * n_regs + 1.1)
    subtitle_size = text_sizes["panel_title"] * 0.88
    title_pad = 12
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        header_height = min(0.95, fig_height * 0.18)
        body_height = max(1.0, fig_height - header_height)
        outer = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[header_height, body_height],
            hspace=0.05,
        )
        ax_header = fig.add_subplot(outer[0, 0])
        ax_header.set_axis_off()
        ax_header.set_label("header")
        ax_header.text(
            0.5,
            0.76,
            f"Stage-A core diversity (tfbs_core only; baseline vs actual) — {input_name}",
            ha="center",
            va="center",
            fontsize=text_sizes["fig_title"],
            color="#111111",
        )
        body = outer[1].subgridspec(
            nrows=n_regs,
            ncols=3,
            width_ratios=[0.22, 1.15, 1.0],
            hspace=0.32,
            wspace=0.35,
        )
        axes_left: list[mpl.axes.Axes] = []
        axes_right: list[mpl.axes.Axes] = []
        label_axes: list[mpl.axes.Axes] = []
        for idx in range(n_regs):
            label_axes.append(fig.add_subplot(body[idx, 0]))
            axes_left.append(fig.add_subplot(body[idx, 1]))
            axes_right.append(fig.add_subplot(body[idx, 2]))

        diversity_by_reg = {str(row.get("regulator") or ""): row.get("diversity") for row in eligible_hist}

        def _ecdf_from_counts(bins: list[float] | list[int], counts: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
            if not bins or not counts:
                return None
            arr = np.asarray(counts, dtype=float)
            total = float(arr.sum())
            if total <= 0:
                return None
            x = np.asarray(bins, dtype=float)
            y = np.cumsum(arr) / total
            return x, y

        for idx, reg in enumerate(regulators):
            hue = reg_colors.get(reg, "#4c78a8")
            diversity = diversity_by_reg.get(reg) if isinstance(diversity_by_reg.get(reg), dict) else None
            core_len = None
            if isinstance(diversity, dict):
                entropy_block = diversity.get("core_entropy")
                if isinstance(entropy_block, dict):
                    actual_vals = entropy_block.get("actual", {}).get("values", [])
                    if actual_vals:
                        core_len = len(actual_vals)
            label = format_regulator_label(reg)
            if core_len:
                label = f"{label} (core {core_len} bp)"
            ax_label = label_axes[idx]
            ax_left = axes_left[idx]
            ax_right = axes_right[idx]
            ax_label.set_axis_off()
            ax_label.text(
                0.98,
                0.5,
                label,
                ha="right",
                va="center",
                fontsize=text_sizes["regulator_label"] * 0.95,
                color="#222222",
            )
            core_hamming = diversity.get("core_hamming") if isinstance(diversity, dict) else None
            if not isinstance(core_hamming, dict):
                ax_left.text(0.5, 0.5, "No diversity data", ha="center", va="center", transform=ax_left.transAxes)
                ax_right.text(0.5, 0.5, "No diversity data", ha="center", va="center", transform=ax_right.transAxes)
            else:
                nnd_k5 = core_hamming.get("nnd_k5") if isinstance(core_hamming.get("nnd_k5"), dict) else None
                nnd_k1 = core_hamming.get("nnd_k1") if isinstance(core_hamming.get("nnd_k1"), dict) else None
                plot_block = nnd_k5 or nnd_k1
                note_lines: list[str] = []
                if isinstance(plot_block, dict):
                    baseline = plot_block.get("baseline") if isinstance(plot_block.get("baseline"), dict) else None
                    actual = plot_block.get("actual") if isinstance(plot_block.get("actual"), dict) else None
                    bins = None
                    if isinstance(baseline, dict) and isinstance(baseline.get("bins"), list):
                        bins = baseline.get("bins")
                    elif isinstance(actual, dict) and isinstance(actual.get("bins"), list):
                        bins = actual.get("bins")
                    if baseline and actual and bins:
                        base_ecdf = _ecdf_from_counts(bins, baseline.get("counts", []))
                        act_ecdf = _ecdf_from_counts(bins, actual.get("counts", []))
                        if base_ecdf and act_ecdf:
                            x_base, y_base = base_ecdf
                            x_act, y_act = act_ecdf
                            base_line = ax_left.step(
                                x_base,
                                y_base,
                                where="post",
                                color="#777777",
                                linewidth=1.2,
                                label="baseline",
                            )[0]
                            act_line = ax_left.step(
                                x_act,
                                y_act,
                                where="post",
                                color=hue,
                                linewidth=1.5,
                                label="actual",
                            )[0]
                            ax_left.fill_between(x_act, y_act, step="post", color=hue, alpha=0.12)
                            ax_left.set_xlim(0, max(x_act.max(), x_base.max()))
                            ax_left.set_ylim(0, 1.0)
                            ax_left.set_ylabel("Fraction <= d" if idx == 0 else "")
                            ax_left.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                            if idx == 0:
                                ax_left.legend(
                                    handles=[base_line, act_line],
                                    loc="lower right",
                                    frameon=False,
                                    fontsize=text_sizes["annotation"] * 0.8,
                                )
                if nnd_k5 is None and nnd_k1 is not None:
                    note_lines.append("k=1 (n<6)")
                if isinstance(plot_block, dict):
                    baseline = plot_block.get("baseline") if isinstance(plot_block.get("baseline"), dict) else None
                    actual = plot_block.get("actual") if isinstance(plot_block.get("actual"), dict) else None
                    baseline_med = baseline.get("median") if baseline is not None else None
                    actual_med = actual.get("median") if actual is not None else None
                    if baseline_med is not None and actual_med is not None:
                        note_lines.append(f"delta div {float(actual_med) - float(baseline_med):+.2f}")
                pairwise = core_hamming.get("pairwise") if isinstance(core_hamming.get("pairwise"), dict) else None
                if isinstance(pairwise, dict):
                    base_pair = pairwise.get("baseline") if isinstance(pairwise.get("baseline"), dict) else None
                    act_pair = pairwise.get("actual") if isinstance(pairwise.get("actual"), dict) else None
                    base_med = base_pair.get("median") if base_pair is not None else None
                    act_med = act_pair.get("median") if act_pair is not None else None
                    if base_med is not None and act_med is not None:
                        note_lines.append(f"delta pairwise {float(act_med) - float(base_med):+.2f}")
                overlap = diversity.get("overlap_actual_fraction") if isinstance(diversity, dict) else None
                swaps = diversity.get("overlap_actual_swaps") if isinstance(diversity, dict) else None
                if overlap is not None:
                    overlap_label = f"overlap {float(overlap) * 100:.1f}%"
                    if swaps is not None:
                        overlap_label = f"{overlap_label} (swaps={int(swaps)})"
                    note_lines.append(overlap_label)
                pool_size = diversity.get("candidate_pool_size") if isinstance(diversity, dict) else None
                shortlist_target = diversity.get("shortlist_target") if isinstance(diversity, dict) else None
                if pool_size is not None or shortlist_target is not None:
                    note_lines.append(
                        f"k(pool/target) {pool_size if pool_size is not None else '-'}"
                        f"/{shortlist_target if shortlist_target is not None else '-'}"
                    )
                score_block = diversity.get("score_quantiles") if isinstance(diversity, dict) else None
                if isinstance(score_block, dict):
                    base = score_block.get("baseline") if isinstance(score_block.get("baseline"), dict) else None
                    actual = score_block.get("actual") if isinstance(score_block.get("actual"), dict) else None
                    base_global = (
                        score_block.get("baseline_global")
                        if isinstance(score_block.get("baseline_global"), dict)
                        else None
                    )
                    if base is not None and actual is not None:
                        p10_delta = None
                        p50_delta = None
                        if base.get("p10") is not None and actual.get("p10") is not None:
                            p10_delta = float(actual.get("p10")) - float(base.get("p10"))
                        if base.get("p50") is not None and actual.get("p50") is not None:
                            p50_delta = float(actual.get("p50")) - float(base.get("p50"))
                        if p10_delta is not None or p50_delta is not None:
                            p10_text = f"{p10_delta:+.2f}" if p10_delta is not None else "-"
                            p50_text = f"{p50_delta:+.2f}" if p50_delta is not None else "-"
                            note_lines.append(f"delta score p10/med {p10_text} / {p50_text}")
                    if base_global is not None and actual is not None:
                        g10_delta = None
                        g50_delta = None
                        if base_global.get("p10") is not None and actual.get("p10") is not None:
                            g10_delta = float(actual.get("p10")) - float(base_global.get("p10"))
                        if base_global.get("p50") is not None and actual.get("p50") is not None:
                            g50_delta = float(actual.get("p50")) - float(base_global.get("p50"))
                        if g10_delta is not None or g50_delta is not None:
                            g10_text = f"{g10_delta:+.2f}" if g10_delta is not None else "-"
                            g50_text = f"{g50_delta:+.2f}" if g50_delta is not None else "-"
                            note_lines.append(f"delta score global {g10_text} / {g50_text}")
                if note_lines:
                    _add_anchored_box(
                        ax_left,
                        note_lines,
                        loc="upper left",
                        fontsize=text_sizes["annotation"] * 0.75,
                        alpha=0.9,
                        edgecolor="none",
                    )
                core_entropy = diversity.get("core_entropy") if isinstance(diversity, dict) else None
                if isinstance(core_entropy, dict):
                    base_entropy = core_entropy.get("baseline", {}).get("values", [])
                    actual_entropy = core_entropy.get("actual", {}).get("values", [])
                    if actual_entropy:
                        positions = np.arange(1, len(actual_entropy) + 1)
                        ax_right.bar(
                            positions,
                            actual_entropy,
                            color=hue,
                            alpha=0.65,
                            width=0.8,
                            label="actual",
                        )
                        if base_entropy:
                            base_vals = base_entropy[: len(actual_entropy)]
                            ax_right.plot(
                                positions,
                                base_vals,
                                color="#777777",
                                linewidth=1.2,
                                marker="o",
                                markersize=2.5,
                                label="baseline",
                            )
                        ax_right.set_ylabel("Entropy (bits)" if idx == 0 else "")
                        ax_right.set_xlim(0.5, len(actual_entropy) + 0.5)
                        ax_right.set_ylim(0.0, 2.0)
                        ax_right.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                        if idx == 0:
                            ax_right.legend(
                                loc="upper right",
                                frameon=False,
                                fontsize=text_sizes["annotation"] * 0.8,
                            )
                        base_sum = float(np.sum(base_entropy)) if base_entropy else None
                        act_sum = float(np.sum(actual_entropy))
                        if act_sum is not None:
                            sum_line = (
                                f"sumH {base_sum:.1f} -> {act_sum:.1f}"
                                if base_sum is not None
                                else f"sumH {act_sum:.1f}"
                            )
                            _add_anchored_box(
                                ax_right,
                                [sum_line],
                                loc="upper left",
                                fontsize=text_sizes["annotation"] * 0.7,
                                alpha=0.85,
                                edgecolor="none",
                            )
            ax_left.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))
            ax_right.grid(axis="y", alpha=float(style.get("grid_alpha", 0.2)))

        if axes_left:
            axes_left[0].set_title("Core k-NN distance", fontsize=subtitle_size, pad=title_pad)
            axes_right[0].set_title("Core positional entropy", fontsize=subtitle_size, pad=title_pad)
            axes_left[-1].set_xlabel("Hamming distance (k=5 neighbor)")
            axes_right[-1].set_xlabel("Core position")
            for ax in axes_left[:-1]:
                ax.tick_params(labelbottom=False)
            for ax in axes_right[:-1]:
                ax.tick_params(labelbottom=False)

        for ax in axes_left + axes_right:
            _apply_style(ax, style)
    return fig, axes_left, axes_right


def plot_stage_a_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if pools is None or pool_manifest is None:
        raise ValueError("Stage-A summary requires pool manifests; run stage-a build-pool first.")
    raw_style = style or {}
    style = _style(raw_style)
    style["seaborn_style"] = False
    if "figsize" not in raw_style:
        style["figsize"] = (11, 4)
    paths: list[Path] = []
    for input_name, pool_df in pools.items():
        entry = pool_manifest.entry_for(input_name)
        sampling = entry.stage_a_sampling
        if sampling is None:
            raise ValueError(f"Stage-A sampling metadata missing for input '{input_name}'.")
        fig, _, _ = _build_stage_a_strata_overview_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths.append(path)

        fig2, _, _, _ = _build_stage_a_yield_bias_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__yield_bias{out_path.suffix}"
        path2 = out_path.parent / fname
        fig2.savefig(path2, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig2)
        paths.append(path2)

        fig3, _, _ = _build_stage_a_diversity_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__diversity{out_path.suffix}"
        path3 = out_path.parent / fname
        fig3.savefig(path3, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig3)
        paths.append(path3)
    return paths


def plot_stage_b_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    library_builds_df: pd.DataFrame,
    library_members_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    cfg: dict | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if library_builds_df is None or library_builds_df.empty:
        raise ValueError("stage_b_summary requires library_builds.parquet data.")
    if library_members_df is None or library_members_df.empty:
        raise ValueError("stage_b_summary requires library_members.parquet data.")
    if composition_df is None or composition_df.empty:
        raise ValueError("stage_b_summary requires composition.parquet data.")
    required_build_cols = {"input_name", "plan_name", "library_index", "library_hash", "slack_bp"}
    missing = required_build_cols - set(library_builds_df.columns)
    if missing:
        raise ValueError(f"library_builds.parquet missing required columns: {sorted(missing)}")
    required_member_cols = {"input_name", "plan_name", "library_index", "tf", "tfbs"}
    missing = required_member_cols - set(library_members_df.columns)
    if missing:
        raise ValueError(f"library_members.parquet missing required columns: {sorted(missing)}")
    required_comp_cols = {"input_name", "plan_name", "solution_id", "tf", "tfbs"}
    missing = required_comp_cols - set(composition_df.columns)
    if missing:
        raise ValueError(f"composition.parquet missing required columns: {sorted(missing)}")
    raw_style = style or {}
    style = _style(raw_style)
    if "figsize" not in raw_style:
        style["figsize"] = (14, 7)
    paths: list[Path] = []

    for (input_name, plan_name), builds in library_builds_df.groupby(["input_name", "plan_name"]):
        members = library_members_df[
            (library_members_df["input_name"].astype(str) == str(input_name))
            & (library_members_df["plan_name"].astype(str) == str(plan_name))
        ]
        if members.empty:
            raise ValueError(f"library_members.parquet missing rows for {input_name}/{plan_name}.")
        metrics = members.groupby(["library_index", "library_hash"]).agg(
            library_size=("tfbs", "size"),
            unique_tfbs_count=("tfbs", pd.Series.nunique),
            total_bp=("tfbs", lambda x: int(sum(len(str(v)) for v in x))),
        )
        metrics = metrics.reset_index()
        merged = builds.merge(metrics, on=["library_index", "library_hash"], how="left")
        if "library_size_x" in merged.columns and "library_size_y" in merged.columns:
            merged["library_size"] = pd.to_numeric(merged["library_size_x"], errors="coerce").fillna(
                pd.to_numeric(merged["library_size_y"], errors="coerce")
            )
        elif "library_size_x" in merged.columns:
            merged["library_size"] = merged["library_size_x"]
        elif "library_size_y" in merged.columns:
            merged["library_size"] = merged["library_size_y"]

        offered_counts = members.groupby(["tf", "tfbs"])["library_index"].nunique().rename("offered_count")
        used_counts = (
            composition_df[
                (composition_df["input_name"].astype(str) == str(input_name))
                & (composition_df["plan_name"].astype(str) == str(plan_name))
            ]
            .groupby(["tf", "tfbs"])["solution_id"]
            .nunique()
            .rename("used_count")
        )
        offered_vs_used = pd.concat([offered_counts, used_counts], axis=1).fillna(0)
        offered_vs_used["ratio"] = offered_vs_used.apply(
            lambda row: float(row["used_count"]) / float(row["offered_count"]) if row["offered_count"] > 0 else 0.0,
            axis=1,
        )

        fig, axes = plt.subplots(2, 3, figsize=style["figsize"])
        ax_slack, ax_size, ax_unique, ax_bp, ax_scatter, ax_ratio = axes.flatten()

        slack = pd.to_numeric(merged["slack_bp"], errors="coerce").dropna()
        if slack.empty:
            ax_slack.text(0.5, 0.5, "No slack data", ha="center", va="center", transform=ax_slack.transAxes)
        else:
            ax_slack.hist(slack, bins="auto", color="#4c78a8", alpha=0.8)
            ax_slack.axvline(0.0, color="#e15759", linestyle="--", linewidth=1)
        ax_slack.set_title("Slack distribution")
        ax_slack.set_xlabel("Slack bp")
        ax_slack.set_ylabel("Libraries")

        sizes = pd.to_numeric(merged["library_size"], errors="coerce").dropna()
        if sizes.empty:
            ax_size.text(0.5, 0.5, "No size data", ha="center", va="center", transform=ax_size.transAxes)
        else:
            ax_size.hist(sizes, bins="auto", color="#59a14f", alpha=0.8)
        ax_size.set_title("Library size")
        ax_size.set_xlabel("TFBS per library")
        ax_size.set_ylabel("Libraries")

        uniques = pd.to_numeric(merged["unique_tfbs_count"], errors="coerce").dropna()
        if uniques.empty:
            ax_unique.text(0.5, 0.5, "No TFBS counts", ha="center", va="center", transform=ax_unique.transAxes)
        else:
            ax_unique.hist(uniques, bins="auto", color="#f28e2b", alpha=0.8)
        ax_unique.set_title("Unique TFBS count")
        ax_unique.set_xlabel("Unique TFBS per library")
        ax_unique.set_ylabel("Libraries")

        total_bp = pd.to_numeric(merged["total_bp"], errors="coerce").dropna()
        if total_bp.empty:
            ax_bp.text(0.5, 0.5, "No bp data", ha="center", va="center", transform=ax_bp.transAxes)
        else:
            ax_bp.hist(total_bp, bins="auto", color="#edc949", alpha=0.8)
        ax_bp.set_title("Total TFBS bp")
        ax_bp.set_xlabel("Total TFBS bp")
        ax_bp.set_ylabel("Libraries")

        if offered_vs_used.empty:
            ax_scatter.text(0.5, 0.5, "No offered/used data", ha="center", va="center", transform=ax_scatter.transAxes)
        else:
            x = offered_vs_used["offered_count"].to_numpy(dtype=float)
            y = offered_vs_used["used_count"].to_numpy(dtype=float)
            if len(offered_vs_used) > 200:
                ax_scatter.hexbin(x, y, gridsize=35, cmap="Blues", mincnt=1)
            else:
                ax_scatter.scatter(x, y, alpha=0.6, s=18, color="#4c78a8")
        ax_scatter.set_title("Offered vs used (TFBS)")
        ax_scatter.set_xlabel("Offered count")
        ax_scatter.set_ylabel("Used count")

        ratios = pd.to_numeric(offered_vs_used["ratio"], errors="coerce").dropna()
        if ratios.empty:
            ax_ratio.text(0.5, 0.5, "No ratio data", ha="center", va="center", transform=ax_ratio.transAxes)
        else:
            ax_ratio.hist(ratios, bins="auto", color="#af7aa1", alpha=0.8)
        ax_ratio.set_title("Used / offered ratio")
        ax_ratio.set_xlabel("Used / offered")
        ax_ratio.set_ylabel("TFBS count")

        for ax in axes.flatten():
            _apply_style(ax, style)
        fig.suptitle(f"Stage-B summary — {input_name}/{plan_name}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.92])
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__{_safe_filename(plan_name)}{out_path.suffix}"
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
    "placement_map": set(),
    "tfbs_usage": set(),
    "run_health": set(),
    "stage_a_summary": set(),
    "stage_b_summary": set(),
}


def _filter_kwargs(name: str, kwargs: dict) -> dict:
    allowed = _ALLOWED_OPTIONS.get(name)
    if allowed is None:
        raise ValueError(f"Unknown plot name: {name}")
    unknown = [
        k
        for k in list(kwargs.keys())
        if k not in allowed and k not in {"dims", "palette", "palette_no_repeat", "style"}
    ]
    if unknown:
        raise ValueError(f"Unknown options for plot '{name}': {unknown}")
    return kwargs


def _plot_required_columns(selected: Iterable[str], options: Dict[str, Dict[str, object]]) -> list[str]:
    return []


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
    absolute: bool = False,
) -> None:
    plots_cfg = root_cfg.plots
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_dir = _ensure_out_dir(plots_cfg, cfg_path, run_root)
    plot_format = plots_cfg.format if plots_cfg and getattr(plots_cfg, "format", None) else "png"
    default_list = (
        plots_cfg.default if (plots_cfg and plots_cfg.default) else ["placement_map", "tfbs_usage", "run_health"]
    )
    selected = [p.strip() for p in (only.split(",") if only else default_list)]
    options = plots_cfg.options if plots_cfg else {}
    global_style = plots_cfg.style if plots_cfg else {}
    required_sources = _plot_required_sources(selected)
    cols = _plot_required_columns(selected, options)
    max_rows = plots_cfg.sample_rows if plots_cfg else None
    df = pd.DataFrame()
    src_label = "none"
    row_count = 0
    attempts_df: pd.DataFrame | None = None
    events_df: pd.DataFrame | None = None
    composition_df: pd.DataFrame | None = None
    library_builds_df: pd.DataFrame | None = None
    library_members_df: pd.DataFrame | None = None
    cfg_effective: dict | None = None

    if "outputs" in required_sources:
        df, src_label = load_records_from_config(root_cfg, cfg_path, columns=cols, max_rows=max_rows)
        src_label = _format_source_label(src_label, run_root, absolute)
        row_count = len(df)
    if "composition" in required_sources:
        composition_df = _load_composition(run_root)
        if row_count == 0:
            row_count = len(composition_df)
            src_label = _format_source_label(
                f"composition:{run_root / 'outputs' / 'tables' / 'composition.parquet'}", run_root, absolute
            )
    if "libraries" in required_sources:
        library_builds_df, library_members_df = _load_libraries(run_root)
        if row_count == 0:
            row_count = len(library_members_df)
            src_label = _format_source_label(f"libraries:{run_root / 'outputs' / 'libraries'}", run_root, absolute)
    if "config" in required_sources:
        cfg_effective = _load_effective_config(run_root)
        if row_count == 0:
            row_count = 1
            src_label = _format_source_label(
                f"config:{run_root / 'outputs' / 'meta' / 'effective_config.json'}", run_root, absolute
            )
    if "attempts" in required_sources:
        attempts_df = _load_attempts(run_root)
        if row_count == 0:
            row_count = len(attempts_df)
            src_label = _format_source_label(
                f"attempts:{run_root / 'outputs' / 'tables' / 'attempts.parquet'}", run_root, absolute
            )
        events_path = run_root / "outputs" / "meta" / "events.jsonl"
        if events_path.exists():
            events_df = _load_events(run_root)
            if row_count == 0:
                row_count = len(events_df)
                src_label = _format_source_label(f"events:{events_path}", run_root, absolute)
    pools: dict[str, pd.DataFrame] | None = None
    pool_manifest: TFBSPoolArtifact | None = None
    if "pools" in required_sources:
        pool_manifest, pools = _load_stage_a_pools(run_root)
        if row_count == 0:
            row_count = sum(len(pool_df) for pool_df in pools.values())
            src_label = _format_source_label(f"pools:{run_root / 'outputs' / 'pools'}", run_root, absolute)
    if "tfbs_usage" in selected and pools is None:
        pool_manifest, pools = _maybe_load_stage_a_pools(run_root)
    if "tfbs_usage" in selected and library_members_df is None:
        libs = _maybe_load_libraries(run_root)
        if libs is not None:
            library_builds_df, library_members_df = libs

    out_label = _format_plot_path(out_dir, run_root, absolute)
    _console.print(
        make_panel(
            f"source: {src_label} • rows: {row_count:,}\nOutput: {out_label}",
            title="DenseGen plotting",
        )
    )
    summary = make_table("plot", "saved to", "status")
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
            if name == "placement_map":
                result = fn(df, out_path, style=style, composition_df=composition_df, cfg=cfg_effective, **kwargs)
            elif name == "tfbs_usage":
                result = fn(
                    df,
                    out_path,
                    style=style,
                    composition_df=composition_df,
                    pools=pools,
                    library_members_df=library_members_df,
                    **kwargs,
                )
            elif name == "run_health":
                result = fn(df, out_path, style=style, attempts_df=attempts_df, events_df=events_df, **kwargs)
            elif name == "stage_a_summary":
                result = fn(df, out_path, style=style, pools=pools, pool_manifest=pool_manifest, **kwargs)
            elif name == "stage_b_summary":
                result = fn(
                    df,
                    out_path,
                    style=style,
                    library_builds_df=library_builds_df,
                    library_members_df=library_members_df,
                    composition_df=composition_df,
                    cfg=cfg_effective,
                    **kwargs,
                )
            else:
                result = fn(df, out_path, style=style, **kwargs)
            if result is None:
                paths = [out_path]
            elif isinstance(result, (list, tuple, set)):
                paths = [Path(p) for p in result]
            else:
                paths = [Path(result)]
            saved_label = _format_plot_path(paths[0], run_root, absolute)
            if len(paths) > 1:
                saved_label = f"{saved_label} (+{len(paths) - 1})"
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
