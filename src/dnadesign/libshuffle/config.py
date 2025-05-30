"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/config.py

Loads and validates the libshuffle configuration from a YAML file.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml


@dataclass
class LatentThreshold:
    type: Literal["iqr", "percentile"] = "iqr"
    factor: float = 1.5


@dataclass
class SelectionConfig:
    method: Literal["max_min_latent"] = "max_min_latent"
    latent_threshold: LatentThreshold = field(default_factory=LatentThreshold)


@dataclass
class ScatterConfig:
    x: str = "mean_cosine"
    y: str = "log1p_euclidean"
    low_alpha: float = 0.25
    high_alpha: float = 0.45
    star_size: int = 100
    threshold_line: bool = True
    threshold: LatentThreshold = field(default_factory=LatentThreshold)
    colors: Dict[str, Any] = field(
        default_factory=lambda: {
            "base": "purple",
            "literal_drop": {"jaccard": "orange", "levenshtein": "red", "literal": "red"},
            "winner": "limegreen",
            "threshold_line": "lightgray",
            "cluster_drop": "gray",
        }
    )
    annotate_winner: bool = True
    annotate_ids: bool = True
    figsize: List[int] = field(default_factory=lambda: [8, 6])
    dpi: int = 300


@dataclass
class KDEConfig:
    figsize: List[int] = field(default_factory=lambda: [8, 5])
    dpi: int = 300


@dataclass
class PairplotConfig:
    figsize: List[int] = field(default_factory=lambda: [10, 10])
    dpi: int = 300


@dataclass
class HitzoneConfig:
    figsize: List[int] = field(default_factory=lambda: [10, 4])
    dpi: int = 300


@dataclass
class PlotConfig:
    scatter: ScatterConfig = field(default_factory=ScatterConfig)
    kde: KDEConfig = field(default_factory=KDEConfig)
    pairplot: PairplotConfig = field(default_factory=PairplotConfig)
    hitzone: HitzoneConfig = field(default_factory=HitzoneConfig)


@dataclass
class LibShuffleConfig:
    input_pt_path: Path
    output_dir_prefix: str

    subsample_size: int = 16
    num_draws: int = 1000
    random_seed: int = 42
    with_replacement: bool = False
    max_attempts_per_draw: int = 10

    evo2_metric_type: Literal["l2", "log1p_l2", "cosine"] = "cosine"

    literal_filters: List[Literal["jaccard", "levenshtein"]] = field(default_factory=lambda: ["jaccard", "levenshtein"])
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    billboard_core_metrics: List[str] = field(default_factory=list)

    save_selected: bool = False
    save_sublibraries: List[str] = field(default_factory=list)

    plot: PlotConfig = field(default_factory=PlotConfig)
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Any) -> "LibShuffleConfig":
        # read YAML
        p = Path(path)
        if p.exists():
            raw_all = yaml.safe_load(p.read_text())
        else:
            # search upward
            for anc in Path(__file__).resolve().parents:
                candidate = anc / path
                if candidate.exists():
                    raw_all = yaml.safe_load(candidate.read_text())
                    break
            else:
                raise FileNotFoundError(f"Config file not found: {path}")

        if "libshuffle" not in raw_all:
            raise KeyError("Top-level 'libshuffle' key missing in config.")
        raw = raw_all["libshuffle"]

        # required keys
        for k in ["input_pt_path", "output_dir_prefix", "billboard_core_metrics", "selection"]:
            if k not in raw:
                raise KeyError(f"Missing required config key: {k}")

        # default instance for fallback values
        default = cls(input_pt_path=Path("."), output_dir_prefix=".")
        cfg: Dict[str, Any] = {
            "input_pt_path": Path(raw["input_pt_path"]),
            "output_dir_prefix": raw["output_dir_prefix"],
            "subsample_size": raw.get("subsample_size", default.subsample_size),
            "num_draws": raw.get("num_draws", default.num_draws),
            "random_seed": raw.get("random_seed", default.random_seed),
            "with_replacement": raw.get("with_replacement", default.with_replacement),
            "max_attempts_per_draw": raw.get("max_attempts_per_draw", default.max_attempts_per_draw),
            "evo2_metric_type": raw.get("evo2_metric_type", default.evo2_metric_type),
            "literal_filters": raw.get("literal_filters", default.literal_filters),
            "billboard_core_metrics": raw["billboard_core_metrics"],
            "save_selected": raw.get("save_selected", default.save_selected),
            "save_sublibraries": raw.get("save_sublibraries", default.save_sublibraries),
        }

        # selection config
        sel = raw["selection"]
        cfg["selection"] = SelectionConfig(
            method=sel["method"], latent_threshold=LatentThreshold(**sel.get("latent_threshold", {}))
        )

        # helper to deep merge nested dicts
        def deep_merge(base: dict, override: dict) -> dict:
            merged = base.copy()
            for k, v in override.items():
                if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                    merged[k] = deep_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        # plot subconfigs
        pr = raw.get("plot", {})

        # scatter
        sd = default.plot.scatter
        rd = pr.get("scatter", {})
        colors = deep_merge(sd.colors, rd.get("colors", {}))
        scatter = ScatterConfig(
            x=rd.get("x", sd.x),
            y=rd.get("y", sd.y),
            low_alpha=rd.get("low_alpha", sd.low_alpha),
            high_alpha=rd.get("high_alpha", sd.high_alpha),
            star_size=rd.get("star_size", sd.star_size),
            threshold_line=rd.get("threshold_line", sd.threshold_line),
            threshold=LatentThreshold(
                **rd.get("threshold", {"type": sd.threshold.type, "factor": sd.threshold.factor})
            ),
            colors=colors,
            annotate_winner=rd.get("annotate_winner", sd.annotate_winner),
            annotate_ids=rd.get("annotate_ids", sd.annotate_ids),
            figsize=rd.get("figsize", sd.figsize),
            dpi=rd.get("dpi", sd.dpi),
        )

        # kde
        kd, rk = default.plot.kde, pr.get("kde", {})
        kde = KDEConfig(figsize=rk.get("figsize", kd.figsize), dpi=rk.get("dpi", kd.dpi))

        # pairplot
        pd_, rp = default.plot.pairplot, pr.get("pairplot", {})
        pairplot = PairplotConfig(figsize=rp.get("figsize", pd_.figsize), dpi=rp.get("dpi", pd_.dpi))

        # hitzone
        hz, rh = default.plot.hitzone, pr.get("hitzone", {})
        hitzone = HitzoneConfig(figsize=rh.get("figsize", hz.figsize), dpi=rh.get("dpi", hz.dpi))

        cfg["plot"] = PlotConfig(scatter=scatter, kde=kde, pairplot=pairplot, hitzone=hitzone)

        inst = cls(**cfg)
        inst._raw = raw
        return inst
