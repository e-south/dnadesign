"""Configured OPAL plot generation for DenseGen axis probe run roots."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dnadesign.opal import run_campaign_plots

from .artifacts import ProbeArtifactLayout


def generate_probe_campaign_plots(run_root: Path, *, round_selector: str = "all") -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    if not layout.run_root.exists():
        raise RuntimeError(f"run root not found: {layout.run_root}")
    mpl_config_dir = layout.run_root / ".opal" / "mpl"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    results: list[dict[str, Any]] = []
    for config_path in _campaign_config_paths(layout):
        result = run_campaign_plots(config_path, round_selector=round_selector)
        result["run_key"] = config_path.parents[1].name
        results.append(result)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.plot.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "round_selector": round_selector,
        "mpl_config_dir": str(mpl_config_dir),
        "campaign_count": len(results),
        "any_fail": any(bool(row.get("any_fail")) for row in results),
        "campaigns": results,
    }


def _campaign_config_paths(layout: ProbeArtifactLayout) -> list[Path]:
    if not layout.scratch_campaigns_dir.exists():
        return []
    return sorted(layout.scratch_campaigns_dir.glob("*/configs/campaign.yaml"))
