"""Configured OPAL plot generation for DenseGen axis probe run roots."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dnadesign.opal import load_plot_artifact_manifest, load_plot_manifest_index, run_campaign_plots

from .artifacts import ProbeArtifactLayout


def generate_probe_campaign_plots(
    run_root: Path, *, round_selector: str = "all", quiet: bool = False
) -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    if not layout.run_root.exists():
        raise RuntimeError(f"run root not found: {layout.run_root}")
    mpl_config_dir = layout.run_root / ".opal" / "mpl"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    results: list[dict[str, Any]] = []
    for config_path in _campaign_config_paths(layout):
        result = run_campaign_plots(config_path, round_selector=round_selector, quiet=quiet)
        result["run_key"] = config_path.parents[1].name
        result["plot_contract"] = _assert_opal_plot_contract(Path(str(result["plot_manifest_path"])))
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


def _assert_opal_plot_contract(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        raise RuntimeError(f"OPAL plot manifest index was not written: {index_path}")
    index = load_plot_manifest_index(index_path)
    manifests = index.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError(f"OPAL plot manifest index has no plot manifests: {index_path}")
    status_counts: dict[str, int] = {}
    for row in manifests:
        if not isinstance(row, dict):
            raise RuntimeError(f"OPAL plot manifest index contains a non-object row: {index_path}")
        manifest_path = Path(str(row.get("manifest_path") or ""))
        if not manifest_path.exists():
            raise RuntimeError(f"OPAL plot artifact manifest is missing: {manifest_path}")
        manifest = load_plot_artifact_manifest(manifest_path)
        for key in ("name", "kind", "status", "rounds", "outputs", "metadata"):
            if key not in manifest:
                raise RuntimeError(f"OPAL plot artifact manifest missing {key!r}: {manifest_path}")
        status = str(manifest.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.plot_contract.v1",
        "status": "ok",
        "plot_count": len(manifests),
        "status_counts": status_counts,
        "manifest_index": str(index_path),
    }
