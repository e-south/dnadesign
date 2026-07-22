"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/plotting.py

Configured OPAL plot generation for DenseGen axis probe run roots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from dnadesign.opal import load_plot_artifact_manifest, load_plot_manifest_index, run_campaign_plots

from ..core.artifacts import ProbeArtifactLayout
from ..runtime.plan import build_plan
from ..runtime.scratch import write_campaign_plot_config


def generate_probe_campaign_plots(
    run_root: Path,
    *,
    round_selector: str = "all",
    name: str | None = None,
    tags: Sequence[str] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    if not layout.run_root.exists():
        raise RuntimeError(f"run root not found: {layout.run_root}")
    mpl_config_dir = layout.run_root / ".opal" / "mpl"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    refreshed_plot_config_count = refresh_probe_campaign_plot_configs(layout.run_root)

    results: list[dict[str, Any]] = []
    for config_path in _campaign_config_paths(layout):
        result = run_campaign_plots(
            config_path,
            round_selector=round_selector,
            name=name,
            tags=tags,
            quiet=quiet,
        )
        result["run_key"] = config_path.parents[1].name
        result["plot_contract"] = _assert_opal_plot_contract(Path(str(result["plot_manifest_path"])))
        results.append(result)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.plot.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "round_selector": round_selector,
        "name": name,
        "tags": [str(tag) for tag in (tags or [])],
        "mpl_config_dir": str(mpl_config_dir),
        "refreshed_plot_config_count": refreshed_plot_config_count,
        "campaign_count": len(results),
        "any_fail": any(bool(row.get("any_fail")) for row in results),
        "campaigns": results,
    }


def refresh_probe_campaign_plot_configs(run_root: Path) -> int:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    if not layout.probe_plan_path.exists():
        raise RuntimeError(f"probe plan missing for plot-config refresh: {layout.probe_plan_path}")
    plan_record = json.loads(layout.probe_plan_path.read_text(encoding="utf-8"))
    plan_payload = plan_record.get("plan") if isinstance(plan_record.get("plan"), dict) else {}
    if not plan_payload:
        raise RuntimeError(f"probe plan payload missing for plot-config refresh: {layout.probe_plan_path}")
    plan = build_plan(
        run_root=layout.run_root,
        initial_label_count=int(plan_payload.get("initial_label_count") or 0),
        selection_k=int(plan_payload.get("selection_k") or 0),
        seed=int(plan_payload.get("seed") or 0),
        rounds=int(plan_payload.get("rounds") or 1),
        gate=str(plan_payload.get("gate") or "all"),
        splits=tuple(plan_payload.get("split_ids") or ()),
        apply=True,
        stop_after=str(plan_payload.get("stop_after") or "status"),
        suite_id=str(plan_payload.get("suite_id") or ""),
        max_x_matrix_gib=plan_payload.get("max_x_matrix_gib"),
        score_batch_size=plan_payload.get("score_batch_size"),
        active_label_families=tuple(plan_payload.get("active_label_families") or ()),
    )
    expected_count = int(plan_payload.get("planned_runs") or 0)
    if expected_count and len(plan.runs) != expected_count:
        raise RuntimeError(
            f"probe plan would refresh {len(plan.runs)} plot configs, but expected {expected_count}: "
            f"{layout.probe_plan_path}"
        )
    refreshed = 0
    for run in plan.runs:
        if not run.config_path.exists():
            raise RuntimeError(f"campaign config missing for plot-config refresh: {run.config_path}")
        write_campaign_plot_config(run)
        refreshed += 1
    return refreshed


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
