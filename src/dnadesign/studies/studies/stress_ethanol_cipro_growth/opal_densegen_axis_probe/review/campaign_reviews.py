"""Bridge study probe runs to OPAL campaign review artifacts."""

from __future__ import annotations

from typing import Any, Mapping

from dnadesign.opal import build_campaign_review

from ..artifacts import ProbeArtifactLayout


def _build_campaign_reviews(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
    include_plots: bool,
) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in metrics_payload.get("runs") or []:
        if not isinstance(row, Mapping):
            raise RuntimeError("metrics runs entries must be objects")
        run_key = str(row.get("run_key") or "").strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        config_path = layout.campaign_config_path(run_key)
        if not config_path.exists():
            raise RuntimeError(f"scratch campaign config missing for scored run {run_key}: {config_path}")
        run_id = str(row.get("run_id") or "").strip() or None
        round_value = row.get("as_of_round")
        round_selector = str(int(round_value)) if round_value is not None else "latest"
        result = build_campaign_review(
            config_path,
            round_selector=round_selector,
            run_id=run_id,
            include_plots=include_plots,
        )
        reviewed.append(
            {
                "run_key": run_key,
                "status": "written",
                "config_path": str(config_path),
                "review_path": str(result.review_path),
                "index_path": str(result.index_path),
                "manifest_path": str(result.manifest_path),
                "plot_paths": [str(path) for path in result.plot_paths],
                "round_index": result.manifest["review_scope"]["round_index"],
                "run_id": result.manifest["review_scope"]["run_id"],
            }
        )
    return reviewed
