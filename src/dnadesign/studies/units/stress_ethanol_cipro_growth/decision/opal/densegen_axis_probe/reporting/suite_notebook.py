"""OPAL campaign-set notebook generation across DenseGen probe seed roots."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from dnadesign.opal import (
    build_campaign_set_notebook_view_model,
    materialize_campaign_set_collection_visuals,
    render_campaign_set_notebook,
    smoke_check_notebook,
)

from ..core.constants import DEFAULT_SUITE_ID
from ..runtime.execution import probe_campaign_collection_manifest_payload


def build_probe_suite_opal_notebook(
    run_roots: Sequence[Path],
    *,
    out_dir: Path,
    round_selector: str = "all",
) -> dict[str, Any]:
    """Write a suite-scope OPAL notebook so seed-replicate intervals are visible."""

    roots = [Path(root).resolve() for root in run_roots]
    config_paths = _campaign_config_paths(roots)
    if not config_paths:
        raise RuntimeError("suite OPAL notebook requires at least one campaign config.")
    out_dir.mkdir(parents=True, exist_ok=True)
    collection_path = out_dir / "campaign_collection.json"
    notebook_path = out_dir / "opal_campaign_set_analysis.py"
    visual_dir = out_dir / "collection_visuals"
    relationships = [_positive_null_relationship()]
    collection = probe_campaign_collection_manifest_payload(
        collection_id=f"{DEFAULT_SUITE_ID}_all_seed_replicates",
        relationships=relationships,
    )
    collection_path.write_text(json.dumps(collection, indent=2, sort_keys=True), encoding="utf-8")
    view_model = build_campaign_set_notebook_view_model(
        config_paths,
        round_selector=round_selector,
        collection_manifest_path=collection_path,
    )
    visual_index = materialize_campaign_set_collection_visuals(
        view_model["campaigns"],
        collection=view_model["collection"],
        output_dir=visual_dir,
    )
    visual_index_path = visual_dir / "collection_visual_manifest.json"
    notebook_path.write_text(
        render_campaign_set_notebook(
            config_paths,
            round_selector=round_selector,
            collection_manifest_path=collection_path,
            collection_visual_index_path=visual_index_path,
        ),
        encoding="utf-8",
    )
    smoke_check_notebook(notebook_path, run_marimo_check=True)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.suite_opal_notebook.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_roots": [str(root) for root in roots],
        "campaign_count": len(config_paths),
        "round_selector": str(round_selector),
        "collection_manifest": str(collection_path),
        "collection_visual_index": str(visual_index_path),
        "collection_visual_count": int(visual_index.get("visual_count") or 0),
        "comparison_set_count": int(visual_index.get("comparison_set_count") or 0),
        "notebook": str(notebook_path),
    }


def _campaign_config_paths(run_roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in run_roots:
        root_path = Path(root)
        if not root_path.exists():
            raise RuntimeError(f"suite run root not found: {root_path}")
        paths.extend(sorted((root_path / "scratch_campaigns").glob("*/configs/campaign.yaml")))
    resolved = [str(path.resolve()) for path in paths]
    duplicates = sorted({path for path in resolved if resolved.count(path) > 1})
    if duplicates:
        raise RuntimeError("suite OPAL notebook got duplicate campaign configs: " + ", ".join(duplicates))
    return paths


def _positive_null_relationship() -> dict[str, Any]:
    return {
        "id": "positive_vs_null",
        "kind": "control_pair",
        "label": "Positive vs null oracle control",
        "role_dimension": "label_oracle_kind",
        "left_role": "positive",
        "right_role": "null",
        "match_on": ["target", "label_family_id", "label_split_id", "seed"],
        "replicate_on": ["seed"],
    }
