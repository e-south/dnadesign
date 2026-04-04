"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/publish.py

Publish YIU v4 bundles and BaseRender-ready view contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.bundle_models import (
    PayloadBundleManifest,
    PayloadViewEntry,
    PayloadVisualInventory,
    payload_summary_from_normalized,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.spec_models import YiuPayloadRenderingSpec
from dnadesign.cruncher.yiu.view_contracts import (
    build_assembled_payload_view_contract,
    build_payload_view_contract,
    build_split_payload_view_rows,
    build_yiu_style_overrides,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    return path


def _relative_to_bundle(bundle_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(bundle_dir.resolve()))


def _render_job_payload(*, entry: PayloadViewEntry) -> dict[str, object]:
    return {
        "version": 3,
        "results_root": "..",
        "input": {
            "kind": entry.input_kind,
            "path": f"../{entry.view_contract_path}",
            "adapter": {"kind": entry.contract_kind},
            "alphabet": "iupac_dna",
        },
        "render": {
            "renderer": entry.renderer_kind,
            "style": {"preset": entry.style_preset, "overrides": entry.style_overrides},
        },
        "outputs": [{"kind": "images", "path": f"../debug/rerenders/{entry.view_id}.pdf", "fmt": "pdf"}],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }


def _published_artifacts(spec: YiuPayloadRenderingSpec) -> dict[str, str]:
    artifacts = {
        "normalized_payload": "normalized_payload.json",
        "bundle_manifest": "bundle_manifest.json",
        "visual_inventory": "visual_inventory.json",
        "payload_view": "payload_view.json",
        "split_payload_view": "split_payload_view.json",
        "assembled_payload_view": "assembled_payload_view.json",
        "payload_views_pdf": "payload_views.pdf",
    }
    if spec.output.published_plot_path is not None:
        artifacts["published_plot_pdf"] = str(spec.output.published_plot_path)
    return artifacts


def _build_view_entries(
    *,
    bundle_dir: Path,
    normalized: NormalizedPayload,
    payload_path: Path,
    split_path: Path,
    assembled_path: Path,
    combined_render_path: Path,
) -> list[PayloadViewEntry]:
    return [
        PayloadViewEntry(
            view_id="payload",
            contract_kind="yiu_payload_visual_v1",
            schema_version=1,
            input_kind="json",
            view_contract_path=_relative_to_bundle(bundle_dir, payload_path),
            render_artifact_path=_relative_to_bundle(bundle_dir, combined_render_path),
            renderer_kind="nucleotide_evidence_map",
            style_overrides=build_yiu_style_overrides("payload"),
            motif_layers_required=normalized.motif_context.effective,
        ),
        PayloadViewEntry(
            view_id="split_payload",
            contract_kind="sequence_evidence_map_v1",
            schema_version=1,
            input_kind="jsonl",
            view_contract_path=_relative_to_bundle(bundle_dir, split_path),
            render_artifact_path=_relative_to_bundle(bundle_dir, combined_render_path),
            renderer_kind="sequence_rows",
            style_overrides=build_yiu_style_overrides("split_payload"),
            motif_layers_required=False,
        ),
        PayloadViewEntry(
            view_id="assembled_payload",
            contract_kind="sequence_evidence_map_v1",
            schema_version=1,
            input_kind="json",
            view_contract_path=_relative_to_bundle(bundle_dir, assembled_path),
            render_artifact_path=_relative_to_bundle(bundle_dir, combined_render_path),
            renderer_kind="nucleotide_evidence_map",
            style_overrides=build_yiu_style_overrides("assembled_payload"),
            motif_layers_required=False,
        ),
    ]


def publish_payload_bundle(
    *,
    spec: YiuPayloadRenderingSpec,
    normalized: NormalizedPayload,
    bundle_dir: Path,
) -> tuple[PayloadBundleManifest, PayloadVisualInventory]:
    bundle_dir = bundle_dir.resolve()
    render_jobs_dir = bundle_dir / "baserender_jobs"
    combined_render_path = bundle_dir / "payload_views.pdf"

    payload_path = bundle_dir / "payload_view.json"
    split_path = bundle_dir / "split_payload_view.json"
    assembled_path = bundle_dir / "assembled_payload_view.json"
    normalized_path = bundle_dir / "normalized_payload.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    inventory_path = bundle_dir / "visual_inventory.json"

    payload_contract = build_payload_view_contract(normalized)
    _write_json(payload_path, payload_contract)
    _write_jsonl(split_path, build_split_payload_view_rows(normalized))
    _write_json(assembled_path, build_assembled_payload_view_contract(normalized))

    published_artifacts = _published_artifacts(spec)

    normalized_with_artifacts = normalized.model_copy(update={"published_artifacts": published_artifacts})
    _write_json(normalized_path, normalized_with_artifacts.model_dump(mode="json"))

    view_entries = _build_view_entries(
        bundle_dir=bundle_dir,
        normalized=normalized,
        payload_path=payload_path,
        split_path=split_path,
        assembled_path=assembled_path,
        combined_render_path=combined_render_path,
    )

    if spec.output.emit_render_jobs_debug:
        for entry in view_entries:
            render_jobs_dir.mkdir(parents=True, exist_ok=True)
            job_path = render_jobs_dir / f"{entry.view_id}.job.yaml"
            job_path.write_text(yaml.safe_dump(_render_job_payload(entry=entry), sort_keys=False), encoding="utf-8")

    inventory = PayloadVisualInventory(
        spec_name=spec.yiu.name,
        input_kind=normalized.input_kind,
        view_count=len(view_entries),
        render_count=0,
        render_status="not_requested",
        composite_render_artifact_path=_relative_to_bundle(bundle_dir, combined_render_path),
        published_plot_artifact_path=None
        if spec.output.published_plot_path is None
        else str(spec.output.published_plot_path),
        pwm_effective=normalized.motif_context.effective,
        payload_view_requires_motif_layers=normalized.motif_context.effective,
        views=view_entries,
    )
    manifest = PayloadBundleManifest(
        spec_name=spec.yiu.name,
        provenance=normalized.source_provenance,
        payload_view_requires_motif_layers=normalized.motif_context.effective,
        view_contracts=view_entries,
        composite_render_artifact_path=_relative_to_bundle(bundle_dir, combined_render_path),
        published_plot_artifact_path=None
        if spec.output.published_plot_path is None
        else str(spec.output.published_plot_path),
        render_status=inventory.render_status,
        **payload_summary_from_normalized(normalized),
    )
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    _write_json(inventory_path, inventory.model_dump(mode="json"))
    return manifest, inventory
