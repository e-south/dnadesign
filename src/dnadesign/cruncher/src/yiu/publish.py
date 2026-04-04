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
from typing import Any, Sequence

import yaml

from dnadesign.baserender import cruncher_showcase_style_overrides
from dnadesign.contracts.visual import SequenceEvidenceMapV1, YiuPayloadVisualV1
from dnadesign.contracts.visual.yiu_payload_visual_v1 import (
    YiuPayloadDisplayV1,
    YiuPayloadJunctionV1,
    YiuPayloadMismatchV1,
    YiuPayloadMotifLayerV1,
)
from dnadesign.cruncher.yiu.bsmbi import assembled_payload_aligned_complement_3to5, build_split_fragment_display_specs
from dnadesign.cruncher.yiu.bundle_models import (
    PayloadBundleManifest,
    PayloadViewEntry,
    PayloadVisualInventory,
    payload_summary_from_normalized,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.spec_models import YiuPayloadRenderingSpec

_YIU_ROW_LABELS = {
    "primary": "Selected payload",
    "complement": "Selected complement",
}

_MOTIF_PASTEL_PALETTE: tuple[str, ...] = (
    "#67BFA5",
    "#D883A4",
    "#7BA4D9",
    "#C08A56",
    "#5DA79F",
    "#D1B06C",
    "#74C0CB",
    "#86A5D8",
    "#9BC47B",
    "#C9B082",
    "#D68AA7",
    "#D9A78A",
)


def _motif_palette_token(motif_instance_id: str) -> str:
    return f"motif:{motif_instance_id}"


def _motif_palette_entries(motif_layers: Sequence[YiuPayloadMotifLayerV1]) -> dict[str, str]:
    return {
        _motif_palette_token(motif.motif_instance_id): _MOTIF_PASTEL_PALETTE[index % len(_MOTIF_PASTEL_PALETTE)]
        for index, motif in enumerate(motif_layers)
    }


def _yiu_style_overrides(view_id: str, *, motif_layers: Sequence[YiuPayloadMotifLayerV1] = ()) -> dict[str, object]:
    if view_id == "payload":
        base = dict(cruncher_showcase_style_overrides())
        base["palette"] = {
            **dict(base.get("palette", {})),
            **_motif_palette_entries(motif_layers),
        }
        base["padding_x"] = 42.0
        base["padding_y"] = 24.0
        base["font_size_seq"] = 13
        base["font_size_label"] = 11
        base["legend"] = False
        base["connectors"] = True
        base["connector_width"] = 1.1
        base["connector_alpha"] = 0.78
        base["connector_dash"] = ()
        return base

    base: dict[str, object] = {
        "figure_scale": 1.12,
        "padding_x": 42.0,
        "padding_y": 24.0,
        "font_size_seq": 13,
        "font_size_label": 11,
        "legend_font_size": 10,
        "legend_gap_x": 10.0,
        "legend_height_px": 52.0,
        "layout": {"outer_pad_cells": 0.18},
        "sequence": {"strand_gap_cells": 0.22, "to_kmer_gap_cells": 0.18},
        "kmer": {"box_height_cells": 1.02, "fill_alpha": 0.94, "text_y_nudge_cells": 0.0},
        "connector_width": 1.1,
        "connector_alpha": 0.78,
        "connector_dash": (),
    }
    if view_id in {"payload", "split_payload", "assembled_payload"}:
        base["legend"] = False
    if view_id == "assembled_payload":
        base["padding_y"] = 28.0
    return base


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


def _span(*, start: int, end: int, coordinate_space: str) -> dict[str, object]:
    return {"start": start, "end": end, "coordinate_space": coordinate_space}


def _sequence_contract(
    *,
    state_id: str,
    title: str,
    sequence: str,
    complement_sequence: str,
    meta: dict[str, object],
) -> dict[str, object]:
    return SequenceEvidenceMapV1.model_validate(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": state_id,
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": sequence,
            "complement_sequence": complement_sequence,
            "owners": [],
            "effect_tags": [],
            "boundaries": [],
            "pairings": [],
            "display": {"title": title},
            "meta": meta,
        }
    ).model_dump(mode="json")


def _payload_contract(normalized: NormalizedPayload) -> dict[str, object]:
    motif_layers = [
        YiuPayloadMotifLayerV1(
            motif_instance_id=motif.motif_instance_id,
            tf_name=motif.tf_name,
            motif_name=motif.motif_name,
            reference_strand=motif.reference_strand,
            start=motif.start,
            end=motif.end,
            label=f"{motif.tf_name} ({motif.reference_strand})",
            matrix=[list(row) for row in motif.probabilities.rows],
        )
        for motif in normalized.motif_context.motifs
    ]
    return YiuPayloadVisualV1(
        state_id="payload",
        alphabet="iupac_dna",
        reference_payload_sequence=normalized.reference_payload_sequence,
        selected_payload_sequence=normalized.selected_payload_sequence,
        selected_complement_sequence=normalized.selected_complement_sequence,
        show_reference_payload_row=normalized.selected_payload_sequence != normalized.reference_payload_sequence,
        junction=YiuPayloadJunctionV1(
            start=normalized.junction.start, end=normalized.junction.end, offsets=[0, 1, 2, 3]
        ),
        mismatches=[
            YiuPayloadMismatchV1(
                payload_index=entry.payload_index,
                junction_offset=entry.junction_offset,
                mutated_strand=entry.mutated_strand,
                native_base=entry.native_base,
                mutated_base=entry.mutated_base,
                opposing_base=entry.opposing_base,
            )
            for entry in normalized.mismatches
        ],
        motif_layers=motif_layers,
        display=YiuPayloadDisplayV1(title=normalized.payload_label or "Payload"),
        meta={
            "payload_label": normalized.payload_label,
            "site_label": normalized.site_label,
            "row_labels": _YIU_ROW_LABELS,
            "pwm_effective": normalized.motif_context.effective,
            "motif_ids": [motif.motif_instance_id for motif in normalized.motif_context.motifs],
        },
    ).model_dump(mode="json")


def _split_contract_rows(normalized: NormalizedPayload) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fragment in sorted(build_split_fragment_display_specs(normalized), key=lambda item: item.panel_order):
        span = fragment.sticky_end_display_span.model_dump(mode="json")
        ghost = fragment.ghost_excised_context.model_dump(mode="json") if fragment.ghost_excised_context else None
        rows.append(
            _sequence_contract(
                state_id=f"split_payload_{fragment.fragment_side}",
                title=fragment.title,
                sequence=fragment.display_primary_sequence_5to3,
                complement_sequence=fragment.display_complement_sequence_3to5,
                meta={
                    "view_id": "split_payload",
                    "fragment_side": fragment.fragment_side,
                    "panel_order": fragment.panel_order,
                    "retained_primary_sequence_5to3": fragment.retained_primary_sequence_5to3,
                    "retained_complement_sequence_3to5": fragment.retained_complement_sequence_3to5,
                    "retained_payload_body_sequence_5to3": fragment.retained_payload_body_sequence_5to3,
                    "selected_sticky_end_sequence_5to3": fragment.selected_sticky_end_sequence_5to3,
                    "canonical_sticky_end_sequence_5to3": fragment.canonical_sticky_end_sequence_5to3,
                    "sticky_end_display_span": span,
                    "payload_body_display_span": fragment.payload_body_display_span.model_dump(mode="json"),
                    "retained_primary_display_span": fragment.retained_primary_display_span.model_dump(mode="json"),
                    "retained_complement_display_span": fragment.retained_complement_display_span.model_dump(
                        mode="json"
                    ),
                    "payload_junction_window": fragment.payload_junction_window.model_dump(mode="json"),
                    "sticky_end_orientation": fragment.sticky_end_orientation,
                    "recognition_site_orientation": fragment.recognition_site_orientation,
                    "ghost_excised_context": ghost,
                    "row_labels": _YIU_ROW_LABELS,
                    "connector_hidden_indices": list(range(span["start"], span["end"])),
                    "connector_cross_indices": [],
                    "connector_overhang_spans": [span],
                },
            )
        )
    return rows


def _assembled_contract(normalized: NormalizedPayload) -> dict[str, object]:
    highlight_indices = [site.payload_index for site in normalized.mismatches]
    hidden_indices = [
        index
        for index in range(normalized.junction.start, normalized.junction.end)
        if index not in set(highlight_indices)
    ]
    junction_span = _span(
        start=normalized.junction.start, end=normalized.junction.end, coordinate_space="payload_forward"
    )
    return _sequence_contract(
        state_id="assembled_payload",
        title="Assembled payload",
        sequence=normalized.selected_payload_sequence,
        complement_sequence=assembled_payload_aligned_complement_3to5(normalized),
        meta={
            "view_id": "assembled_payload",
            "junction_span": junction_span,
            "mismatches": [site.model_dump(mode="json") for site in normalized.mismatches],
            "sequence_identity_to_reference_payload": normalized.selected_payload_sequence
            == normalized.reference_payload_sequence,
            "base_highlights": {"primary": highlight_indices, "complement": highlight_indices},
            "connector_hidden_indices": hidden_indices,
            "connector_cross_indices": highlight_indices,
            "connector_overhang_spans": [junction_span],
            "row_labels": _YIU_ROW_LABELS,
        },
    )


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

    payload_contract = _payload_contract(normalized)
    _write_json(payload_path, payload_contract)
    _write_jsonl(split_path, _split_contract_rows(normalized))
    _write_json(assembled_path, _assembled_contract(normalized))

    published_artifacts = {
        "normalized_payload": "normalized_payload.json",
        "bundle_manifest": "bundle_manifest.json",
        "visual_inventory": "visual_inventory.json",
        "payload_view": "payload_view.json",
        "split_payload_view": "split_payload_view.json",
        "assembled_payload_view": "assembled_payload_view.json",
        "payload_views_pdf": "payload_views.pdf",
    }
    if spec.output.published_plot_path is not None:
        published_artifacts["published_plot_pdf"] = str(spec.output.published_plot_path)

    normalized_with_artifacts = normalized.model_copy(update={"published_artifacts": published_artifacts})
    _write_json(normalized_path, normalized_with_artifacts.model_dump(mode="json"))

    payload_motif_layers = [YiuPayloadMotifLayerV1.model_validate(layer) for layer in payload_contract["motif_layers"]]

    view_entries = [
        PayloadViewEntry(
            view_id="payload",
            contract_kind="yiu_payload_visual_v1",
            schema_version=1,
            input_kind="json",
            view_contract_path=_relative_to_bundle(bundle_dir, payload_path),
            render_artifact_path=_relative_to_bundle(bundle_dir, combined_render_path),
            renderer_kind="nucleotide_evidence_map",
            style_overrides=_yiu_style_overrides("payload", motif_layers=payload_motif_layers),
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
            style_overrides=_yiu_style_overrides("split_payload"),
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
            style_overrides=_yiu_style_overrides("assembled_payload"),
            motif_layers_required=False,
        ),
    ]

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
