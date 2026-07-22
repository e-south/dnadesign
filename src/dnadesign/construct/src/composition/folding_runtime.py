"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/folding_runtime.py

Folding and ViennaRNA visual artifact emission for composition bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

import dnadesign.folding as folding
from dnadesign.contracts.folding import SecondaryStructurePredictionRequestV1

from ..contracts.errors import ValidationError
from .models import ComposedLinearSsdna
from .visual import (
    CANONICAL_FOLDING_SEQUENCE_PATH,
    SEQUENCE_EVIDENCE_MAP_PATH,
    canonical_sequence_artifact_payload,
)


def write_folding_artifacts(artifact_bundle: Path, composed: ComposedLinearSsdna) -> None:
    if not composed.config.folding.enabled:
        return
    folding_dir = artifact_bundle / "folding"
    folding_dir.mkdir(parents=True, exist_ok=True)
    folding_sequence_payload = canonical_sequence_artifact_payload(composed)
    _write_json(artifact_bundle / CANONICAL_FOLDING_SEQUENCE_PATH, folding_sequence_payload)
    request = _folding_request(composed, folding_sequence_payload=folding_sequence_payload)
    request_path = folding_dir / "secondary_structure_prediction_request_v1.yaml"
    request_path.write_text(yaml.safe_dump(request.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    try:
        prediction = folding.run_prediction_request(
            request,
            output_dir=folding_dir,
            request_path=request_path,
        )
    except folding.FoldingError as exc:
        raise ValidationError(f"folding failed: {exc}") from exc
    if prediction.status != "ok":
        return
    visual_contract_path = artifact_bundle / SEQUENCE_EVIDENCE_MAP_PATH
    prediction_path = folding_dir / "secondary_structure_prediction_v1.json"
    try:
        folding.enrich_prediction_pairing_qa(
            prediction_path,
            visual_contract_path=visual_contract_path,
            output_path=prediction_path,
        )
    except folding.FoldingError as exc:
        raise ValidationError(f"folding QA enrichment failed: {exc}") from exc
    if not _visual_emit_enabled(composed, "viennarna_secondary_structure_svg_v1"):
        return
    try:
        folding.publish_viennarna_structure_svg(
            prediction_path,
            assembled_sequence_path=artifact_bundle / CANONICAL_FOLDING_SEQUENCE_PATH,
            visual_contract_path=visual_contract_path,
            output_dir=artifact_bundle / "visual" / "viennarna_secondary_structure",
            python_module=composed.config.visual.viennarna_structure_plot.python_module,
            layout_algorithm=composed.config.visual.viennarna_structure_plot.layout_algorithm,
            emphasize_stem_base_nucleotides=(
                composed.config.visual.viennarna_structure_plot.emphasize_stem_base_nucleotides
            ),
        )
    except folding.FoldingError as exc:
        raise ValidationError(f"ViennaRNA structure plotting failed: {exc}") from exc


def folding_artifact_refs(composed: ComposedLinearSsdna) -> dict[str, str]:
    if not composed.config.folding.enabled:
        return {}
    return {
        "folding_input_sequence": CANONICAL_FOLDING_SEQUENCE_PATH.as_posix(),
        "folding_preflight": "folding/folding_preflight.json",
        "folding_request": "folding/secondary_structure_prediction_request_v1.yaml",
        "folding_prediction": "folding/secondary_structure_prediction_v1.json",
    }


def viennarna_structure_plot_artifacts(composed: ComposedLinearSsdna, *, artifact_bundle: Path) -> dict[str, str]:
    if not composed.config.folding.enabled:
        return {}
    if "viennarna_secondary_structure_svg_v1" not in set(composed.config.visual.emit):
        return {}
    plot_manifest = (
        artifact_bundle / "visual" / "viennarna_secondary_structure" / "viennarna_secondary_structure_svg_v1.json"
    )
    if not plot_manifest.is_file():
        return {}
    return {
        "viennarna_structure_plot": "visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json",
        "viennarna_structure_native_svg": "visual/viennarna_secondary_structure/secondary_structure.native.svg",
        "viennarna_structure_annotated_svg": "visual/viennarna_secondary_structure/secondary_structure.annotated.svg",
        "viennarna_structure_annotation_manifest": (
            "visual/viennarna_secondary_structure/secondary_structure.annotation_manifest.json"
        ),
    }


def _visual_emit_enabled(composed: ComposedLinearSsdna, kind: str) -> bool:
    return kind in set(composed.config.visual.emit)


def _folding_request(
    composed: ComposedLinearSsdna,
    *,
    folding_sequence_payload: dict[str, object],
) -> SecondaryStructurePredictionRequestV1:
    folding_config = composed.config.folding
    if folding_config.scope != "canonical_component_unit":
        raise ValidationError(
            "folding.scope must be canonical_component_unit. Repeat-expanded folding is not supported because "
            "visual/folding QA must not concatenate product copies."
        )
    backend = folding_config.backend
    dna_policy = folding_config.dna_policy
    if backend is None or dna_policy is None:
        raise ValidationError("folding.backend and folding.dna_policy are required when folding is enabled.")
    backend_payload: dict[str, object] = {
        "name": backend.name,
        "interface": backend.interface,
        "backend_contract": backend.backend_contract or "secondary_structure_prediction_v1",
        "parameters": backend.parameters,
        "dna_policy": {
            "mode": dna_policy.mode,
            "output_coordinates": "original_dna_sequence",
        },
    }
    if backend.interface == "cli":
        backend_payload["executable"] = backend.executable or backend.name
    if backend.interface == "python_api":
        backend_payload["python_module"] = backend.python_module
    sequence_payload = folding_sequence_payload["sequence"]
    if not isinstance(sequence_payload, dict):
        raise ValidationError("canonical folding sequence payload is missing sequence metadata.")
    return SecondaryStructurePredictionRequestV1(
        request_id=f"{composed.config.composition_id}.viennafold.canonical_component_unit",
        input={
            "sequence_artifact": CANONICAL_FOLDING_SEQUENCE_PATH.name,
            "sequence_id": sequence_payload["id"],
            "sequence_sha256": sequence_payload["sha256"],
            "alphabet": composed.config.alphabet,
            "topology": composed.config.topology,
            "length": sequence_payload["length"],
        },
        scope={"mode": folding_config.scope},
        backend=backend_payload,
        policy={
            "required": folding_config.required,
            "fail_on_malformed_output": True,
            "fail_on_length_mismatch": True,
        },
    )


def _write_json(path: Path, payload: object) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
