"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/viennarna_plot.py

ViennaRNA-native SVG publishing and dnadesign pairing QA enrichment.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import importlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError
from dnadesign.contracts.folding import SecondaryStructurePredictionV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import SecondaryStructurePairingSummaryV1
from dnadesign.contracts.visual import SequenceEvidenceMapV1, ViennaRNAStructureSvgV1

from .errors import FoldingConfigError, FoldingExecutionError
from .pairing_qa import (
    cross_copy_pairing_annotations,
    intended_pair_lookup,
    intended_pairing_qa,
    pair_key,
    unit_copy_spans,
)
from .viennarna_ontology import component_token, hue_for_owners
from .viennarna_svg import SVG_NS, annotate_svg_surface, load_svg_surface, validate_svg_annotation_contract

_VIENNARNA_PLOT_MANIFEST_FILENAME = "viennarna_secondary_structure_svg_v1.json"
_VIENNARNA_NATIVE_SVG_FILENAME = "secondary_structure.native.svg"
_VIENNARNA_ANNOTATED_SVG_FILENAME = "secondary_structure.annotated.svg"
_VIENNARNA_ANNOTATION_MANIFEST_FILENAME = "secondary_structure.annotation_manifest.json"


@dataclass(frozen=True)
class _PlotSequence:
    sequence_id: str
    sequence_sha256: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def publish_viennarna_structure_svg(
    prediction: SecondaryStructurePredictionV1 | str | Path,
    *,
    assembled_sequence_path: str | Path,
    visual_contract_path: str | Path | None = None,
    output_dir: str | Path,
    source_prediction_ref: str | Path | None = None,
    python_module: str = "RNA",
    layout_algorithm: str = "naview",
    emphasize_stem_base_nucleotides: bool = True,
) -> ViennaRNAStructureSvgV1:
    prediction_model, prediction_ref = _load_prediction(prediction)
    if prediction_model.status != "ok" or prediction_model.result is None:
        raise FoldingExecutionError("ViennaRNA structure plotting requires an ok folding prediction.")
    if prediction_model.backend is None or prediction_model.dna_policy is None:
        raise FoldingExecutionError("ViennaRNA structure plotting requires backend and DNA policy metadata.")

    assembled = _load_assembled_sequence(assembled_sequence_path)
    if assembled.sequence_id != prediction_model.input.sequence_id:
        raise FoldingConfigError("Prediction sequence_id does not match assembled sequence artifact.")
    if assembled.sequence_sha256 != prediction_model.input.sequence_sha256:
        raise FoldingConfigError("Prediction sequence_sha256 does not match assembled sequence artifact.")
    if assembled.length != prediction_model.input.length:
        raise FoldingConfigError("Prediction length does not match assembled sequence artifact.")

    submitted_sequence, _submitted_alphabet = _submitted_sequence(
        assembled.sequence,
        dna_policy=prediction_model.dna_policy.mode,
    )
    visual_contract = _optional_sequence_evidence_map(visual_contract_path)
    component_palette = _component_palette(visual_contract)
    validate_svg_annotation_contract(visual_contract)
    nucleotide_annotations = _nucleotide_annotations(
        assembled=assembled,
        submitted_sequence=submitted_sequence,
        visual_contract=visual_contract,
        component_palette=component_palette,
    )
    try:
        module = importlib.import_module(python_module)
    except ImportError as exc:
        raise FoldingConfigError(f"ViennaRNA Python module '{python_module}' is not available: {exc}") from exc

    output_path = Path(output_dir).expanduser().resolve()
    try:
        publication = CreateOnlyDirectoryPublication.prepare(output_path)
    except PublicationError as exc:
        raise FoldingConfigError(str(exc)) from exc
    staging_path = publication.stage
    native_svg_path = staging_path / _VIENNARNA_NATIVE_SVG_FILENAME
    annotated_svg_path = staging_path / _VIENNARNA_ANNOTATED_SVG_FILENAME
    annotation_manifest_path = staging_path / _VIENNARNA_ANNOTATION_MANIFEST_FILENAME
    manifest_path = staging_path / _VIENNARNA_PLOT_MANIFEST_FILENAME

    # Folding may use a U-substituted RNA surrogate, but the published plot is a
    # DNA artifact whose coordinates map to the original assembled sequence.
    try:
        command = _write_native_viennarna_svg(
            module,
            sequence=assembled.sequence,
            structure=prediction_model.result.dot_bracket,
            output_path=native_svg_path,
            layout_algorithm=layout_algorithm,
        )
        surface = load_svg_surface(native_svg_path)
        if len(surface.nucleotide_nodes) != assembled.length:
            raise FoldingExecutionError(
                "ViennaRNA SVG nucleotide node count "
                f"{len(surface.nucleotide_nodes)} does not match input length {assembled.length}."
            )

        svg_annotations = annotate_svg_surface(
            surface,
            nucleotide_annotations=nucleotide_annotations,
            unit_copy_spans=unit_copy_spans(visual_contract),
            intended_pair_lookup=intended_pair_lookup(visual_contract),
            intended_pairing_metrics=tuple(
                item.model_dump(mode="json") for item in prediction_model.qa.intended_pairings
            ),
            visual_contract=visual_contract,
            emphasize_stem_base_nucleotides=emphasize_stem_base_nucleotides,
        )
        ET.register_namespace("", SVG_NS)
        surface.tree.write(annotated_svg_path, encoding="utf-8", xml_declaration=True)
        _write_json(
            annotation_manifest_path,
            {
                "contract_kind": "viennarna_secondary_structure_annotation_manifest_v1",
                "sequence_id": assembled.sequence_id,
                "sequence_sha256": assembled.sequence_sha256,
                "coordinate_system": "zero_based_half_open",
                "viennarna_display_coordinates": "one_based",
                "nucleotides": nucleotide_annotations,
                "basepairs": svg_annotations.basepairs,
                "section_annotations": svg_annotations.section_annotations,
                "layout_normalization": svg_annotations.layout_normalization,
                "palette": component_palette,
            },
        )

        manifest = ViennaRNAStructureSvgV1(
            plot_id=f"{prediction_model.prediction_id}.viennarna_svg",
            prediction_id=prediction_model.prediction_id,
            sequence_id=assembled.sequence_id,
            sequence_sha256=assembled.sequence_sha256,
            length=assembled.length,
            backend_name=prediction_model.backend.name,
            backend_version=prediction_model.backend.version,
            layout_algorithm=layout_algorithm,
            command=command,
            source_prediction=(
                _portable_source_ref(source_prediction_ref)
                if source_prediction_ref is not None
                else _portable_source_ref(prediction_ref)
            ),
            source_visual_contract=(
                _portable_source_ref(visual_contract_path) if visual_contract_path is not None else None
            ),
            artifacts={
                "native_svg": _VIENNARNA_NATIVE_SVG_FILENAME,
                "annotated_svg": _VIENNARNA_ANNOTATED_SVG_FILENAME,
                "annotation_manifest": _VIENNARNA_ANNOTATION_MANIFEST_FILENAME,
            },
            qa={
                "nucleotide_node_count": len(surface.nucleotide_nodes),
                "basepair_node_count": len(surface.basepair_nodes),
                "cross_copy_pair_count": sum(1 for pair in svg_annotations.basepairs if pair["is_cross_copy"]),
                "length_matches_svg_nodes": True,
                "warnings": [],
                "errors": [],
            },
        )
        _write_json(manifest_path, manifest.model_dump(mode="json"))
        publication.publish(required_manifest=_VIENNARNA_PLOT_MANIFEST_FILENAME)
        return manifest
    except PublicationError as exc:
        raise FoldingConfigError(str(exc)) from exc
    finally:
        publication.close()


def _portable_source_ref(source: str | Path) -> str:
    text = str(source)
    if text == "in_memory":
        return text
    path = Path(source).expanduser()
    if not path.is_file() or path.is_symlink():
        raise FoldingConfigError(f"ViennaRNA plot source is unavailable or unsafe: {path}")
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def enrich_prediction_pairing_qa(
    prediction: SecondaryStructurePredictionV1 | str | Path,
    *,
    visual_contract_path: str | Path,
    output_path: str | Path | None = None,
) -> SecondaryStructurePredictionV1:
    prediction_model, _prediction_ref = _load_prediction(prediction)
    if prediction_model.status != "ok" or prediction_model.result is None:
        raise FoldingExecutionError("Pairing QA enrichment requires an ok folding prediction.")
    visual_contract = _load_sequence_evidence_map(visual_contract_path)
    if visual_contract.primary_sequence and len(visual_contract.primary_sequence) != prediction_model.input.length:
        raise FoldingConfigError("Sequence evidence map length does not match folding prediction input.")

    copy_spans = unit_copy_spans(visual_contract)
    predicted_pair_labels = {pair_key(pair.left, pair.right): pair.pair for pair in prediction_model.result.pair_map}
    predicted_pairs = set(predicted_pair_labels)
    cross_copy_pairings = cross_copy_pairing_annotations(
        prediction_model,
        unit_copy_spans=copy_spans,
    )
    intended_pairings = intended_pairing_qa(
        visual_contract,
        predicted_pairs=predicted_pairs,
        predicted_pair_labels=predicted_pair_labels,
    )
    summary = SecondaryStructurePairingSummaryV1(
        predicted_pair_count=len(prediction_model.result.pair_map),
        cross_copy_pair_count=len(cross_copy_pairings),
        intended_pairing_count=len(intended_pairings),
        intended_recovered_count=sum(1 for item in intended_pairings if item.status == "fully_recovered"),
        intended_partially_recovered_count=sum(1 for item in intended_pairings if item.status == "partially_recovered"),
        intended_missed_count=sum(1 for item in intended_pairings if item.status == "missed"),
    )
    qa_payload = prediction_model.qa.model_dump(mode="json")
    qa_payload["pairing_summary"] = summary.model_dump(mode="json")
    qa_payload["cross_copy_pairings"] = cross_copy_pairings
    qa_payload["intended_pairings"] = [item.model_dump(mode="json") for item in intended_pairings]
    payload = prediction_model.model_dump(mode="json")
    payload["qa"] = qa_payload
    enriched = SecondaryStructurePredictionV1.model_validate(payload)
    if output_path is not None:
        _write_json(Path(output_path).expanduser().resolve(), enriched.model_dump(mode="json"))
    return enriched


def _load_prediction(
    prediction: SecondaryStructurePredictionV1 | str | Path,
) -> tuple[SecondaryStructurePredictionV1, str]:
    if isinstance(prediction, SecondaryStructurePredictionV1):
        return prediction, "in_memory"
    prediction_path = Path(prediction).expanduser().resolve()
    try:
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoldingConfigError(f"Invalid folding prediction JSON: {prediction_path}") from exc
    try:
        return SecondaryStructurePredictionV1.model_validate(payload), prediction_path.as_posix()
    except PydanticValidationError as exc:
        raise FoldingConfigError(f"Invalid folding prediction {prediction_path}: {exc}") from exc


def _load_assembled_sequence(path: str | Path) -> _PlotSequence:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FoldingConfigError(f"Assembled sequence artifact not found: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoldingConfigError(f"Invalid assembled sequence JSON: {artifact_path}") from exc
    sequence_payload = payload.get("sequence")
    if not isinstance(sequence_payload, dict):
        raise FoldingConfigError("Assembled sequence artifact missing sequence object.")
    sequence = str(sequence_payload.get("sequence") or "")
    sequence_sha256 = str(sequence_payload.get("sha256") or "")
    sequence_id = str(sequence_payload.get("id") or "")
    observed_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    if not sequence or not sequence_id or not sequence_sha256:
        raise FoldingConfigError("Assembled sequence artifact is missing sequence id, digest, or sequence.")
    if observed_sha256 != sequence_sha256:
        raise FoldingConfigError("Assembled sequence artifact sha256 does not match sequence.")
    return _PlotSequence(sequence_id=sequence_id, sequence_sha256=sequence_sha256, sequence=sequence)


def _submitted_sequence(sequence: str, *, dna_policy: str) -> tuple[str, str]:
    upper = sequence.upper()
    if dna_policy == "convert_t_to_u_for_rna_backend":
        return upper.replace("T", "U"), "rna_surrogate"
    if dna_policy == "backend_accepts_dna_directly":
        return upper, "dna"
    raise FoldingConfigError(f"Unsupported DNA/RNA backend policy: {dna_policy}")


def _write_native_viennarna_svg(
    module: Any,
    *,
    sequence: str,
    structure: str,
    output_path: Path,
    layout_algorithm: str,
) -> list[str]:
    layout = _viennarna_layout(module, structure=structure, layout_algorithm=layout_algorithm)
    plot_structure_svg = getattr(module, "plot_structure_svg", None)
    if callable(plot_structure_svg):
        rc = plot_structure_svg(output_path.as_posix(), sequence, structure, layout=layout)
        command = [f"{module.__name__}.plot_structure_svg", layout_algorithm]
    else:
        svg_rna_plot = getattr(module, "svg_rna_plot", None)
        if not callable(svg_rna_plot):
            raise FoldingExecutionError("ViennaRNA Python module does not expose SVG plotting functions.")
        rc = svg_rna_plot(sequence, structure, output_path.as_posix())
        command = [f"{module.__name__}.svg_rna_plot"]
    if int(rc) != 1:
        raise FoldingExecutionError("ViennaRNA SVG plotting returned a non-success status.")
    return command


def _viennarna_layout(module: Any, *, structure: str, layout_algorithm: str) -> object | None:
    algorithm = str(layout_algorithm or "").strip().lower()
    layout_functions = {
        "simple": "plot_layout_simple",
        "naview": "plot_layout_naview",
        "circular": "plot_layout_circular",
        "turtle": "plot_layout_turtle",
        "puzzler": "plot_layout_puzzler",
    }
    function_name = layout_functions.get(algorithm)
    if function_name is None:
        raise FoldingConfigError(
            "Unsupported ViennaRNA layout algorithm. Expected simple, naview, circular, turtle, or puzzler."
        )
    layout_factory = getattr(module, function_name, None)
    if not callable(layout_factory):
        return None
    return layout_factory(structure)


def _load_sequence_evidence_map(path: str | Path | None) -> SequenceEvidenceMapV1:
    if path is None:
        raise FoldingConfigError("Sequence evidence map is required for pairing QA.")
    contract_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoldingConfigError(f"Invalid sequence evidence map JSON: {contract_path}") from exc
    try:
        return SequenceEvidenceMapV1.model_validate(payload)
    except PydanticValidationError as exc:
        raise FoldingConfigError(f"Invalid sequence evidence map {contract_path}: {exc}") from exc


def _optional_sequence_evidence_map(path: str | Path | None) -> SequenceEvidenceMapV1 | None:
    if path is None:
        return None
    return _load_sequence_evidence_map(path)


def _component_palette(visual_contract: SequenceEvidenceMapV1 | None) -> dict[str, str]:
    if visual_contract is None:
        return {}
    raw_palette = visual_contract.meta.get("component_palette")
    if not isinstance(raw_palette, dict):
        return {}
    return {str(key): str(value) for key, value in raw_palette.items() if str(key).strip() and str(value).strip()}


def _nucleotide_annotations(
    *,
    assembled: _PlotSequence,
    submitted_sequence: str,
    visual_contract: SequenceEvidenceMapV1 | None,
    component_palette: dict[str, str],
) -> list[dict[str, object]]:
    if visual_contract is not None and visual_contract.primary_sequence.upper() != assembled.sequence.upper():
        raise FoldingConfigError("Sequence evidence map primary_sequence does not match assembled sequence.")
    owner_ids: list[list[str]] = [[] for _ in assembled.sequence]
    effect_tags: list[list[str]] = [[] for _ in assembled.sequence]
    if visual_contract is not None:
        for owner in visual_contract.owners:
            if owner.row_id != "primary":
                continue
            for index in range(owner.start, owner.end):
                owner_ids[index].append(owner.owner_id)
        for tag in visual_contract.effect_tags:
            if tag.row_id != "primary":
                continue
            for index in range(tag.start, tag.end):
                effect_tags[index].append(tag.tag_kind)
        for tag in _stem_base_tags_from_segment_labels(visual_contract):
            for index in range(tag["start"], tag["end"]):
                effect_tags[index].append(str(tag["tag_kind"]))
    annotations: list[dict[str, object]] = []
    for index, base in enumerate(assembled.sequence):
        owners = tuple(owner_ids[index])
        tags = tuple(dict.fromkeys(effect_tags[index]))
        hue = _hue_for_nucleotide(owners=owners, effect_tags=tags, component_palette=component_palette)
        annotations.append(
            {
                "index_0": index,
                "display_index_1": index + 1,
                "base_dna": base,
                "base_submitted": submitted_sequence[index],
                "owner_ids": list(owners),
                "effect_tags": list(tags),
                "hue": hue,
                "css_class": f"dnadesign-component-{component_token(owners)}",
            }
        )
    return annotations


def _hue_for_nucleotide(
    *,
    owners: tuple[str, ...],
    effect_tags: tuple[str, ...],
    component_palette: dict[str, str],
) -> str:
    for tag in effect_tags:
        color = component_palette.get(tag)
        if color:
            return color
    return hue_for_owners(owners, palette=component_palette)


def _stem_base_tags_from_segment_labels(visual_contract: SequenceEvidenceMapV1) -> list[dict[str, int | str]]:
    raw_labels = visual_contract.meta.get("segment_labels")
    if not isinstance(raw_labels, list):
        return []
    tags: list[dict[str, int | str]] = []
    for raw in raw_labels:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip().lower()
        if "stem base" not in text and text not in {"left base", "right base"}:
            continue
        try:
            start = int(raw.get("start"))
            end = int(raw.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(visual_contract.primary_sequence):
            continue
        if "left" in text:
            tag_kind = "stem_base_left"
        elif "right" in text:
            tag_kind = "stem_base_right"
        else:
            tag_kind = "stem_base"
        tags.append({"tag_kind": tag_kind, "start": start, "end": end})
    return tags


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "enrich_prediction_pairing_qa",
    "publish_viennarna_structure_svg",
]
