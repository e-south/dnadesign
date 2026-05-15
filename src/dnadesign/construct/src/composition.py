"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition.py

Generic linear ssDNA composition runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

import dnadesign.folding as folding
from dnadesign.contracts.folding import SecondaryStructurePredictionRequestV1
from dnadesign.contracts.sequence import LinearSsdnaCompositionV1
from dnadesign.contracts.sequence.linear_ssdna_composition_v1 import (
    LinearSsdnaAnnotationV1,
    LinearSsdnaSegmentV1,
    LinearSsdnaUnitV1,
)

from .composition_exports import write_sequence_exports
from .composition_visual import (
    CANONICAL_FOLDING_SEQUENCE_PATH,
    DEPRECATED_VISUAL_ARTIFACT_PATHS,
    SEQUENCE_EVIDENCE_MAP_PATH,
    canonical_sequence_artifact_payload,
    visual_contract_payload,
)
from .errors import ConfigError, ValidationError
from .orientation import reverse_complement

_DNA4 = frozenset("ACGTacgt")
_IUPAC_DNA = frozenset("ACGTRYSWKMBDHVNacgtryswkmbdhvn")
_DEPRECATED_GENERATED_DIR_PATHS = (Path("folding/src"),)


@dataclass(frozen=True)
class LinearSsdnaCompositionResult:
    composition_id: str
    sequence_length: int
    sequence_sha256: str
    artifact_bundle: Path
    manifest_path: Path


@dataclass(frozen=True)
class LinearSsdnaCompositionSummary:
    composition_id: str
    unit_count: int
    expanded_copy_count: int
    sequence_length: int


@dataclass(frozen=True)
class _ResolvedSegment:
    unit_id: str
    segment_id: str
    role: str
    sequence: str
    source: dict[str, Any] | None
    transform: dict[str, Any] | None


@dataclass(frozen=True)
class _SegmentSpan:
    copy_index: int
    unit_id: str
    segment_id: str
    role: str
    start: int
    end: int
    sequence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "copy_index": self.copy_index,
            "unit_id": self.unit_id,
            "segment_id": self.segment_id,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class _AnnotationSpan:
    copy_index: int
    unit_id: str
    annotation_id: str
    role: str
    semantic_label: str | None
    start: int
    end: int
    sequence: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "copy_index": self.copy_index,
            "unit_id": self.unit_id,
            "annotation_id": self.annotation_id,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "sequence": self.sequence,
        }
        if self.semantic_label is not None:
            payload["semantic_label"] = self.semantic_label
        return payload


@dataclass(frozen=True)
class _UnitCopySpan:
    unit_id: str
    copy_index: int
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "copy_index": self.copy_index,
            "span": {"start": self.start, "end": self.end},
        }


@dataclass(frozen=True)
class _ComposedLinearSsdna:
    config: LinearSsdnaCompositionV1
    sequence: str
    sequence_sha256: str
    config_sha256: str
    unit_copies: list[_UnitCopySpan]
    segment_spans: list[_SegmentSpan]
    annotation_spans: list[_AnnotationSpan]
    assertions: list[dict[str, object]]
    provenance_inputs: list[dict[str, object]]


def load_linear_ssdna_composition_config(path: str | Path) -> tuple[LinearSsdnaCompositionV1, Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Composition config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in composition config: {config_path}") from exc
    try:
        return LinearSsdnaCompositionV1.model_validate(data), config_path
    except PydanticValidationError as exc:
        raise ConfigError(f"Invalid composition config {config_path}: {exc}") from exc


def summarize_linear_ssdna_composition(path: str | Path) -> LinearSsdnaCompositionSummary:
    config, config_path = load_linear_ssdna_composition_config(path)
    composed = _compose(config, config_path=config_path)
    return LinearSsdnaCompositionSummary(
        composition_id=config.composition_id,
        unit_count=len(config.units),
        expanded_copy_count=sum(unit.repeat_count for unit in config.units),
        sequence_length=len(composed.sequence),
    )


def run_linear_ssdna_composition(path: str | Path) -> LinearSsdnaCompositionResult:
    config, config_path = load_linear_ssdna_composition_config(path)
    composed = _compose(config, config_path=config_path)
    artifact_bundle = _resolve_artifact_bundle(config, config_path=config_path)
    _write_bundle(composed, artifact_bundle=artifact_bundle, config_path=config_path)
    return LinearSsdnaCompositionResult(
        composition_id=config.composition_id,
        sequence_length=len(composed.sequence),
        sequence_sha256=composed.sequence_sha256,
        artifact_bundle=artifact_bundle,
        manifest_path=artifact_bundle / "manifest.json",
    )


def _compose(config: LinearSsdnaCompositionV1, *, config_path: Path) -> _ComposedLinearSsdna:
    sequence_parts: list[str] = []
    unit_copies: list[_UnitCopySpan] = []
    segment_spans: list[_SegmentSpan] = []
    annotation_spans: list[_AnnotationSpan] = []
    assertion_results: list[dict[str, object]] = []
    provenance_inputs: list[dict[str, object]] = []
    cursor = 0
    config_sha256 = _sha256_text(config_path.read_text(encoding="utf-8"))

    for unit in config.units:
        resolved_segments = _resolve_unit_segments(config, unit)
        resolved_by_id = {segment.segment_id: segment for segment in resolved_segments}
        unit_sequence = "".join(segment.sequence for segment in resolved_segments)
        if not unit_sequence:
            raise ValidationError(f"unit '{unit.unit_id}' assembled to an empty sequence.")
        for assertion in unit.assertions:
            status = _check_reverse_complement(
                left=resolved_by_id[assertion.left_segment_id].sequence,
                right=resolved_by_id[assertion.right_segment_id].sequence,
            )
            if not status and assertion.severity == "error":
                raise ValidationError(
                    f"assertion '{assertion.assertion_id}' failed: "
                    f"{assertion.right_segment_id} is not the reverse complement of {assertion.left_segment_id}."
                )
            assertion_results.append(
                {
                    "assertion_id": assertion.assertion_id,
                    "kind": assertion.kind,
                    "severity": assertion.severity,
                    "status": "pass" if status else "fail",
                }
            )
        for copy_index in range(unit.repeat_count):
            unit_start = cursor
            sequence_parts.append(unit_sequence)
            unit_end = unit_start + len(unit_sequence)
            unit_copies.append(
                _UnitCopySpan(unit_id=unit.unit_id, copy_index=copy_index, start=unit_start, end=unit_end)
            )
            copy_segment_spans = _segment_spans_for_copy(
                copy_index=copy_index,
                unit_id=unit.unit_id,
                unit_start=unit_start,
                resolved_segments=resolved_segments,
            )
            segment_spans.extend(copy_segment_spans)
            annotation_spans.extend(
                _annotation_spans_for_copy(
                    unit=unit,
                    copy_index=copy_index,
                    unit_start=unit_start,
                    unit_end=unit_end,
                    sequence="".join(sequence_parts),
                    segment_spans=copy_segment_spans,
                )
            )
            cursor = unit_end
        for segment in resolved_segments:
            if segment.source is None:
                continue
            provenance_inputs.append(
                {
                    "unit_id": unit.unit_id,
                    "segment_id": segment.segment_id,
                    "source": segment.source,
                    "transform": segment.transform,
                }
            )

    sequence = "".join(sequence_parts)
    return _ComposedLinearSsdna(
        config=config,
        sequence=sequence,
        sequence_sha256=_sha256_text(sequence),
        config_sha256=config_sha256,
        unit_copies=unit_copies,
        segment_spans=segment_spans,
        annotation_spans=annotation_spans,
        assertions=assertion_results,
        provenance_inputs=provenance_inputs,
    )


def _resolve_unit_segments(config: LinearSsdnaCompositionV1, unit: LinearSsdnaUnitV1) -> list[_ResolvedSegment]:
    resolved: list[_ResolvedSegment] = []
    resolved_by_id: dict[str, _ResolvedSegment] = {}
    for segment in unit.segments:
        source_sequence = _resolve_segment_sequence(config, unit=unit, segment=segment, resolved_by_id=resolved_by_id)
        resolved_segment = _ResolvedSegment(
            unit_id=unit.unit_id,
            segment_id=segment.segment_id,
            role=segment.role,
            sequence=source_sequence,
            source=_dump_model(segment.source),
            transform=_dump_model(segment.transform),
        )
        resolved.append(resolved_segment)
        resolved_by_id[segment.segment_id] = resolved_segment
    return resolved


def _resolve_segment_sequence(
    config: LinearSsdnaCompositionV1,
    *,
    unit: LinearSsdnaUnitV1,
    segment: LinearSsdnaSegmentV1,
    resolved_by_id: dict[str, _ResolvedSegment],
) -> str:
    if segment.transform is None:
        if segment.sequence is None:
            raise ValidationError(f"segment '{segment.segment_id}' requires sequence.")
        _validate_sequence(config, segment.sequence, label=f"segment '{segment.segment_id}'")
        return segment.sequence

    if segment.transform.kind != "reverse_complement":
        raise ValidationError(f"segment '{segment.segment_id}' has unsupported transform '{segment.transform.kind}'.")
    source = resolved_by_id.get(segment.transform.source_segment_id)
    if source is None:
        raise ValidationError(
            f"segment '{segment.segment_id}' transform source '{segment.transform.source_segment_id}' "
            f"must appear earlier in unit '{unit.unit_id}'."
        )
    derived = reverse_complement(source.sequence)
    if segment.sequence is None:
        _validate_sequence(config, derived, label=f"segment '{segment.segment_id}'")
        return derived
    _validate_sequence(config, segment.sequence, label=f"segment '{segment.segment_id}'")
    if segment.transform.assert_expected_sequence and segment.sequence.upper() != derived.upper():
        raise ValidationError(
            f"{segment.segment_id} reverse_complement does not match {segment.transform.source_segment_id}: "
            f"expected {derived}, observed {segment.sequence}."
        )
    return segment.sequence


def _validate_sequence(config: LinearSsdnaCompositionV1, sequence: str, *, label: str) -> None:
    allowed = _IUPAC_DNA if config.qa.allow_degenerate_bases else _DNA4
    invalid = sorted({base for base in sequence if base not in allowed})
    if invalid:
        joined = "".join(invalid)
        if config.qa.allow_degenerate_bases:
            raise ValidationError(f"{label} contains invalid IUPAC DNA characters: {joined}.")
        raise ValidationError(f"{label} contains non-ACGT bases: {joined}.")
    if config.qa.require_no_unknown_bases and any(base.upper() == "N" for base in sequence):
        raise ValidationError(f"{label} contains unknown base N while require_no_unknown_bases is true.")


def _segment_spans_for_copy(
    *,
    copy_index: int,
    unit_id: str,
    unit_start: int,
    resolved_segments: list[_ResolvedSegment],
) -> list[_SegmentSpan]:
    spans: list[_SegmentSpan] = []
    cursor = unit_start
    for segment in resolved_segments:
        end = cursor + len(segment.sequence)
        spans.append(
            _SegmentSpan(
                copy_index=copy_index,
                unit_id=unit_id,
                segment_id=segment.segment_id,
                role=segment.role,
                start=cursor,
                end=end,
                sequence=segment.sequence,
            )
        )
        cursor = end
    return spans


def _annotation_spans_for_copy(
    *,
    unit: LinearSsdnaUnitV1,
    copy_index: int,
    unit_start: int,
    unit_end: int,
    sequence: str,
    segment_spans: list[_SegmentSpan],
) -> list[_AnnotationSpan]:
    spans_by_segment = {span.segment_id: span for span in segment_spans}
    annotations: list[_AnnotationSpan] = []
    for annotation in unit.annotations:
        start, end = _resolve_annotation_bounds(
            annotation,
            unit_start=unit_start,
            unit_end=unit_end,
            spans_by_segment=spans_by_segment,
        )
        annotations.append(
            _AnnotationSpan(
                copy_index=copy_index,
                unit_id=unit.unit_id,
                annotation_id=annotation.annotation_id,
                role=annotation.role,
                semantic_label=annotation.semantic_label,
                start=start,
                end=end,
                sequence=sequence[start:end],
            )
        )
    return annotations


def _resolve_annotation_bounds(
    annotation: LinearSsdnaAnnotationV1,
    *,
    unit_start: int,
    unit_end: int,
    spans_by_segment: dict[str, _SegmentSpan],
) -> tuple[int, int]:
    location = annotation.location
    if location.basis == "segment":
        assert location.segment_id is not None
        segment = spans_by_segment[location.segment_id]
        if location.end > segment.end - segment.start:
            raise ValidationError(
                f"annotation '{annotation.annotation_id}' exceeds segment '{location.segment_id}' bounds."
            )
        return segment.start + location.start, segment.start + location.end
    if location.basis == "unit":
        if unit_start + location.end > unit_end:
            raise ValidationError(f"annotation '{annotation.annotation_id}' exceeds unit bounds.")
        return unit_start + location.start, unit_start + location.end
    return location.start, location.end


def _check_reverse_complement(*, left: str, right: str) -> bool:
    return reverse_complement(left).upper() == right.upper()


def _resolve_artifact_bundle(config: LinearSsdnaCompositionV1, *, config_path: Path) -> Path:
    configured = config.output.artifact_bundle or f"artifacts/{config.composition_id}"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _write_bundle(composed: _ComposedLinearSsdna, *, artifact_bundle: Path, config_path: Path) -> None:
    artifact_bundle.mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "baserender_jobs").mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "folding").mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "visual").mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "visual" / "renders").mkdir(parents=True, exist_ok=True)
    _remove_deprecated_generated_artifacts(artifact_bundle)
    _write_json(artifact_bundle / "assembled_sequence.json", _assembled_sequence_payload(composed))
    _write_json(artifact_bundle / "segment_spans.json", _segment_spans_payload(composed))
    _write_json(artifact_bundle / "annotation_spans.json", _annotation_spans_payload(composed))
    _write_json(artifact_bundle / "provenance.json", _provenance_payload(composed, config_path=config_path))
    _write_json(artifact_bundle / "validation_report.json", _validation_report_payload(composed))
    _write_json(artifact_bundle / SEQUENCE_EVIDENCE_MAP_PATH, visual_contract_payload(composed))
    _write_folding_artifacts(artifact_bundle, composed)
    _write_baserender_jobs(artifact_bundle, composed)
    write_sequence_exports(artifact_bundle, composed)
    _write_json(artifact_bundle / "manifest.json", _manifest_payload(composed, artifact_bundle=artifact_bundle))


def _remove_deprecated_generated_artifacts(artifact_bundle: Path) -> None:
    for relative_path in _DEPRECATED_GENERATED_DIR_PATHS:
        path = artifact_bundle / relative_path
        if path.is_dir():
            shutil.rmtree(path)
    for relative_path in DEPRECATED_VISUAL_ARTIFACT_PATHS:
        path = artifact_bundle / relative_path
        if path.is_file():
            path.unlink()
    contracts_dir = artifact_bundle / "visual" / "contracts"
    try:
        contracts_dir.rmdir()
    except OSError:
        pass


def _assembled_sequence_payload(composed: _ComposedLinearSsdna) -> dict[str, object]:
    config = composed.config
    return {
        "contract": config.contract,
        "schema_version": config.schema_version,
        "composition_id": config.composition_id,
        "status": "ok",
        "alphabet": config.alphabet,
        "topology": config.topology,
        "coordinate_system": config.coordinate_system,
        "sequence": {
            "id": config.composition_id,
            "length": len(composed.sequence),
            "sha256": composed.sequence_sha256,
            "sequence": composed.sequence,
        },
        "unit_copies": [copy.to_dict() for copy in composed.unit_copies],
        "segment_spans": [span.to_dict() for span in composed.segment_spans],
        "annotation_spans": [span.to_dict() for span in composed.annotation_spans],
        "assertions": composed.assertions,
        "provenance": {
            "inputs": composed.provenance_inputs,
            "source_refs": composed.provenance_inputs,
            "created_by": "construct.linear_ssdna_composition",
            "config_sha256": composed.config_sha256,
        },
        "artifacts": {
            "fasta": "sequence.fa",
            "genbank": "sequence.gb",
            "reverse_complement_fasta": "sequence.reverse_complement.fa",
            "reverse_complement_genbank": "sequence.reverse_complement.gb",
            "features_csv": "features.csv",
            "segment_spans": "segment_spans.json",
            "annotation_spans": "annotation_spans.json",
            "provenance": "provenance.json",
            "validation_report": "validation_report.json",
            "visual_contract": SEQUENCE_EVIDENCE_MAP_PATH.as_posix(),
            **_folding_artifact_refs(composed),
            **_baserender_job_artifacts(composed),
        },
    }


def _segment_spans_payload(composed: _ComposedLinearSsdna) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_segment_spans_v1",
        "coordinate_system": composed.config.coordinate_system,
        "sequence_id": composed.config.composition_id,
        "sequence_length": len(composed.sequence),
        "segments": [span.to_dict() for span in composed.segment_spans],
    }


def _annotation_spans_payload(composed: _ComposedLinearSsdna) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_annotation_spans_v1",
        "coordinate_system": composed.config.coordinate_system,
        "sequence_id": composed.config.composition_id,
        "sequence_length": len(composed.sequence),
        "annotations": [span.to_dict() for span in composed.annotation_spans],
    }


def _provenance_payload(composed: _ComposedLinearSsdna, *, config_path: Path) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_composition_provenance_v1",
        "composition_id": composed.config.composition_id,
        "config_path": config_path.as_posix(),
        "config_sha256": composed.config_sha256,
        "sequence_sha256": composed.sequence_sha256,
        "created_by": "construct.linear_ssdna_composition",
        "inputs": composed.provenance_inputs,
    }


def _validation_report_payload(composed: _ComposedLinearSsdna) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_composition_validation_report_v1",
        "composition_id": composed.config.composition_id,
        "status": "ok",
        "errors": [],
        "warnings": [],
        "checks": {
            "sequence_alphabet": "pass",
            "segment_span_coverage": "pass",
            "annotation_bounds": "pass",
            "declared_transform_checks": "pass",
            "reverse_complement_assertions": "pass"
            if all(a["status"] == "pass" for a in composed.assertions)
            else "fail",
        },
    }


def _write_folding_artifacts(artifact_bundle: Path, composed: _ComposedLinearSsdna) -> None:
    if not composed.config.folding.enabled:
        return
    folding_dir = artifact_bundle / "folding"
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


def _visual_emit_enabled(composed: _ComposedLinearSsdna, kind: str) -> bool:
    return kind in set(composed.config.visual.emit)


def _folding_request(
    composed: _ComposedLinearSsdna,
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


def _folding_artifact_refs(composed: _ComposedLinearSsdna) -> dict[str, str]:
    if not composed.config.folding.enabled:
        return {}
    return {
        "folding_input_sequence": CANONICAL_FOLDING_SEQUENCE_PATH.as_posix(),
        "folding_preflight": "folding/folding_preflight.json",
        "folding_request": "folding/secondary_structure_prediction_request_v1.yaml",
        "folding_prediction": "folding/secondary_structure_prediction_v1.json",
    }


def _viennarna_structure_plot_artifacts(artifact_bundle: Path) -> dict[str, str]:
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


def _baserender_export_formats(composed: _ComposedLinearSsdna) -> list[str]:
    requested = composed.config.visual.render_exports.formats or ["svg"]
    formats: list[str] = []
    allowed = {"svg", "pdf", "png"}
    for raw_format in requested:
        fmt = raw_format.strip().lower()
        if fmt not in allowed:
            raise ValidationError(f"Unsupported BaseRender export format '{raw_format}'. Expected svg, pdf, or png.")
        if fmt not in formats:
            formats.append(fmt)
    return formats


def _baserender_job_artifacts(composed: _ComposedLinearSsdna) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for fmt in _baserender_export_formats(composed):
        artifacts[f"baserender_component_span_{fmt}_job"] = f"baserender_jobs/component_span_qa_{fmt}.yaml"
    return artifacts


def _write_baserender_jobs(artifact_bundle: Path, composed: _ComposedLinearSsdna) -> None:
    for fmt in _baserender_export_formats(composed):
        job_path = artifact_bundle / "baserender_jobs" / f"component_span_qa_{fmt}.yaml"
        payload = {
            "version": 3,
            "contract": {"kind": "nucleotide_evidence_map_render_v3"},
            "results_root": "../visual/renders",
            "input": {
                "kind": "json",
                "path": f"../{SEQUENCE_EVIDENCE_MAP_PATH.as_posix()}",
                "adapter": {
                    "kind": "sequence_evidence_map_v1",
                    "columns": {},
                    "policies": {},
                },
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "nucleotide_evidence_map",
                "style": {
                    "preset": "presentation_default",
                    "overrides": {
                        "figure_scale": 1.15,
                        "padding_y": 16.0,
                        "overlay_align": "center",
                        "baseline_spacing": 28.0,
                        "layout": {"outer_pad_cells": 0.35},
                        "sequence": {"strand_gap_cells": 0.08},
                        "connectors": True,
                        "color_ticks": "#CBD5E1",
                        "connector_alpha": 0.42,
                        "connector_width": 0.5,
                        "connector_dash": [],
                        "show_reverse_complement": True,
                        "legend": False,
                        "legend_mode": "none",
                        "legend_height_px": 48.0,
                        "uniform_display_font_size": True,
                        "legend_font_size": 6,
                        "font_size_seq": 6,
                        "font_size_label": 6,
                        "font_size_feature_label": 6,
                        "font_size_annotation_label": 6,
                        "font_size_span_link_label": 6,
                        "span_link_line_width": 1.4,
                        "span_link_tick_line_width": 1.0,
                    },
                },
            },
            "outputs": [
                {
                    "kind": "images",
                    "path": f"component_span_qa.{fmt}",
                    "fmt": fmt,
                }
            ],
            "run": {
                "strict": True,
                "fail_on_skips": True,
                "emit_report": True,
                "report_path": "component_span_qa.run_report.json",
            },
        }
        job_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _manifest_payload(composed: _ComposedLinearSsdna, *, artifact_bundle: Path) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_composition_bundle_manifest_v1",
        "composition_id": composed.config.composition_id,
        "status": "ok",
        "artifact_bundle": artifact_bundle.as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contracts": {
            "composition": "linear_ssdna_composition_v1",
            **({"folding": "secondary_structure_prediction_v1"} if composed.config.folding.enabled else {}),
            "visual": "sequence_evidence_map_v1",
        },
        "artifacts": {
            "assembled_sequence": "assembled_sequence.json",
            "segment_spans": "segment_spans.json",
            "annotation_spans": "annotation_spans.json",
            "provenance": "provenance.json",
            "validation_report": "validation_report.json",
            "fasta": "sequence.fa",
            "genbank": "sequence.gb",
            "reverse_complement_fasta": "sequence.reverse_complement.fa",
            "reverse_complement_genbank": "sequence.reverse_complement.gb",
            "features_csv": "features.csv",
            **_folding_artifact_refs(composed),
            **_viennarna_structure_plot_artifacts(artifact_bundle),
            "visual_contract": SEQUENCE_EVIDENCE_MAP_PATH.as_posix(),
            **_baserender_job_artifacts(composed),
        },
        "operator_hints": _operator_hints(artifact_bundle),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _operator_hints(artifact_bundle: Path) -> dict[str, object]:
    genbank_path = artifact_bundle / "sequence.gb"
    return {
        "genbank": {
            "path": "sequence.gb",
            "macos_finder_reveal": f"open -R {shlex.quote(genbank_path.as_posix())}",
        }
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_model(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return]
    return None


__all__ = [
    "LinearSsdnaCompositionResult",
    "LinearSsdnaCompositionSummary",
    "load_linear_ssdna_composition_config",
    "run_linear_ssdna_composition",
    "summarize_linear_ssdna_composition",
]
