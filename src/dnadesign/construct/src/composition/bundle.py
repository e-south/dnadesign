"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/bundle.py

Bundle publication contracts for generic linear ssDNA composition artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..contracts.errors import ValidationError
from .baserender_jobs import baserender_job_artifacts, write_baserender_jobs
from .exports import write_sequence_exports
from .folding_runtime import folding_artifact_refs, viennarna_structure_plot_artifacts, write_folding_artifacts
from .models import ComposedLinearSsdna
from .visual import (
    DEPRECATED_VISUAL_ARTIFACT_PATHS,
    SEQUENCE_EVIDENCE_MAP_PATH,
    visual_contract_payload,
)

_DEPRECATED_GENERATED_DIR_PATHS = (Path("folding/src"),)
_OPTIONAL_GENERATED_DIR_PATHS = (
    Path("visual/viennarna_secondary_structure"),
    Path("manifest/visual/secondary_structure"),
)


def write_composition_bundle(composed: ComposedLinearSsdna, *, artifact_bundle: Path, config_path: Path) -> None:
    artifact_bundle.mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "baserender_jobs").mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "folding").mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "visual").mkdir(parents=True, exist_ok=True)
    (artifact_bundle / "visual" / "renders").mkdir(parents=True, exist_ok=True)
    _remove_deprecated_generated_artifacts(artifact_bundle)
    _remove_optional_generated_artifacts(artifact_bundle)
    _write_json(artifact_bundle / "assembled_sequence.json", _assembled_sequence_payload(composed))
    _write_json(artifact_bundle / "segment_spans.json", _segment_spans_payload(composed))
    _write_json(artifact_bundle / "annotation_spans.json", _annotation_spans_payload(composed))
    _write_json(artifact_bundle / "provenance.json", _provenance_payload(composed, config_path=config_path))
    _write_json(artifact_bundle / "validation_report.json", _validation_report_payload(composed))
    _write_json(artifact_bundle / SEQUENCE_EVIDENCE_MAP_PATH, visual_contract_payload(composed))
    write_folding_artifacts(artifact_bundle, composed)
    write_baserender_jobs(artifact_bundle, composed)
    write_sequence_exports(artifact_bundle, composed)
    _write_json(artifact_bundle / "manifest.json", _manifest_payload(composed, artifact_bundle=artifact_bundle))
    _write_semantic_manifest_mirror(artifact_bundle)


def _remove_deprecated_generated_artifacts(artifact_bundle: Path) -> None:
    for relative_path in _DEPRECATED_GENERATED_DIR_PATHS:
        path = artifact_bundle / relative_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            raise ValidationError(
                f"Deprecated generated artifact path '{relative_path.as_posix()}' is not a directory. "
                "Remove the stale path or choose a fresh artifact bundle."
            )
    for relative_path in DEPRECATED_VISUAL_ARTIFACT_PATHS:
        path = artifact_bundle / relative_path
        if path.is_file():
            path.unlink()
        elif path.exists():
            raise ValidationError(
                f"Deprecated generated artifact path '{relative_path.as_posix()}' is not a file. "
                "Remove the stale path or choose a fresh artifact bundle."
            )
    _remove_empty_deprecated_directory(artifact_bundle, Path("visual/contracts"))


def _remove_optional_generated_artifacts(artifact_bundle: Path) -> None:
    for relative_path in _OPTIONAL_GENERATED_DIR_PATHS:
        path = artifact_bundle / relative_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            raise ValidationError(
                f"Optional generated artifact path '{relative_path.as_posix()}' is not a directory. "
                "Remove the stale path or choose a fresh artifact bundle."
            )


def _remove_empty_deprecated_directory(artifact_bundle: Path, relative_path: Path) -> None:
    path = artifact_bundle / relative_path
    if not path.exists():
        return
    if not path.is_dir():
        raise ValidationError(
            f"Deprecated generated artifact path '{relative_path.as_posix()}' is not a directory. "
            "Remove the stale path or choose a fresh artifact bundle."
        )
    stale_children = sorted(child.relative_to(artifact_bundle).as_posix() for child in path.rglob("*"))
    if stale_children:
        joined = ", ".join(stale_children[:10])
        suffix = "" if len(stale_children) <= 10 else f", ... ({len(stale_children)} total)"
        raise ValidationError(
            f"Deprecated generated artifact directory '{relative_path.as_posix()}' is non-empty: "
            f"{joined}{suffix}. Remove stale artifacts or choose a fresh artifact bundle."
        )
    path.rmdir()


def _write_semantic_manifest_mirror(artifact_bundle: Path) -> None:
    mirrored_paths = [
        ("manifest.json", "manifest/bundle/manifest.json"),
        ("assembled_sequence.json", "manifest/composition/assembled_sequence.json"),
        ("segment_spans.json", "manifest/composition/segment_spans.json"),
        ("annotation_spans.json", "manifest/composition/annotation_spans.json"),
        ("validation_report.json", "manifest/composition/validation_report.json"),
        ("provenance.json", "manifest/provenance/provenance.json"),
        ("visual/sequence_evidence_map_v1.json", "manifest/visual/sequence_evidence_map_v1.json"),
        ("folding/folding_preflight.json", "manifest/folding/folding_preflight.json"),
        (
            "folding/secondary_structure_prediction_request_v1.yaml",
            "manifest/folding/secondary_structure_prediction_request_v1.yaml",
        ),
        ("folding/secondary_structure_prediction_v1.json", "manifest/folding/secondary_structure_prediction_v1.json"),
        (
            "folding/secondary_structure_input_sequence.json",
            "manifest/folding/secondary_structure_input_sequence.json",
        ),
        (
            "visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json",
            "manifest/visual/secondary_structure/viennarna_secondary_structure_svg_v1.json",
        ),
        (
            "visual/viennarna_secondary_structure/secondary_structure.annotation_manifest.json",
            "manifest/visual/secondary_structure/annotation_manifest.json",
        ),
        (
            "visual/viennarna_secondary_structure/secondary_structure.native.svg",
            "manifest/visual/secondary_structure/native.svg",
        ),
        (
            "visual/viennarna_secondary_structure/secondary_structure.annotated.svg",
            "manifest/visual/secondary_structure/annotated.svg",
        ),
    ]
    for source_name, destination_name in mirrored_paths:
        source = artifact_bundle / source_name
        if not source.is_file():
            continue
        destination = artifact_bundle / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _assembled_sequence_payload(composed: ComposedLinearSsdna) -> dict[str, object]:
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
            **folding_artifact_refs(composed),
            **baserender_job_artifacts(composed),
        },
    }


def _segment_spans_payload(composed: ComposedLinearSsdna) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_segment_spans_v1",
        "coordinate_system": composed.config.coordinate_system,
        "sequence_id": composed.config.composition_id,
        "sequence_length": len(composed.sequence),
        "segments": [span.to_dict() for span in composed.segment_spans],
    }


def _annotation_spans_payload(composed: ComposedLinearSsdna) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_annotation_spans_v1",
        "coordinate_system": composed.config.coordinate_system,
        "sequence_id": composed.config.composition_id,
        "sequence_length": len(composed.sequence),
        "annotations": [span.to_dict() for span in composed.annotation_spans],
    }


def _provenance_payload(composed: ComposedLinearSsdna, *, config_path: Path) -> dict[str, object]:
    return {
        "contract": "linear_ssdna_composition_provenance_v1",
        "composition_id": composed.config.composition_id,
        "config_path": config_path.as_posix(),
        "config_sha256": composed.config_sha256,
        "sequence_sha256": composed.sequence_sha256,
        "created_by": "construct.linear_ssdna_composition",
        "inputs": composed.provenance_inputs,
    }


def _validation_report_payload(composed: ComposedLinearSsdna) -> dict[str, object]:
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


def _manifest_payload(composed: ComposedLinearSsdna, *, artifact_bundle: Path) -> dict[str, object]:
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
            **folding_artifact_refs(composed),
            **viennarna_structure_plot_artifacts(composed, artifact_bundle=artifact_bundle),
            "visual_contract": SEQUENCE_EVIDENCE_MAP_PATH.as_posix(),
            **baserender_job_artifacts(composed),
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


__all__ = ["write_composition_bundle"]
