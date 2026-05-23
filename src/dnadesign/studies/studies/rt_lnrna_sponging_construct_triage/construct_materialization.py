"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/construct_materialization.py

Construct materialization helpers for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import yaml

from dnadesign.construct import RunResult, run_from_config
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces

from .construct_projection import validate_projection_manifest_payload
from .genbank_authority import GenBankAuthorityAudit, run_default_authority_audit

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_PROJECTION_MANIFEST_PATH = _STUDY_DIR / "operations/contract/fixtures/construct/construct-projection-manifest.yaml"
_INPUT_DATASET = "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1"
_OUTPUT_DATASET = "rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1"
_MATERIALIZATION_SOURCE = "rt_lnrna_sponging_construct_triage construct materialization"
_REQUIRED_SLOT_IDS = ("lnrna", "rt_cds")
_SEQUENCE_ID_SOURCE_MAP = {
    "1600bp-region.gb": "dual_cassette_1600bp_region",
    "pes-retron-26.gb": "pes_retron_26_vector",
    "pes-retron-26-a1-a2.gb": "pes_retron_26_lnrna_a1_a2",
    "retron-eco1-rt.gb": "retron_eco1_rt",
    "pes-retron-43.gb": "pes_retron_43_vector",
}


class MaterializationContractError(ValueError):
    """Raised when study-owned projection inputs cannot safely materialize."""


@dataclass(frozen=True)
class ControlConstructMaterializationReport:
    usr_root: Path
    input_dataset: str
    output_dataset: str
    input_ids_by_candidate_id: dict[str, str]
    config_paths: tuple[Path, ...]
    run_results: tuple[RunResult, ...]
    template_sequence: str
    template_context_sequence: str
    expected_sequences: dict[str, str]


def materialize_control_construct_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
    candidate_sequence_overrides: Mapping[str, Mapping[str, str]] | None = None,
    omitted_candidate_fields: tuple[str, ...] = (),
) -> ControlConstructMaterializationReport:
    """Materialize the two checked-in control candidates into temp USR outputs."""
    root = _resolve_repo_root(repo_root)
    manifest = _load_projection_manifest(root)
    _require_valid_projection_manifest(manifest)
    authority = _require_genbank_authority(root)
    template_sequence = _template_sequence(manifest=manifest, authority=authority)
    target_start, target_end = _target_context_bounds(manifest)
    template_context_sequence = template_sequence[target_start:target_end]

    rows, expected_sequences = _candidate_rows(
        manifest=manifest,
        authority=authority,
        template_sequence=template_sequence,
        target_start=target_start,
        target_end=target_end,
        candidate_sequence_overrides=candidate_sequence_overrides or {},
        omitted_candidate_fields=set(omitted_candidate_fields),
    )

    work = Path(work_root).resolve()
    usr_root = work / "usr"
    config_dir = work / "construct_configs"
    usr_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    input_ids_by_candidate_id = _write_candidate_dataset(usr_root=usr_root, rows=rows)

    context_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_candidate_id=input_ids_by_candidate_id,
        job_id="rt_lnrna_control_context_views",
        output_on_conflict="error",
        output_variants=[
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "forward",
                "recommended_pooling": "seq_mean",
                "view_name": "dual_cassette_1600bp_seq_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "reverse_complement",
                "recommended_pooling": "seq_mean",
                "view_name": "dual_cassette_1600bp_fwd_rc_concat",
            },
        ],
    )
    slot_anchor_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_candidate_id=input_ids_by_candidate_id,
        job_id="rt_lnrna_slot_anchor_views",
        output_on_conflict="ignore",
        output_variants=[
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "forward",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "lnrna",
                "view_name": "lnrna_span_in_construct_anchor_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "reverse_complement",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "lnrna",
                "view_name": "lnrna_span_in_construct_reverse_complement_anchor_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "forward",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "rt_cds",
                "view_name": "rt_cds_span_in_construct_anchor_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "reverse_complement",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "rt_cds",
                "view_name": "rt_cds_span_in_construct_reverse_complement_anchor_mean",
            },
        ],
    )
    context_path = _write_config(config_dir / "construct-context-views.yaml", context_config)
    slot_anchor_path = _write_config(config_dir / "construct-slot-anchor-views.yaml", slot_anchor_config)
    context_result = run_from_config(context_path)
    slot_anchor_result = run_from_config(slot_anchor_path)
    return ControlConstructMaterializationReport(
        usr_root=usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_candidate_id=input_ids_by_candidate_id,
        config_paths=(context_path, slot_anchor_path),
        run_results=(context_result, slot_anchor_result),
        template_sequence=template_sequence,
        template_context_sequence=template_context_sequence,
        expected_sequences=expected_sequences,
    )


def _load_projection_manifest(repo_root: Path) -> dict[str, object]:
    payload = yaml.safe_load((repo_root / _PROJECTION_MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MaterializationContractError("Construct projection manifest must be a mapping.")
    return payload


def _require_valid_projection_manifest(manifest: dict[str, object]) -> None:
    audit = validate_projection_manifest_payload(manifest)
    if not audit.ok:
        joined = "; ".join(audit.errors)
        raise MaterializationContractError(f"Construct projection manifest is invalid: {joined}")


def _require_genbank_authority(repo_root: Path) -> GenBankAuthorityAudit:
    audit = run_default_authority_audit(repo_root=repo_root)
    if not audit.ok:
        joined = "; ".join(audit.errors)
        raise MaterializationContractError(f"GenBank source authority is invalid: {joined}")
    return audit


def _template_sequence(*, manifest: dict[str, object], authority: GenBankAuthorityAudit) -> str:
    template = _mapping(manifest["construct_template"], label="construct_template")
    source_id = str(template["source_authority_id"])
    return authority.source(source_id).sequence


def _target_context_bounds(manifest: dict[str, object]) -> tuple[int, int]:
    template = _mapping(manifest["construct_template"], label="construct_template")
    target = _mapping(template["target_context"], label="construct_template.target_context")
    start = int(target["window_start_0"])
    end = int(target["window_end_0"])
    if end <= start:
        raise MaterializationContractError("target_context.window_end_0 must be greater than window_start_0.")
    expected_length = int(target.get("length_nt", end - start))
    if end - start != expected_length:
        raise MaterializationContractError("target_context window span must equal target_context.length_nt.")
    return start, end


def _candidate_rows(
    *,
    manifest: dict[str, object],
    authority: GenBankAuthorityAudit,
    template_sequence: str,
    target_start: int,
    target_end: int,
    candidate_sequence_overrides: Mapping[str, Mapping[str, str]],
    omitted_candidate_fields: set[str],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    candidates = tuple(
        _mapping(candidate, label="candidates[]") for candidate in _list(manifest["candidates"], label="candidates")
    )
    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        slot_bindings = _mapping(candidate["slot_bindings"], label=f"{candidate_id}.slot_bindings")
        row: dict[str, object] = {
            "id": candidate_id,
            # USR base row ids stay canonical sequence ids; candidate identity
            # travels through the candidate overlay and usr_label namespace.
            "sequence": "A" * (index + 1),
            "source": _MATERIALIZATION_SOURCE,
        }
        for slot in slots:
            slot_id = str(slot["slot_id"])
            field_name = str(slot["sequence_field"])
            binding = _mapping(slot_bindings[slot_id], label=f"{candidate_id}.slot_bindings.{slot_id}")
            sequence = _sequence_for_binding(binding=binding, authority=authority)
            sequence = candidate_sequence_overrides.get(candidate_id, {}).get(field_name, sequence)
            expected_length = int(binding["sequence_length_nt"])
            if len(sequence) != expected_length:
                raise MaterializationContractError(
                    f"{candidate_id}: {field_name} length {len(sequence)} does not match "
                    f"declared {slot_id} length {expected_length}."
                )
            row[field_name] = None if field_name in omitted_candidate_fields else sequence
        rows.append(row)
        expected_sequences[candidate_id] = _expected_context_sequence(
            template_sequence=template_sequence,
            slots=slots,
            row=row,
            target_start=target_start,
            target_end=target_end,
        )
    return rows, expected_sequences


def _sequence_for_binding(*, binding: dict[str, object], authority: GenBankAuthorityAudit) -> str:
    sequence_id = str(binding["sequence_id"])
    source_id = _source_id_for_sequence_id(sequence_id)
    start, end = _span_0(binding["source_sequence_span_0"], label=f"{sequence_id}.source_sequence_span_0")
    return authority.source(source_id).sequence[start:end]


def _source_id_for_sequence_id(sequence_id: str) -> str:
    if not sequence_id.startswith("genbank:"):
        raise MaterializationContractError(f"Unsupported sequence authority id: {sequence_id}")
    path_part = sequence_id.removeprefix("genbank:").split("#", maxsplit=1)[0]
    source_id = _SEQUENCE_ID_SOURCE_MAP.get(path_part)
    if source_id is None:
        raise MaterializationContractError(f"No GenBank source mapping for sequence id: {sequence_id}")
    return source_id


def _expected_context_sequence(
    *,
    template_sequence: str,
    slots: tuple[dict[str, object], ...],
    row: Mapping[str, object],
    target_start: int,
    target_end: int,
) -> str:
    cursor = 0
    out: list[str] = []
    realized_spans: dict[str, tuple[int, int]] = {}
    out_len = 0
    for slot in sorted(slots, key=lambda item: _span_0(item["template_span_0"], label="template_span_0")[0]):
        start, end = _span_0(slot["template_span_0"], label=f"{slot['slot_id']}.template_span_0")
        field_name = str(slot["sequence_field"])
        value = row.get(field_name)
        if value is None:
            raise MaterializationContractError(
                f"Input row '{row.get('id')}' is missing field '{field_name}' for part '{slot['slot_id']}'."
            )
        prefix = template_sequence[cursor:start]
        sequence = str(value)
        out.append(prefix)
        out_len += len(prefix)
        realized_start = out_len
        out.append(sequence)
        out_len += len(sequence)
        realized_spans[str(slot["slot_id"])] = (realized_start, out_len)
        cursor = end
    out.append(template_sequence[cursor:])
    full_construct = "".join(out)
    window_start, window_end = _candidate_window_bounds(
        slots=slots,
        realized_spans=realized_spans,
        target_start=target_start,
        target_end=target_end,
    )
    if window_start < 0 or window_end > len(full_construct):
        raise MaterializationContractError(
            f"Input row '{row.get('id')}' target context [{window_start}, {window_end}) falls outside the "
            f"realized construct length {len(full_construct)}."
        )
    return full_construct[window_start:window_end]


def _write_candidate_dataset(*, usr_root: Path, rows: list[dict[str, object]]) -> dict[str, str]:
    _ensure_candidate_overlay_namespace(usr_root)
    dataset = Dataset(usr_root, _INPUT_DATASET)
    dataset.init(source=_MATERIALIZATION_SOURCE, notes="Temp RT-lnRNA Construct materialization inputs.")
    add_result = dataset.add_sequences(
        [str(row["sequence"]) for row in rows],
        bio_type="dna",
        alphabet="dna_4",
        source=_MATERIALIZATION_SOURCE,
    )
    input_ids_by_candidate_id = {str(row["id"]): input_id for row, input_id in zip(rows, add_result.ids, strict=True)}
    field_names = ("candidate__lnrna_sequence", "candidate__rt_cds_sequence")
    input_ids = [input_ids_by_candidate_id[str(row["id"])] for row in rows]
    columns: dict[str, pa.Array] = {
        "id": pa.array(input_ids, type=pa.string()),
        "candidate__candidate_id": pa.array([str(row["id"]) for row in rows], type=pa.string()),
    }
    for field_name in field_names:
        columns[field_name] = pa.array([row.get(field_name) for row in rows], type=pa.string())
    dataset.write_overlay("candidate", pa.table(columns), key="id", overwrite=True)
    dataset.write_overlay(
        "usr_label",
        pa.table(
            {
                "id": pa.array(input_ids, type=pa.string()),
                "usr_label__primary": pa.array([str(row["id"]) for row in rows], type=pa.string()),
                "usr_label__aliases": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
            }
        ),
        key="id",
        overwrite=True,
    )
    return input_ids_by_candidate_id


def _ensure_candidate_overlay_namespace(usr_root: Path) -> None:
    ensure_sequence_contract_namespaces(usr_root)
    registry_path = usr_root / "registry.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise MaterializationContractError(f"{registry_path}: expected registry mapping")
    namespaces = payload.setdefault("namespaces", {})
    if not isinstance(namespaces, dict):
        raise MaterializationContractError(f"{registry_path}: namespaces must be a mapping")
    namespaces["candidate"] = {
        "owner": "study",
        "description": "RT-lnRNA candidate slot sequences.",
        "columns": [
            {"name": "candidate__candidate_id", "type": "string"},
            {"name": "candidate__lnrna_sequence", "type": "string"},
            {"name": "candidate__rt_cds_sequence", "type": "string"},
        ],
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _construct_config(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    usr_root: Path,
    input_ids_by_candidate_id: Mapping[str, str],
    job_id: str,
    output_on_conflict: str,
    output_variants: list[dict[str, object]],
) -> dict[str, object]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    target_start, target_end = _target_context_bounds(manifest)
    window_offset_bp = _centered_window_offset_bp(slots=slots, target_start=target_start, target_end=target_end)
    return {
        "job": {
            "id": job_id,
            "input": {
                "source": {
                    "kind": "usr",
                    "dataset": _INPUT_DATASET,
                    "root": str(usr_root),
                },
                "field": None,
                "ids": [
                    input_ids_by_candidate_id[str(candidate["candidate_id"])]
                    for candidate in _list(manifest["candidates"], label="candidates")
                ],
            },
            "template": {
                "id": str(
                    _mapping(manifest["construct_template"], label="construct_template")["construct_template_id"]
                ),
                "source": {
                    "kind": "literal",
                    "sequence": template_sequence,
                    "label": "genbank:pes-retron-26.gb#record",
                },
                "circular": True,
            },
            "parts": [_part_config(slot=slot, template_sequence=template_sequence) for slot in slots],
            "realize": {
                "mode": "window",
                "focal_part": "lnrna",
                "required_slots": list(_REQUIRED_SLOT_IDS),
                "window": {
                    "semantics": "fixed_total",
                    "reference": "center",
                    "direction": "symmetric",
                    "size_bp": target_end - target_start,
                    "offset_bp": window_offset_bp,
                },
            },
            "output_variants": output_variants,
            "output": {
                "record_source": _MATERIALIZATION_SOURCE,
                "on_conflict": output_on_conflict,
                "target": {
                    "kind": "usr",
                    "dataset": _OUTPUT_DATASET,
                    "root": str(usr_root),
                },
            },
        }
    }


def _centered_window_offset_bp(
    *,
    slots: tuple[dict[str, object], ...],
    target_start: int,
    target_end: int,
) -> int:
    lnrna_slot = next((slot for slot in slots if str(slot["slot_id"]) == "lnrna"), None)
    if lnrna_slot is None:
        raise MaterializationContractError("Centered RT-lnRNA window requires an lnrna slot.")
    lnrna_start, lnrna_end = _span_0(lnrna_slot["template_span_0"], label="lnrna.template_span_0")
    base_center = lnrna_start + ((lnrna_end - lnrna_start) // 2)
    window_length = target_end - target_start
    return target_start - (base_center - (window_length // 2))


def _candidate_window_bounds(
    *,
    slots: tuple[dict[str, object], ...],
    realized_spans: dict[str, tuple[int, int]],
    target_start: int,
    target_end: int,
) -> tuple[int, int]:
    lnrna_slot = next((slot for slot in slots if str(slot["slot_id"]) == "lnrna"), None)
    if lnrna_slot is None:
        raise MaterializationContractError("Centered RT-lnRNA window requires an lnrna slot.")
    base_start, base_end = _span_0(lnrna_slot["template_span_0"], label="lnrna.template_span_0")
    realized_start, realized_end = realized_spans["lnrna"]
    base_center = base_start + ((base_end - base_start) // 2)
    realized_center = realized_start + ((realized_end - realized_start) // 2)
    window_start = target_start + (realized_center - base_center)
    return window_start, window_start + (target_end - target_start)


def _part_config(*, slot: dict[str, object], template_sequence: str) -> dict[str, object]:
    start, end = _span_0(slot["template_span_0"], label=f"{slot['slot_id']}.template_span_0")
    return {
        "name": str(slot["slot_id"]),
        "role": str(slot["role"]),
        "sequence": {
            "source": "input_field",
            "field": str(slot["sequence_field"]),
        },
        "placement": {
            "kind": "replace",
            "orientation": str(slot["orientation"]),
            "locator": {
                "kind": "coordinates",
                "start": start,
                "end": end,
            },
            "guards": {
                "replaced_sequence": template_sequence[start:end],
                "replaced_span_bp": end - start,
            },
        },
    }


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MaterializationContractError(f"{label} must be a mapping.")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise MaterializationContractError(f"{label} must be a list.")
    return value


def _span_0(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise MaterializationContractError(f"{label} must be [start, end].")
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end <= start:
        raise MaterializationContractError(f"{label} must be a valid zero-based half-open span.")
    return start, end


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")
