"""
Retron compiler-generated MSD lnRNA source promotion.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Mapping

from Bio import SeqIO

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import (
    MsdCompilerSpecError,
    resolve_msd_compiler_spec_payload,
)
from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.compiler.msd_unit import (
    MsdCompiledUnitV1,
    compile_msd_design_unit,
)

from .common import (
    ConstructWindowPolicy,
    construct_window_fit_issue,
    format_span,
    require_dna,
    require_no_internal_stop_codons,
    reverse_complement,
    sha256_text,
    slug,
)
from .contracts import SourceConstructSubjectPromotion, SourcePromotionContractError
from .msd_pool_contract import RtLnrnaMsdVariantPoolSpecV1, load_msd_variant_pool_spec

MSD_COMPILER_SOURCE_BASIS = "compiler_generated_msd_lnrna_variant"
MSD_COMPILER_AUTHORITY_KIND = "compiler_generated_lnrna_sequence"
MSD_COMPILER_RT_AUTHORITY_KIND = "fixed_eco1_wt_rt"


def resolve_msd_compiler_promotions(
    *,
    repo_root: Path,
    pool_spec_path: Path,
    wt_rt_cds_sequence: str,
    window_policy: ConstructWindowPolicy,
) -> tuple[SourceConstructSubjectPromotion, ...]:
    root = Path(repo_root).resolve()
    spec_path = Path(pool_spec_path).expanduser().resolve()
    spec = _load_pool_spec(spec_path)
    pool_id = spec.pool_id
    payload_program_id = spec.payload_program_id

    wt_rt = require_dna(wt_rt_cds_sequence, label="wt_rt_cds_sequence")
    if len(wt_rt) % 3:
        raise SourcePromotionContractError("WT RT CDS sequence length must be divisible by 3.")
    require_no_internal_stop_codons(wt_rt, label="wt_rt_cds_sequence")

    template_lnrna_sequence = _load_template_lnrna_sequence(
        root=root,
        payload=spec.template_lnrna.model_dump(),
    )
    compiler_inputs = spec.compiler_inputs.model_dump(mode="python")
    variant_compiler_payload = _variant_compiler_payload(spec=spec)
    template_compiler_payload = {
        "contract": "retron_msd_compiler_spec_v1",
        "schema_version": 1,
        "allow_non_ligatable_s0": True,
        "designs": [spec.template_msd_design.model_dump(exclude_none=True)],
        "payload_sequences": _mapping(
            compiler_inputs.get("payload_sequences"),
            label="compiler_inputs.payload_sequences",
        ),
        "cap_sequences": _mapping(compiler_inputs.get("cap_sequences"), label="compiler_inputs.cap_sequences"),
    }
    resolved_template = _resolve_retron_compiler_payload(
        payload=template_compiler_payload,
        spec_path=spec_path,
        root=root,
    )
    if len(resolved_template.catalog.records) != 1:
        raise SourcePromotionContractError("template_msd_design must compile to exactly one MSD reference.")
    template_unit = compile_msd_design_unit(
        resolved_template.catalog.records[0],
        payload_sequences=resolved_template.payload_sequences,
        cap_sequences=resolved_template.cap_sequences,
    )
    placement = _template_msd_placement(
        template_lnrna_sequence=template_lnrna_sequence,
        template_unit=template_unit,
        placement=spec.placement.model_dump(),
    )

    resolved_variants = _resolve_retron_compiler_payload(
        payload=variant_compiler_payload,
        spec_path=spec_path,
        root=root,
    )
    if (
        spec.expected_variant_count is not None
        and len(resolved_variants.catalog.records) != spec.expected_variant_count
    ):
        raise SourcePromotionContractError(
            f"Compiler pool expected {spec.expected_variant_count} variant(s) but emitted "
            f"{len(resolved_variants.catalog.records)}."
        )

    promotions: list[SourceConstructSubjectPromotion] = []
    seen_lnrna_sequences: dict[str, str] = {}
    for record in resolved_variants.catalog.records:
        unit = compile_msd_design_unit(
            record,
            payload_sequences=resolved_variants.payload_sequences,
            cap_sequences=resolved_variants.cap_sequences,
        )
        lnrna_insert_sequence = reverse_complement(unit.sequence_5to3)
        lnrna_sequence = patch_lnrna_template_with_msd(
            template_lnrna_sequence=template_lnrna_sequence,
            replacement_span_0=placement["replacement_span_0"],
            msd_product_sequence_5to3=unit.sequence_5to3,
            lnrna_insert_sequence_5to3=lnrna_insert_sequence,
        )
        duplicate_record = seen_lnrna_sequences.get(lnrna_sequence)
        if duplicate_record is not None:
            raise SourcePromotionContractError(
                "Duplicate compiler-generated lnRNA sequence from "
                f"{duplicate_record} and {record.msd_design_id}; dedupe_policy=fail."
            )
        seen_lnrna_sequences[lnrna_sequence] = record.msd_design_id
        fit_issue = construct_window_fit_issue(
            lnrna_sequence=lnrna_sequence,
            rt_cds_sequence=wt_rt,
            window_policy=window_policy,
        )
        if fit_issue:
            raise SourcePromotionContractError(f"{record.msd_design_id}: {fit_issue}")
        sequence_sha = sha256_text(lnrna_sequence)
        promotions.append(
            SourceConstructSubjectPromotion(
                construct_subject_id=(f"rt_lnrna_pair__eco1_wt_rt__compiler_msd_{slug(record.msd_design_id)}__tetO"),
                lnrna_sequence=lnrna_sequence,
                rt_cds_sequence=wt_rt,
                source_basis=MSD_COMPILER_SOURCE_BASIS,
                source_collection_id=pool_id,
                source_record_id=record.msd_design_id,
                source_record_count=1,
                source_lnrna_design_id=record.msd_design_id,
                source_sequence_sha256=sequence_sha,
                lnrna_authority_kind=MSD_COMPILER_AUTHORITY_KIND,
                rt_cds_authority_kind=MSD_COMPILER_RT_AUTHORITY_KIND,
                overlay_fields=_overlay_fields(
                    pool_id=pool_id,
                    payload_program_id=payload_program_id,
                    record=record,
                    unit=unit,
                    sequence_sha=sequence_sha,
                    template_lnrna_ref=spec.template_lnrna.sequence_ref,
                    replacement_span_0=placement["replacement_span_0"],
                    source_refs=spec.source_refs,
                ),
            )
        )
    return tuple(promotions)


def patch_lnrna_template_with_msd(
    *,
    template_lnrna_sequence: str,
    replacement_span_0: tuple[int, int],
    msd_product_sequence_5to3: str,
    lnrna_insert_sequence_5to3: str,
) -> str:
    template = require_dna(template_lnrna_sequence, label="template_lnrna_sequence")
    msd_product = require_dna(msd_product_sequence_5to3, label="msd_product_sequence_5to3")
    lnrna_insert = require_dna(lnrna_insert_sequence_5to3, label="lnrna_insert_sequence_5to3")
    expected_insert = reverse_complement(msd_product)
    if lnrna_insert != expected_insert:
        raise SourcePromotionContractError(
            "lnRNA insert sequence must be the reverse complement of the 5'->3' MSD product sequence."
        )
    start, end = replacement_span_0
    if start < 0 or end <= start or end > len(template):
        raise SourcePromotionContractError(
            f"MSD replacement span {start}:{end} is outside template lnRNA length {len(template)}."
        )
    return template[:start] + lnrna_insert + template[end:]


def _variant_compiler_payload(*, spec: RtLnrnaMsdVariantPoolSpecV1) -> dict[str, Any]:
    compiler_inputs = spec.compiler_inputs.model_dump(mode="python")
    if spec.compiler_spec is not None:
        payload = dict(spec.compiler_spec)
        count = len(_list(payload.get("labels", []), label="compiler_spec.labels")) + len(
            _list(payload.get("designs", []), label="compiler_spec.designs")
        )
        if count > spec.max_variant_count:
            raise SourcePromotionContractError(
                f"Compiler pool emits {count} variant(s), which exceeds max_variant_count={spec.max_variant_count}."
            )
        return payload

    if spec.design_space is None:
        raise SourcePromotionContractError("Pool spec requires design_space when compiler_spec is absent.")
    design_space = spec.design_space
    payload_ids = design_space.payload_ids
    cap_ids = design_space.cap_ids
    stem_bases = design_space.stem_bases
    count = len(payload_ids) * len(cap_ids) * len(stem_bases)
    if count > spec.max_variant_count:
        raise SourcePromotionContractError(
            f"Compiler design_space emits {count} variant(s), which exceeds max_variant_count={spec.max_variant_count}."
        )
    construct_prefix = design_space.construct_id_prefix
    designs: list[dict[str, Any]] = []
    for payload_id, cap_id, stem_base in product(payload_ids, cap_ids, stem_bases):
        design = {
            "construct_id": f"{construct_prefix}__{slug(payload_id)}__{slug(cap_id)}__{slug(stem_base.stem_base_id)}",
            "payload_id": payload_id,
            "cap_id": cap_id,
            **stem_base.compiler_design_fields(),
        }
        designs.append(design)
    return {
        "contract": "retron_msd_compiler_spec_v1",
        "schema_version": 1,
        "allow_non_ligatable_s0": spec.allow_non_ligatable_s0,
        "designs": designs,
        "payload_sequences": _mapping(
            compiler_inputs.get("payload_sequences"),
            label="compiler_inputs.payload_sequences",
        ),
        "cap_sequences": _mapping(compiler_inputs.get("cap_sequences"), label="compiler_inputs.cap_sequences"),
    }


def _resolve_retron_compiler_payload(*, payload: Mapping[str, Any], spec_path: Path, root: Path):
    try:
        return resolve_msd_compiler_spec_payload(
            payload,
            spec_path=spec_path,
            study_dir=root / "docs/studies/retron_hairpin_design",
        )
    except (MsdCompilerSpecError, RetronMsdCompilerError, ValueError) as exc:
        raise SourcePromotionContractError(str(exc)) from exc


def _template_msd_placement(
    *,
    template_lnrna_sequence: str,
    template_unit: MsdCompiledUnitV1,
    placement: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    template = require_dna(template_lnrna_sequence, label="template_lnrna_sequence")
    template_insert = reverse_complement(template_unit.sequence_5to3)
    spans = _find_all_spans(template, template_insert)
    if len(spans) != 1:
        raise SourcePromotionContractError(
            f"Template lnRNA must contain the reverse-complemented template MSD unit exactly once; found {len(spans)}."
        )
    start, end = spans[0]
    expected_5p = require_dna(_required_str(placement, "expected_5p_flank"), label="placement.expected_5p_flank")
    expected_3p = require_dna(_required_str(placement, "expected_3p_flank"), label="placement.expected_3p_flank")
    actual_5p = template[start - len(expected_5p) : start] if start >= len(expected_5p) else ""
    actual_3p = template[end : end + len(expected_3p)]
    if actual_5p != expected_5p:
        raise SourcePromotionContractError(
            f"Template 5' flank mismatch at MSD placement: expected {expected_5p}, observed {actual_5p or '<missing>'}."
        )
    if actual_3p != expected_3p:
        raise SourcePromotionContractError(
            f"Template 3' flank mismatch at MSD placement: expected {expected_3p}, observed {actual_3p or '<missing>'}."
        )
    return {"replacement_span_0": (start, end)}


def _overlay_fields(
    *,
    pool_id: str,
    payload_program_id: str,
    record,
    unit: MsdCompiledUnitV1,
    sequence_sha: str,
    template_lnrna_ref: str,
    replacement_span_0: tuple[int, int],
    source_refs: list[str],
) -> dict[str, object]:
    return {
        "construct_subject__role": "compiler_lnrna_variant",
        "construct_subject__variant_class": MSD_COMPILER_SOURCE_BASIS,
        "construct_subject__construct_projection_status": "representable",
        "construct_subject__source_basis": MSD_COMPILER_SOURCE_BASIS,
        "construct_subject__source_collection_id": pool_id,
        "construct_subject__source_record_id": record.msd_design_id,
        "construct_subject__source_record_count": 1,
        "construct_subject__source_literature_id": "",
        "construct_subject__source_label_kind": "compiler_design_reference",
        "construct_subject__source_regime": "study_owned_msd_combinatorics",
        "construct_subject__source_lnrna_design_id": record.msd_design_id,
        "construct_subject__source_sequence_sha256": sequence_sha,
        "construct_subject__lnrna_authority_kind": MSD_COMPILER_AUTHORITY_KIND,
        "construct_subject__rt_cds_authority_kind": MSD_COMPILER_RT_AUTHORITY_KIND,
        "construct_subject__rt_source": "Retron-Eco1",
        "construct_subject__rt_variant": "WT",
        "construct_subject__payload_program_id": payload_program_id,
        "construct_subject__msd_design_id": record.msd_design_id,
        "construct_subject__msd_payload_id": record.payload_or_target.id,
        "construct_subject__msd_cap_id": record.cap.id,
        "construct_subject__msd_stem_base_left": unit.segment_sequence("stem_base_left"),
        "construct_subject__msd_stem_base_right": unit.segment_sequence("stem_base_right"),
        "construct_subject__msd_profile_s3s2s1s0": record.scar_nick.profile_s3s2s1s0,
        "construct_subject__msd_insert_orientation": "reverse_complement",
        "construct_subject__msd_template_lnrna_ref": template_lnrna_ref,
        "construct_subject__msd_template_replacement_span_0": format_span(replacement_span_0),
        "construct_subject__msd_product_length_nt": len(unit.sequence_5to3),
        "construct_subject__msd_source_refs": ";".join(source_refs),
        "construct_subject__msd_cap_source_construct": record.cap.source_construct or "",
        "construct_subject__msd_snapback_topology_source": unit.provenance.get("snapback_topology_source") or "",
        "construct_subject__msd_scar_nick_route_status": record.scar_nick.route_status,
        "construct_subject__msd_scar_nick_route_note": record.scar_nick.route_note or "",
        "construct_subject__msd_nick_orientation": record.scar_nick.nick_orientation or "",
        "construct_subject__msd_nickase": record.scar_nick.nickase or "",
    }


def reject_duplicate_msd_compiler_lnrna_sequences(
    promotions: tuple[SourceConstructSubjectPromotion, ...] | list[SourceConstructSubjectPromotion],
) -> None:
    seen: dict[str, str] = {}
    for promotion in promotions:
        if promotion.source_basis != MSD_COMPILER_SOURCE_BASIS:
            continue
        provenance = f"{promotion.source_collection_id}/{promotion.source_record_id}"
        duplicate = seen.get(promotion.lnrna_sequence)
        if duplicate is not None:
            raise SourcePromotionContractError(
                "Duplicate compiler-generated lnRNA sequence across pools from "
                f"{duplicate} and {provenance}; dedupe_policy=fail."
            )
        seen[promotion.lnrna_sequence] = provenance


def _load_pool_spec(path: Path) -> RtLnrnaMsdVariantPoolSpecV1:
    return load_msd_variant_pool_spec(path)


def _load_template_lnrna_sequence(*, root: Path, payload: Mapping[str, Any]) -> str:
    raw_path = _required_str(payload, "genbank_path")
    genbank_path = (root / raw_path).resolve()
    try:
        genbank_path.relative_to(root)
    except ValueError as exc:
        raise SourcePromotionContractError(f"template_lnrna.genbank_path escapes repo root: {raw_path}") from exc
    if not genbank_path.is_file():
        raise SourcePromotionContractError(f"template_lnrna.genbank_path does not exist: {genbank_path}")
    span = _span(payload.get("sequence_span_0"), label="template_lnrna.sequence_span_0")
    record = SeqIO.read(genbank_path, "genbank")
    sequence = str(record.seq).upper()
    start, end = span
    if end > len(sequence):
        raise SourcePromotionContractError(
            f"template_lnrna.sequence_span_0 {start}:{end} exceeds GenBank length {len(sequence)}."
        )
    return require_dna(sequence[start:end], label="template_lnrna.sequence")


def _find_all_spans(sequence: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = sequence.find(needle)
    while start >= 0:
        spans.append((start, start + len(needle)))
        start = sequence.find(needle, start + 1)
    return spans


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourcePromotionContractError(f"{label} must be a mapping.")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise SourcePromotionContractError(f"{label} must be a list.")
    return list(value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    text = str(value or "").strip()
    if not text:
        raise SourcePromotionContractError(f"{key} must be non-empty.")
    return text


def _span(value: Any, *, label: str) -> tuple[int, int]:
    items = _list(value, label=label)
    if len(items) != 2:
        raise SourcePromotionContractError(f"{label} must contain [start, end].")
    start = int(items[0])
    end = int(items[1])
    if start < 0 or end <= start:
        raise SourcePromotionContractError(f"{label} must be a zero-based half-open span.")
    return (start, end)


__all__ = [
    "MSD_COMPILER_AUTHORITY_KIND",
    "MSD_COMPILER_RT_AUTHORITY_KIND",
    "MSD_COMPILER_SOURCE_BASIS",
    "patch_lnrna_template_with_msd",
    "reject_duplicate_msd_compiler_lnrna_sequences",
    "resolve_msd_compiler_promotions",
]
