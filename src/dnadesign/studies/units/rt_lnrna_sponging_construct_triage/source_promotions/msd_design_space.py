"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/source_promotions/msd_design_space.py

Primitive-backed MSD design-space expansion for RT-lnRNA promotions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from dnadesign.cruncher.scar_nick import ScarNickPrimitiveExportError, load_scar_nick_stem_base_primitives
from dnadesign.cruncher.snapback import SnapbackPrimitiveExportError, load_released_solve_cap_primitives

from .common import slug
from .contracts import SourcePromotionContractError
from .msd_pool_contract import RtLnrnaMsdVariantPoolSpecV1


@dataclass(frozen=True, slots=True)
class _DesignSpaceCapOption:
    cap_id: str
    source: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _DesignSpaceStemBaseOption:
    stem_base_id: str
    compiler_fields: Mapping[str, Any]


def resolve_design_space_compiler_payload(
    *,
    spec: RtLnrnaMsdVariantPoolSpecV1,
    compiler_inputs: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    if spec.design_space is None:
        raise SourcePromotionContractError("Pool spec requires design_space when compiler_spec is absent.")
    design_space = spec.design_space
    payload_ids = design_space.payload_ids
    cap_options = _design_space_cap_options(design_space=design_space, root=root)
    stem_base_options = _design_space_stem_base_options(design_space=design_space, root=root)
    count = len(payload_ids) * len(cap_options) * len(stem_base_options)
    if count > spec.max_variant_count:
        raise SourcePromotionContractError(
            f"Compiler design_space emits {count} variant(s), which exceeds max_variant_count={spec.max_variant_count}."
        )
    cap_sequences = dict(_mapping(compiler_inputs.get("cap_sequences"), label="compiler_inputs.cap_sequences"))
    for cap_option in cap_options:
        if cap_option.source is None:
            continue
        if cap_option.cap_id in cap_sequences:
            raise SourcePromotionContractError(
                f"Primitive cap option {cap_option.cap_id} collides with compiler_inputs.cap_sequences."
            )
        cap_sequences[cap_option.cap_id] = {"source": dict(cap_option.source)}

    construct_prefix = design_space.construct_id_prefix
    designs: list[dict[str, Any]] = []
    for payload_id, cap_option, stem_base_option in product(payload_ids, cap_options, stem_base_options):
        designs.append(
            {
                "construct_id": (
                    f"{construct_prefix}__{slug(payload_id)}__{slug(cap_option.cap_id)}__"
                    f"{slug(stem_base_option.stem_base_id)}"
                ),
                "payload_id": payload_id,
                "cap_id": cap_option.cap_id,
                **stem_base_option.compiler_fields,
            }
        )
    return {
        "contract": "retron_msd_compiler_spec_v1",
        "schema_version": 1,
        "allow_non_ligatable_s0": spec.allow_non_ligatable_s0,
        "designs": designs,
        "payload_sequences": _mapping(
            compiler_inputs.get("payload_sequences"),
            label="compiler_inputs.payload_sequences",
        ),
        "cap_sequences": cap_sequences,
    }


def _design_space_cap_options(*, design_space, root: Path) -> tuple[_DesignSpaceCapOption, ...]:
    options = [_DesignSpaceCapOption(cap_id=cap_id) for cap_id in design_space.cap_ids]
    for source_spec in design_space.cap_primitives:
        run_dir = _resolve_repo_path(root=root, value=source_spec.run_dir, label="design_space.cap_primitives.run_dir")
        try:
            selected = _select_primitives_by_rank(
                load_released_solve_cap_primitives(run_dir),
                ranks=source_spec.ranks,
                label=f"cap primitive source {source_spec.source_id}",
            )
        except SnapbackPrimitiveExportError as exc:
            raise SourcePromotionContractError(str(exc)) from exc
        if source_spec.expected_primitive_count is not None and len(selected) != source_spec.expected_primitive_count:
            raise SourcePromotionContractError(
                f"cap primitive source {source_spec.source_id} expected "
                f"{source_spec.expected_primitive_count} primitive(s) but selected {len(selected)}."
            )
        for primitive in selected:
            cap_id = f"{source_spec.cap_id_prefix}{primitive.rank:02d}"
            options.append(
                _DesignSpaceCapOption(
                    cap_id=cap_id,
                    source={
                        "kind": source_spec.kind,
                        "run_dir": run_dir.as_posix(),
                        "selector": {"mode": "rank", "rank": primitive.rank},
                    },
                )
            )
    _reject_duplicate_option_ids((option.cap_id for option in options), label="cap option")
    return tuple(options)


def _design_space_stem_base_options(*, design_space, root: Path) -> tuple[_DesignSpaceStemBaseOption, ...]:
    options = [
        _DesignSpaceStemBaseOption(
            stem_base_id=stem_base.stem_base_id,
            compiler_fields=stem_base.compiler_design_fields(),
        )
        for stem_base in design_space.stem_bases
    ]
    for source_spec in design_space.stem_base_primitives:
        run_dir = _resolve_repo_path(
            root=root,
            value=source_spec.run_dir,
            label="design_space.stem_base_primitives.run_dir",
        )
        try:
            selected = _select_primitives_by_rank(
                load_scar_nick_stem_base_primitives(run_dir),
                ranks=source_spec.ranks,
                label=f"stem-base primitive source {source_spec.source_id}",
            )
        except ScarNickPrimitiveExportError as exc:
            raise SourcePromotionContractError(str(exc)) from exc
        if source_spec.expected_primitive_count is not None and len(selected) != source_spec.expected_primitive_count:
            raise SourcePromotionContractError(
                f"stem-base primitive source {source_spec.source_id} expected "
                f"{source_spec.expected_primitive_count} primitive(s) but selected {len(selected)}."
            )
        for primitive in selected:
            stem_base_id = (
                f"{source_spec.stem_base_id_prefix}{primitive.rank:02d}_"
                f"{primitive.left_base}_{primitive.right_base}_{primitive.profile_s3s2s1s0}"
            )
            options.append(
                _DesignSpaceStemBaseOption(
                    stem_base_id=stem_base_id,
                    compiler_fields={
                        "stem_base_source": {
                            "kind": source_spec.kind,
                            "run_dir": run_dir.as_posix(),
                            "selector": {"mode": "rank", "rank": primitive.rank},
                        },
                        "source_notes": (
                            "YIU-compatible cloning method primitive composition: "
                            f"scar-nick source {source_spec.source_id} rank {primitive.rank}; "
                            f"stem base {primitive.left_base}/{primitive.right_base}; "
                            f"profile {primitive.profile_s3s2s1s0}."
                        ),
                    },
                )
            )
    _reject_duplicate_option_ids((option.stem_base_id for option in options), label="stem-base option")
    return tuple(options)


def _select_primitives_by_rank(primitives: list[Any], *, ranks: list[int], label: str) -> list[Any]:
    by_rank = {}
    for primitive in primitives:
        rank = int(primitive.rank)
        if rank in by_rank:
            raise SourcePromotionContractError(f"{label} contains duplicate primitive rank {rank}.")
        by_rank[rank] = primitive
    missing = [rank for rank in ranks if rank not in by_rank]
    if missing:
        raise SourcePromotionContractError(f"{label} is missing requested primitive rank(s): {missing}.")
    return [by_rank[rank] for rank in ranks]


def _reject_duplicate_option_ids(option_ids, *, label: str) -> None:
    seen: set[str] = set()
    for option_id in option_ids:
        if option_id in seen:
            raise SourcePromotionContractError(f"Duplicate {label} id in MSD compiler design_space: {option_id}")
        seen.add(option_id)


def _resolve_repo_path(*, root: Path, value: str, label: str) -> Path:
    raw_path = _required_text(value, label=label)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourcePromotionContractError(f"{label} must be a mapping.")
    return value


def _required_text(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SourcePromotionContractError(f"{label} must be non-empty.")
    return text


__all__ = ["resolve_design_space_compiler_payload"]
