"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/review_contract.py

Project one plan into strict, study-neutral visual-review rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1
from dnadesign.junction.contracts.identity import canonical_json_bytes
from dnadesign.junction.contracts.plan import AssemblyGroupPlan, JunctionPlan, SelectedJunction, TargetPlan
from dnadesign.junction.errors import JunctionDesignError


def _unique_by_id[T](items: Iterable[T], *, attribute: str, context: str) -> dict[str, T]:
    indexed: dict[str, T] = {}
    for item in items:
        item_id = getattr(item, attribute)
        if item_id in indexed:
            raise JunctionDesignError(f"junction review projection found duplicate {context} {item_id!r}.")
        indexed[item_id] = item
    return indexed


def _junction_rows(
    target: TargetPlan,
    *,
    selected_by_id: dict[str, SelectedJunction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for junction in target.junctions:
        selected = selected_by_id.get(junction.junction_id)
        if selected is None:
            raise JunctionDesignError(f"junction review projection cannot resolve junction {junction.junction_id!r}.")
        if (
            selected.target_id != target.target_id
            or selected.assembly_group_id != target.assembly_group_id
            or selected.toehold != junction.toehold
            or selected.barcode != junction.barcode
        ):
            raise JunctionDesignError(
                f"junction review projection found inconsistent evidence for junction {junction.junction_id!r}."
            )
        rows.append(
            {
                "junction_id": junction.junction_id,
                "toehold_span": {
                    "start": selected.start,
                    "end": selected.start + len(junction.toehold),
                },
                "left_fragment_id": junction.left_fragment_id,
                "right_fragment_id": junction.right_fragment_id,
                "toehold": junction.toehold,
                "toehold_complement": junction.toehold_complement,
                "barcode": junction.barcode,
                "barcode_complement": junction.barcode_complement,
                "complement_nick_sequence_layout_valid": junction.complement_nick_sequence_layout_valid,
                "complement_end_preparation": junction.complement_end_preparation,
            }
        )
    return rows


def _review_mapping(
    plan: JunctionPlan,
    target: TargetPlan,
    *,
    assembly_group: AssemblyGroupPlan,
    selected_by_id: dict[str, SelectedJunction],
) -> dict[str, Any]:
    recovery = target.recovery
    relevant_checks = sorted(
        (
            check
            for check in plan.checks
            if (check.subject.kind == "target" and check.subject.id == target.target_id)
            or (check.subject.kind == "assembly_group" and check.subject.id == target.assembly_group_id)
        ),
        key=lambda check: (check.subject.kind, check.subject.id, check.check),
    )
    if not relevant_checks:
        raise JunctionDesignError(f"junction review projection found no checks for target {target.target_id!r}.")
    return {
        "contract_kind": "three_way_junction_review_v1",
        "source": {
            "plan_schema": plan.schema,
            "plan_id": plan.plan_id,
            "request_sha256": plan.request_sha256,
            "algorithm": plan.algorithm,
        },
        "target": {
            "target_id": target.target_id,
            "assembly_group_id": target.assembly_group_id,
            "sequence_5to3": target.reconstructed_target,
            "sequence_sha256": target.target_sha256,
        },
        "geometry": {
            "fragments": [
                {
                    "fragment_id": fragment.fragment_id,
                    "index": fragment.index,
                    "role": fragment.role,
                    "domain_span": {"start": fragment.domain_start, "end": fragment.domain_end},
                }
                for fragment in target.fragments
            ],
            "junctions": _junction_rows(target, selected_by_id=selected_by_id),
        },
        "strands": [
            {
                "fragment_id": fragment.fragment_id,
                "role": fragment.role,
                "incoming_junction_id": fragment.incoming_junction_id,
                "outgoing_junction_id": fragment.outgoing_junction_id,
                "barcode_bearing_sequence_5to3": fragment.barcode_bearing_strand,
                "complement_sequence_5to3": fragment.complement_strand,
            }
            for fragment in target.fragments
        ],
        "recovery": {
            "mode": recovery.mode,
            "forward": {
                "direction": "forward",
                "binding_sequence_5to3": recovery.forward_binding_sequence,
                "five_prime_extension_5to3": recovery.forward_five_prime_extension,
                "order_sequence_5to3": recovery.forward_order_sequence,
                "target_binding_span": {"start": recovery.forward_start, "end": recovery.forward_end},
            },
            "reverse": {
                "direction": "reverse",
                "binding_sequence_5to3": recovery.reverse_binding_sequence,
                "five_prime_extension_5to3": recovery.reverse_five_prime_extension,
                "order_sequence_5to3": recovery.reverse_order_sequence,
                "target_binding_span": {"start": recovery.reverse_start, "end": recovery.reverse_end},
            },
            "first_fragment_id": recovery.first_fragment_id,
            "last_fragment_id": recovery.last_fragment_id,
            "expected_target_sequence_5to3": recovery.expected_unextended_target,
            "extended_top_sequence_5to3": recovery.extended_top_strand,
            "extended_bottom_sequence_5to3": recovery.extended_bottom_strand,
        },
        "search": {
            field: getattr(assembly_group.search, field)
            for field in (
                "assembly_group_id",
                "toehold_seed",
                "barcode_generation_seed",
                "barcode_subset_seed",
                "matching_seed",
                "locus_count",
                "toehold_paths_evaluated",
                "toehold_min_distance",
                "toehold_mean_distance",
                "toehold_rank_score",
                "barcode_candidates_generated",
                "barcode_forbidden_toehold_k",
                "barcode_forbidden_barcode_k",
                "barcode_subsets_evaluated",
                "barcode_min_distance",
                "barcode_mean_distance",
                "barcode_rank_score",
                "matchings_evaluated",
                "matching_max_pairwise_lcs",
                "thermodynamic_screening",
            )
        },
        "checks": [
            {
                "subject": {"kind": check.subject.kind, "id": check.subject.id},
                "check": check.check,
                "status": check.status,
                "detail": check.detail,
            }
            for check in relevant_checks
        ],
    }


def review_contracts(plan: JunctionPlan) -> tuple[ThreeWayJunctionReviewV1, ...]:
    """Return one strict visual-review row per target in canonical order."""

    assembly_groups_by_id = _unique_by_id(plan.assembly_groups, attribute="assembly_group_id", context="assembly group")
    selected_by_id = _unique_by_id(
        (junction for assembly_group in plan.assembly_groups for junction in assembly_group.junctions),
        attribute="junction_id",
        context="junction",
    )
    reviews: list[ThreeWayJunctionReviewV1] = []
    for target in sorted(plan.targets, key=lambda item: (item.assembly_group_id, item.target_id)):
        assembly_group = assembly_groups_by_id.get(target.assembly_group_id)
        if assembly_group is None:
            raise JunctionDesignError(
                "junction review projection cannot resolve assembly group "
                f"{target.assembly_group_id!r} for target {target.target_id!r}."
            )
        try:
            reviews.append(
                ThreeWayJunctionReviewV1.model_validate(
                    _review_mapping(
                        plan,
                        target,
                        assembly_group=assembly_group,
                        selected_by_id=selected_by_id,
                    )
                )
            )
        except ValidationError as exc:
            raise JunctionDesignError(
                f"junction review projection is invalid for target {target.target_id!r}: {exc}"
            ) from exc
    return tuple(reviews)


def render_review_contracts(plan: JunctionPlan) -> bytes:
    """Serialize all review rows as one canonical BaseRender-compatible JSON array."""

    return canonical_json_bytes([review.model_dump(mode="json") for review in review_contracts(plan)])


__all__ = ["render_review_contracts", "review_contracts"]
