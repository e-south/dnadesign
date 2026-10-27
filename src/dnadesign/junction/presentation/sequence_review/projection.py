"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/sequence_review/projection.py

Project one Junction plan into assembly-group sequence-review records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pydantic import ValidationError

from dnadesign.junction.contracts.identity import canonical_json_bytes
from dnadesign.junction.contracts.plan import JunctionPlan
from dnadesign.junction.errors import JunctionDesignError

from .contract import JunctionSequenceDissimilarityV1


def sequence_dissimilarity_contracts(plan: JunctionPlan) -> tuple[JunctionSequenceDissimilarityV1, ...]:
    """Return one compact sequence-review record per assembly group."""

    target_ids = {target.target_id for target in plan.targets}
    reviews: list[JunctionSequenceDissimilarityV1] = []
    observed_group_ids: set[str] = set()
    for group in sorted(plan.assembly_groups, key=lambda item: item.assembly_group_id):
        if group.assembly_group_id in observed_group_ids:
            raise JunctionDesignError(
                f"junction sequence review found duplicate assembly group {group.assembly_group_id!r}."
            )
        observed_group_ids.add(group.assembly_group_id)
        if len(group.junctions) != group.search.locus_count:
            raise JunctionDesignError(
                f"junction sequence review found an inconsistent locus count for {group.assembly_group_id!r}."
            )
        if group.search.assembly_group_id != group.assembly_group_id or any(
            junction.assembly_group_id != group.assembly_group_id for junction in group.junctions
        ):
            raise JunctionDesignError(
                f"junction sequence review found an inconsistent assembly group {group.assembly_group_id!r}."
            )
        if any(junction.target_id not in target_ids for junction in group.junctions):
            raise JunctionDesignError(
                f"junction sequence review cannot resolve every target in {group.assembly_group_id!r}."
            )
        try:
            reviews.append(
                JunctionSequenceDissimilarityV1.model_validate(
                    {
                        "contract_kind": "junction_sequence_dissimilarity_v1",
                        "source": {
                            "plan_schema": plan.schema,
                            "plan_id": plan.plan_id,
                            "request_sha256": plan.request_sha256,
                            "algorithm": plan.algorithm,
                        },
                        "assembly_group_id": group.assembly_group_id,
                        "junctions": [
                            {
                                "junction_id": junction.junction_id,
                                "target_id": junction.target_id,
                                "toehold_sequence_5to3": junction.toehold,
                                "barcode_sequence_5to3": junction.barcode,
                            }
                            for junction in group.junctions
                        ],
                        "thermodynamic_screening": group.search.thermodynamic_screening,
                    }
                )
            )
        except ValidationError as exc:
            raise JunctionDesignError(
                f"junction sequence review is invalid for assembly group {group.assembly_group_id!r}: {exc}"
            ) from exc
    return tuple(reviews)


def render_sequence_dissimilarity_contracts(plan: JunctionPlan) -> bytes:
    """Serialize assembly-group sequence-review records as canonical JSON."""

    return canonical_json_bytes([review.model_dump(mode="json") for review in sequence_dissimilarity_contracts(plan)])


__all__ = ["render_sequence_dissimilarity_contracts", "sequence_dissimilarity_contracts"]
