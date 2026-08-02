"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/resources/artifact_projection.py

Conservative publication-size projection before TriJunction search allocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from dnadesign.trijunction.contracts.publication.limits import ARTIFACT_BYTE_LIMITS
from dnadesign.trijunction.contracts.request import TriJunctionRequest
from dnadesign.trijunction.errors import TriJunctionDesignError

_JSON_DOCUMENT_SLACK = 4_096
_PLAN_POOL_SLACK = 8_192
_PLAN_TARGET_SLACK = 16_384
_PLAN_LOCUS_SLACK = 12_288
_CHECK_SLACK = 1_536
_ORDER_ROW_SLACK = 1_024
_REVIEW_TARGET_SLACK = 20_480
_REVIEW_LOCUS_SLACK = 10_240


def _bytes(value: str) -> int:
    return len(value.encode("utf-8"))


def project_artifact_bytes(
    request: TriJunctionRequest,
    *,
    predicted_loci_by_target: Mapping[str, int],
) -> dict[str, int]:
    """Overestimate every search-dependent artifact from request geometry alone."""

    expected_ids = {target.id for target in request.targets}
    if set(predicted_loci_by_target) != expected_ids or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in predicted_loci_by_target.values()
    ):
        raise TriJunctionDesignError("Artifact projection requires one nonnegative locus count per target.")

    profile = request.planning
    policy = request.order_policy
    policy_bytes = sum(
        _bytes(value)
        for value in (
            policy.synthesis_scale,
            policy.barcode_bearing_purification,
            policy.complement_purification,
            policy.primer_purification,
            policy.complement_end_preparation,
        )
    )
    pool_ids: set[str] = set()
    plan = _JSON_DOCUMENT_SLACK
    checks = _JSON_DOCUMENT_SLACK
    orders = _JSON_DOCUMENT_SLACK
    review = _JSON_DOCUMENT_SLACK

    for target in request.targets:
        target_id_bytes = _bytes(target.id)
        pool_id_bytes = _bytes(target.pool_id)
        pool_ids.add(target.pool_id)
        loci = predicted_loci_by_target[target.id]
        fragments = loci + 1
        order_rows = 2 * fragments + 2
        sequence_bytes = _bytes(target.sequence)
        forward = target.recovery_primers.forward
        reverse = target.recovery_primers.reverse
        primer_binding_bytes = _bytes(forward.binding_sequence) + _bytes(reverse.binding_sequence)
        primer_extension_bytes = _bytes(forward.five_prime_extension) + _bytes(reverse.five_prime_extension)
        primer_order_bytes = primer_binding_bytes + primer_extension_bytes
        identity_bytes = target_id_bytes + pool_id_bytes

        # Plan repeats the target across reconstruction, recovery, paired strands,
        # selected junctions, and junction evidence. Kilobyte record slack covers
        # JSON keys, numbers, digests, generated identifiers, and search receipts.
        plan += (
            _PLAN_TARGET_SLACK
            + 10 * sequence_bytes
            + 4 * primer_binding_bytes
            + 8 * primer_extension_bytes
            + 64 * identity_bytes
            + loci * (_PLAN_LOCUS_SLACK + 64 * identity_bytes + 32 * (profile.toehold_length + profile.barcode_length))
        )

        checks += 5 * (_CHECK_SLACK + 16 * target_id_bytes)

        # Every target-specific order is an upper bound for universal-primer
        # consolidation: merging changes row ownership but cannot increase the
        # total target-id bytes or sequence bytes.
        paired_strand_bytes = 2 * sequence_bytes + 2 * loci * profile.barcode_length
        orders += (
            paired_strand_bytes
            + 2 * primer_order_bytes
            + order_rows * (_ORDER_ROW_SLACK + 8 * target_id_bytes + 4 * pool_id_bytes + policy_bytes)
        )

        # Review rows repeat exact target/recovery/strand sequences, junction
        # geometry, search evidence, and the target plus pool checks.
        review += (
            _REVIEW_TARGET_SLACK
            + 10 * sequence_bytes
            + 8 * primer_binding_bytes
            + 12 * primer_extension_bytes
            + 64 * identity_bytes
            + loci
            * (_REVIEW_LOCUS_SLACK + 48 * identity_bytes + 32 * (profile.toehold_length + profile.barcode_length))
        )

    for pool_id in pool_ids:
        pool_id_bytes = _bytes(pool_id)
        plan += _PLAN_POOL_SLACK + 32 * pool_id_bytes
        checks += 2 * (_CHECK_SLACK + 16 * pool_id_bytes)

    return {
        "plan": plan,
        "checks": checks,
        "orders": orders,
        "review": review,
    }


def guard_artifact_projection(projected_bytes: Mapping[str, int]) -> None:
    """Reject any design whose publication would exceed verifier ceilings."""

    expected = {"plan", "checks", "orders", "review"}
    if set(projected_bytes) != expected:
        raise TriJunctionDesignError("Artifact projection does not cover the complete planned artifact set.")
    for artifact in ("plan", "checks", "orders", "review"):
        projected = projected_bytes[artifact]
        limit = ARTIFACT_BYTE_LIMITS[artifact]
        if projected > limit:
            raise TriJunctionDesignError(
                f"TriJunction projected '{artifact}' artifact is {projected} bytes, exceeding its "
                f"shared {limit}-byte publication and verification limit. Reduce the request, or split only "
                "genuinely independent physical pools across requests."
            )


__all__ = ["guard_artifact_projection", "project_artifact_bytes"]
