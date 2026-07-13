"""Policy-defined selected-panel orchestration and validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    EXPECTED_SELECTED_POLICY_COUNTS,
    GROUP_ALLOCATIONS,
    PANEL_POLICY_IDS,
    SELECTED_PANEL_SIZE,
    allocation_for_policy,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_ranking import (
    choose_farthest_candidate,
    nearest_sequence_distance,
    select_dissimilar_pair,
    with_nearest_mutation_audit,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_rows import (
    build_panel_row,
)


def build_selected_panel_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    """Select one eight-sequence panel with dissimilar mutation sets within each policy."""

    candidate_by_id = _validated_candidate_lookup(candidate_rows)
    contract_rows = _validated_contract_rows(triage_rows=triage_rows, candidate_by_id=candidate_by_id)
    sequence_by_id = {candidate_id: str(row.get("sequence") or "") for candidate_id, row in candidate_by_id.items()}
    mutation_tokens_by_id = {
        candidate_id: canonical_mutation_tokens(row.get("canonical_mutations"))
        for candidate_id, row in candidate_by_id.items()
    }
    mutation_positions_by_id = {
        candidate_id: canonical_mutation_positions(row.get("canonical_mutations"))
        for candidate_id, row in candidate_by_id.items()
    }
    eligible_by_policy = {
        policy_id: [row for row in contract_rows if str(row.get("policy_id") or "") == policy_id]
        for policy_id in PANEL_POLICY_IDS
    }
    _validate_policy_pool_sizes(eligible_by_policy)

    selected_by_policy = {
        policy_id: list(
            select_dissimilar_pair(
                candidate_rows=eligible_by_policy[policy_id],
                mutation_tokens_by_id=mutation_tokens_by_id,
                mutation_positions_by_id=mutation_positions_by_id,
            )
        )
        for policy_id in PANEL_POLICY_IDS
    }
    for allocation in GROUP_ALLOCATIONS:
        for _ in range(allocation.selected_count - 2):
            policy_id = allocation.policy_id
            selected_ids = {str(row["candidate_id"]) for row in selected_by_policy[policy_id]}
            remaining = [row for row in eligible_by_policy[policy_id] if str(row["candidate_id"]) not in selected_ids]
            chosen, _nearest_distance = choose_farthest_candidate(
                candidate_rows=remaining,
                selected_rows=selected_by_policy[policy_id],
                sequence_by_id=sequence_by_id,
                mutation_tokens_by_id=mutation_tokens_by_id,
                mutation_positions_by_id=mutation_positions_by_id,
            )
            selected_by_policy[policy_id].append(chosen)

    panel_rows = _build_ordered_panel_rows(
        selected_by_policy=selected_by_policy,
        candidate_by_id=candidate_by_id,
        sequence_by_id=sequence_by_id,
        mutation_tokens_by_id=mutation_tokens_by_id,
        mutation_positions_by_id=mutation_positions_by_id,
        input_hashes=input_hashes,
    )
    validate_selected_panel(panel_rows)
    return panel_rows


def _validated_candidate_lookup(
    candidate_rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
    duplicates = sorted(candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Selected-panel candidate pool contains duplicate candidate ids: {_format_list(duplicates)}.")
    return {str(row["candidate_id"]): row for row in candidate_rows}


def _validated_contract_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    candidate_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    contract_rows = [row for row in triage_rows if bool(row.get("selection_contract_pass"))]
    unknown_policy_ids = sorted({str(row.get("policy_id") or "") for row in contract_rows} - set(PANEL_POLICY_IDS))
    if unknown_policy_ids:
        raise ValueError(
            "Selected panel received contract-pass rows with unsupported policy ids: "
            f"{_format_list(unknown_policy_ids)}."
        )
    missing_candidate_ids = sorted(
        str(row["candidate_id"]) for row in contract_rows if str(row["candidate_id"]) not in candidate_by_id
    )
    if missing_candidate_ids:
        raise ValueError(
            "Selected-panel contract-pass rows are absent from the candidate pool: "
            f"{_format_list(missing_candidate_ids)}."
        )
    policy_mismatches = sorted(
        str(row["candidate_id"])
        for row in contract_rows
        if str(candidate_by_id[str(row["candidate_id"])].get("policy_id") or "") != str(row.get("policy_id") or "")
    )
    if policy_mismatches:
        raise ValueError(
            "Selected-panel policy provenance differs between triage and candidate rows: "
            f"{_format_list(policy_mismatches)}."
        )
    return contract_rows


def _validate_policy_pool_sizes(eligible_by_policy: dict[str, list[dict[str, object]]]) -> None:
    for policy_id, rows in eligible_by_policy.items():
        required = allocation_for_policy(policy_id).selected_count
        if len(rows) < required:
            raise ValueError(
                f"Selected panel requires {required} contract-pass rows for {policy_id}, but found {len(rows)}."
            )


def _build_ordered_panel_rows(
    *,
    selected_by_policy: dict[str, list[dict[str, object]]],
    candidate_by_id: dict[str, dict[str, object]],
    sequence_by_id: dict[str, str],
    mutation_tokens_by_id: dict[str, frozenset[str]],
    mutation_positions_by_id: dict[str, frozenset[int]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    ordered: list[tuple[dict[str, object], int]] = [
        (row, rank)
        for allocation in GROUP_ALLOCATIONS
        for rank, row in enumerate(selected_by_policy[allocation.policy_id], start=1)
    ]

    panel_rows = []
    for selection_rank, (selected_row, within_group_rank) in enumerate(ordered, start=1):
        policy_id = str(selected_row["policy_id"])
        allocation = allocation_for_policy(policy_id)
        peers = [row for row in selected_by_policy[policy_id] if row["candidate_id"] != selected_row["candidate_id"]]
        audited = with_nearest_mutation_audit(
            selected_row,
            peer_rows=peers,
            mutation_tokens_by_id=mutation_tokens_by_id,
            mutation_positions_by_id=mutation_positions_by_id,
        )
        audited["canonical_mutations"] = list(
            candidate_by_id[str(selected_row["candidate_id"])].get("canonical_mutations") or []
        )
        panel_rows.append(
            build_panel_row(
                audited,
                within_group_nearest_sequence_distance=nearest_sequence_distance(
                    selected_row,
                    peer_rows=peers,
                    sequence_by_id=sequence_by_id,
                ),
                input_hashes=input_hashes,
                selection_rank=selection_rank,
                design_group_id=allocation.design_group_id,
                within_group_rank=within_group_rank,
            )
        )
    return panel_rows


def validate_selected_panel(panel_rows: Sequence[dict[str, object]]) -> None:
    """Fail unless the selected-panel allocation is complete and unique."""

    candidate_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    duplicates = sorted(candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1)
    policy_counts = Counter(str(row.get("policy_id") or "") for row in panel_rows)
    selection_ranks = sorted(int(row["selection_rank"]) for row in panel_rows)
    invalid_contract_rows = [
        str(row.get("candidate_id") or "") for row in panel_rows if not bool(row.get("selection_contract_pass"))
    ]
    if (
        len(panel_rows) == SELECTED_PANEL_SIZE
        and not duplicates
        and not invalid_contract_rows
        and dict(policy_counts) == EXPECTED_SELECTED_POLICY_COUNTS
        and selection_ranks == list(range(1, SELECTED_PANEL_SIZE + 1))
    ):
        return
    raise ValueError(
        "Selected-panel validation failed: "
        f"expected policy counts {EXPECTED_SELECTED_POLICY_COUNTS}, observed {dict(policy_counts)}; "
        f"selection ranks: {selection_ranks}; "
        f"duplicate candidate ids: {_format_list(duplicates)}; "
        f"contract-failing rows: {_format_list(invalid_contract_rows)}."
    )


def selected_panel_coverage_summary(panel_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return manifest-ready selected-panel composition fields."""

    candidate_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    policy_counts = Counter(str(row.get("policy_id") or "") for row in panel_rows)
    duplicate_candidate_ids = sorted(
        candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1
    )
    contract_failure_ids = [
        str(row.get("candidate_id") or "") for row in panel_rows if not bool(row.get("selection_contract_pass"))
    ]
    return {
        "selected_panel_size": SELECTED_PANEL_SIZE,
        "selected_row_count": len(panel_rows),
        "policy_allocation_role": "experimental_design",
        "selected_generation_policy_counts": {key: policy_counts[key] for key in sorted(policy_counts)},
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "contract_failure_candidate_ids": contract_failure_ids,
        "valid": (
            len(panel_rows) == SELECTED_PANEL_SIZE
            and dict(policy_counts) == EXPECTED_SELECTED_POLICY_COUNTS
            and not duplicate_candidate_ids
            and not contract_failure_ids
        ),
    }


def _format_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"
