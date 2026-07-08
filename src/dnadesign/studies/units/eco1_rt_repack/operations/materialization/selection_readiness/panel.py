"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/panel.py

Primary-panel selection for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    PRIMARY_PANEL_SIZE,
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
    nearest_jaccard_distance,
    nearest_shared_count,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_rows import (
    build_panel_row,
)

_PRIMARY_RANK_FIELDS = (
    (
        "nearest_selected_mutation_position_jaccard_distance",
        "nearest_selected_mutation_position_jaccard_distance",
        "higher",
    ),
    (
        "nearest_selected_mutation_token_jaccard_distance",
        "nearest_selected_mutation_token_jaccard_distance",
        "higher",
    ),
    (
        "nearest_selected_mutation_position_shared_count",
        "nearest_selected_mutation_position_shared_count",
        "lower",
    ),
    (
        "nearest_selected_mutation_token_shared_count",
        "nearest_selected_mutation_token_shared_count",
        "lower",
    ),
    (
        "basic_loss_near_retained_dna_rna",
        "nucleic_acid_facing_basic_loss_count",
        "lower",
    ),
    (
        "proline_glycine_gain_near_retained_dna_rna",
        "nucleic_acid_facing_proline_glycine_gain_count",
        "lower",
    ),
    (
        "selection_msa_observed_fraction",
        "selection_support_alt_observed_fraction",
        "higher",
    ),
    (
        "selection_msa_alt_frequency",
        "selection_support_alt_frequency_mean",
        "higher",
    ),
    (
        "c_terminal_primer_rna_recognition_rmsd",
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom",
        "lower",
    ),
    (
        "thumb_contact_track_rmsd",
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom",
        "lower",
    ),
    ("mean_plddt", "mean_plddt", "higher"),
    ("cryoem_mapped_rmsd", "cryoem_mapped_ca_rmsd", "lower"),
    ("sequence_hash", "sequence_hash", "lexicographic"),
)


def build_selection_panel_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, object]]:
    """Select the primary conservative panel from globally eligible primary candidates."""

    sequence_by_id = {str(row["candidate_id"]): str(row.get("sequence") or "") for row in candidate_rows}
    mutation_tokens_by_id = {
        str(row["candidate_id"]): canonical_mutation_tokens(row.get("canonical_mutations")) for row in candidate_rows
    }
    mutation_positions_by_id = {
        str(row["candidate_id"]): canonical_mutation_positions(row.get("canonical_mutations")) for row in candidate_rows
    }
    primary_rows = [
        row for row in triage_rows if str(row.get("selection_candidate_tier") or "") == "primary_panel_candidate"
    ]
    if len(primary_rows) < PRIMARY_PANEL_SIZE:
        raise ValueError(
            "Primary panel selection failed: "
            f"requires {PRIMARY_PANEL_SIZE} primary-panel candidates but found {len(primary_rows)}."
        )
    selected: list[dict[str, object]] = []
    remaining = list(primary_rows)
    panel_rows: list[dict[str, object]] = []
    for slot_rank in range(1, PRIMARY_PANEL_SIZE + 1):
        chosen, nearest_distance = _choose_primary_candidate(
            candidate_rows=remaining,
            selected_rows=selected,
            sequence_by_id=sequence_by_id,
            mutation_tokens_by_id=mutation_tokens_by_id,
            mutation_positions_by_id=mutation_positions_by_id,
        )
        selected.append(chosen)
        remaining = [row for row in remaining if str(row["candidate_id"]) != str(chosen["candidate_id"])]
        panel_rows.append(
            build_panel_row(
                chosen,
                nearest_distance=nearest_distance,
                input_hashes=input_hashes,
                slot_rank=slot_rank,
            )
        )
    validate_primary_panel(panel_rows, required_panel_size=PRIMARY_PANEL_SIZE)
    return panel_rows


def build_primary_panel_selection_trace_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    panel_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return stage counts for the primary-panel funnel."""

    all_rows = list(triage_rows)
    trace_rows: list[dict[str, object]] = [
        _trace_row(
            stage_order=1,
            stage_id="candidate_pool",
            stage_label="Accepted candidate pool",
            selector_role="input_pool",
            filter_rule="Accepted ProteinMPNN candidate rows before protein-level selection checks.",
            input_count=len(all_rows),
            remaining_count=len(all_rows),
            is_hard_gate=False,
        )
    ]
    preservation_rows = [row for row in all_rows if str(row.get("hard_gate_status") or "") == "eligible"]
    trace_rows.append(
        _trace_row(
            stage_order=2,
            stage_id="preservation_gate",
            stage_label="Preservation gate",
            selector_role="hard_gate",
            filter_rule=(
                "Keep strong-fold candidates with feasible rows, no protected/core/direct-contact/thumb-track edits, "
                "and passed declared local RMSD gate. The C-terminal primer-RNA recognition region is controlled "
                "through that same local-region threshold table, not a second RMSD overlay."
            ),
            input_count=len(all_rows),
            remaining_count=len(preservation_rows),
            is_hard_gate=True,
        )
    )
    primary_rows = [
        row for row in preservation_rows if str(row.get("selection_candidate_tier") or "") == "primary_panel_candidate"
    ]
    trace_rows.append(
        _trace_row(
            stage_order=3,
            stage_id="chemistry_support_gate",
            stage_label="Chemistry and support gate",
            selector_role="hard_gate",
            filter_rule=(
                "Keep preservation-pass rows with zero acidic gains near retained DNA/RNA and zero unobserved "
                "proximal substitutions. Basic losses and Pro/Gly gains remain ranking penalties, not separate "
                "hard gates."
            ),
            input_count=len(preservation_rows),
            remaining_count=len(primary_rows),
            is_hard_gate=True,
        )
    )
    trace_rows.append(
        _trace_row(
            stage_order=4,
            stage_id="global_conservative_diverse_selection",
            stage_label="Conservative-diverse six-row selection",
            selector_role="global_rank",
            filter_rule=(
                "Select six rows globally from primary candidates by simple mutation-set dissimilarity to already "
                "selected rows, fewer near retained DNA/RNA basic losses and Pro/Gly gains, regional MSA support, "
                "local RMSD values inside the gate, fold metrics, and a deterministic tie-break. Design class is "
                "context, not a quota."
            ),
            input_count=len(primary_rows),
            remaining_count=len(panel_rows),
            is_hard_gate=False,
        )
    )
    return trace_rows


def validate_primary_panel(
    panel_rows: Sequence[dict[str, object]],
    *,
    required_panel_size: int = PRIMARY_PANEL_SIZE,
) -> None:
    """Fail unless the selected primary panel has the required size and unique candidate ids."""

    candidate_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    duplicates = sorted(candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1)
    wrong_tier = [
        str(row.get("candidate_id") or "")
        for row in panel_rows
        if str(row.get("selection_candidate_tier") or "") != "primary_panel_candidate"
    ]
    if len(panel_rows) == required_panel_size and not duplicates and not wrong_tier:
        return
    raise ValueError(
        "Primary panel validation failed: "
        f"expected {required_panel_size} selected rows. Selected rows: {len(panel_rows)}. "
        f"Duplicate candidate ids: {_format_list(duplicates)}. "
        f"Non-primary selected rows: {_format_list(wrong_tier)}."
    )


def panel_coverage_summary(panel_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return manifest-ready primary-panel coverage fields."""

    candidate_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    design_class_counts = Counter(str(row.get("design_class_id") or "") for row in panel_rows)
    duplicate_candidate_ids = sorted(
        candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1
    )
    non_primary = [
        str(row.get("candidate_id") or "")
        for row in panel_rows
        if str(row.get("selection_candidate_tier") or "") != "primary_panel_candidate"
    ]
    return {
        "required_primary_panel_size": PRIMARY_PANEL_SIZE,
        "selected_row_count": len(panel_rows),
        "design_class_quota_enforced": False,
        "selected_design_class_counts": {key: design_class_counts[key] for key in sorted(design_class_counts)},
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "non_primary_selected_candidate_ids": non_primary,
        "valid": len(panel_rows) == PRIMARY_PANEL_SIZE and not duplicate_candidate_ids and not non_primary,
    }


def _choose_primary_candidate(
    *,
    candidate_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    sequence_by_id: dict[str, str],
    mutation_tokens_by_id: dict[str, frozenset[str]] | None = None,
    mutation_positions_by_id: dict[str, frozenset[int]] | None = None,
) -> tuple[dict[str, object], int | None]:
    selected_sequences = [sequence_by_id[str(row["candidate_id"])] for row in selected_rows]
    mutation_tokens_by_id = mutation_tokens_by_id or {}
    mutation_positions_by_id = mutation_positions_by_id or {}
    selected_token_sets = [mutation_tokens_by_id.get(str(row["candidate_id"]), frozenset()) for row in selected_rows]
    selected_position_sets = [
        mutation_positions_by_id.get(str(row["candidate_id"]), frozenset()) for row in selected_rows
    ]
    nearest_distance_by_id = {
        str(row["candidate_id"]): _nearest_distance(
            sequence_by_id.get(str(row["candidate_id"]), ""),
            selected_sequences,
        )
        for row in candidate_rows
    }
    nearest_token_jaccard_by_id = {
        str(row["candidate_id"]): nearest_jaccard_distance(
            mutation_tokens_by_id.get(str(row["candidate_id"]), frozenset()),
            selected_token_sets,
        )
        for row in candidate_rows
    }
    nearest_position_jaccard_by_id = {
        str(row["candidate_id"]): nearest_jaccard_distance(
            mutation_positions_by_id.get(str(row["candidate_id"]), frozenset()),
            selected_position_sets,
        )
        for row in candidate_rows
    }
    nearest_token_shared_by_id = {
        str(row["candidate_id"]): nearest_shared_count(
            mutation_tokens_by_id.get(str(row["candidate_id"]), frozenset()),
            selected_token_sets,
        )
        for row in candidate_rows
    }
    nearest_position_shared_by_id = {
        str(row["candidate_id"]): nearest_shared_count(
            mutation_positions_by_id.get(str(row["candidate_id"]), frozenset()),
            selected_position_sets,
        )
        for row in candidate_rows
    }
    chosen = min(
        candidate_rows,
        key=lambda row: _primary_sort_key(
            row,
            nearest_distance=nearest_distance_by_id[str(row["candidate_id"])],
            nearest_mutation_token_jaccard=nearest_token_jaccard_by_id[str(row["candidate_id"])],
            nearest_mutation_position_jaccard=nearest_position_jaccard_by_id[str(row["candidate_id"])],
            nearest_mutation_token_shared=nearest_token_shared_by_id[str(row["candidate_id"])],
            nearest_mutation_position_shared=nearest_position_shared_by_id[str(row["candidate_id"])],
        ),
    )
    chosen["nearest_selected_mutation_token_jaccard_distance"] = nearest_token_jaccard_by_id[
        str(chosen["candidate_id"])
    ]
    chosen["nearest_selected_mutation_position_jaccard_distance"] = nearest_position_jaccard_by_id[
        str(chosen["candidate_id"])
    ]
    chosen["nearest_selected_mutation_token_shared_count"] = nearest_token_shared_by_id[str(chosen["candidate_id"])]
    chosen["nearest_selected_mutation_position_shared_count"] = nearest_position_shared_by_id[
        str(chosen["candidate_id"])
    ]
    return chosen, nearest_distance_by_id[str(chosen["candidate_id"])]


def _primary_sort_key(
    row: dict[str, object],
    *,
    nearest_distance: int | None,
    nearest_mutation_token_jaccard: float | None = None,
    nearest_mutation_position_jaccard: float | None = None,
    nearest_mutation_token_shared: int | None = None,
    nearest_mutation_position_shared: int | None = None,
) -> tuple[object, ...]:
    values: list[object] = []
    for _stage_id, field_name, direction in _PRIMARY_RANK_FIELDS:
        if field_name == "nearest_selected_distance_aa":
            value: object = nearest_distance if nearest_distance is not None else 0
        elif field_name == "nearest_selected_mutation_token_jaccard_distance":
            value = nearest_mutation_token_jaccard if nearest_mutation_token_jaccard is not None else 0.0
        elif field_name == "nearest_selected_mutation_position_jaccard_distance":
            value = nearest_mutation_position_jaccard if nearest_mutation_position_jaccard is not None else 0.0
        elif field_name == "nearest_selected_mutation_token_shared_count":
            value = nearest_mutation_token_shared if nearest_mutation_token_shared is not None else 0
        elif field_name == "nearest_selected_mutation_position_shared_count":
            value = nearest_mutation_position_shared if nearest_mutation_position_shared is not None else 0
        else:
            value = row.get(field_name)
        if direction == "lexicographic":
            values.append(str(value or ""))
        elif direction == "lower":
            values.append(_float_value(value, default=9999.0))
        elif direction == "higher":
            values.append(-_float_value(value, default=-9999.0))
        else:
            raise ValueError(f"Unknown primary-panel rank direction: {direction}")
    return tuple(values)


def _trace_row(
    *,
    stage_order: int,
    stage_id: str,
    stage_label: str,
    selector_role: str,
    filter_rule: str,
    input_count: int,
    remaining_count: int,
    is_hard_gate: bool,
) -> dict[str, object]:
    return {
        "selection_policy_id": SELECTION_POLICY_ID,
        "stage_order": stage_order,
        "stage_id": stage_id,
        "stage_label": stage_label,
        "selector_role": selector_role,
        "filter_rule": filter_rule,
        "input_count": input_count,
        "removed_count": max(input_count - remaining_count, 0),
        "remaining_count": remaining_count,
        "is_hard_gate": is_hard_gate,
    }


def _nearest_distance(sequence: str, selected_sequences: list[str]) -> int | None:
    if not selected_sequences:
        return None
    return min(_hamming_distance(sequence, selected) for selected in selected_sequences)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


def _format_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _float_value(value: object, *, default: float = -1.0) -> float:
    return default if value is None else float(value)


def _int_value(value: object, *, default: int = 0) -> int:
    return default if value is None else int(value)
