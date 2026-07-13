"""Within-policy mutation-set ranking for the selected Eco1 RT panel."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    jaccard_distance,
    nearest_jaccard_distance,
    nearest_shared_count,
)

_RANK_FIELDS = (
    ("nearest_selected_mutation_position_distance", "nearest_selected_mutation_position_jaccard_distance", "higher"),
    ("nearest_selected_exact_substitution_distance", "nearest_selected_mutation_token_jaccard_distance", "higher"),
    ("basic_loss_near_retained_dna_rna", "nucleic_acid_facing_basic_loss_count", "lower"),
    (
        "proline_glycine_gain_near_retained_dna_rna",
        "nucleic_acid_facing_proline_glycine_gain_count",
        "lower",
    ),
    ("selection_msa_observed_fraction", "selection_support_alt_observed_fraction", "higher"),
    ("selection_msa_alt_frequency", "selection_support_alt_frequency_mean", "higher"),
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


def select_dissimilar_pair(
    *,
    candidate_rows: Sequence[dict[str, object]],
    mutation_tokens_by_id: dict[str, frozenset[str]],
    mutation_positions_by_id: dict[str, frozenset[int]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Select the most mutation-set-dissimilar pair under the tie-break order."""

    try:
        left, right = min(
            combinations(candidate_rows, 2),
            key=lambda pair: _dissimilar_pair_sort_key(
                pair,
                mutation_tokens_by_id=mutation_tokens_by_id,
                mutation_positions_by_id=mutation_positions_by_id,
            ),
        )
    except ValueError as exc:
        raise ValueError("Selected-panel pair selection requires at least two candidates per policy.") from exc
    return tuple(sorted((dict(left), dict(right)), key=_late_evidence_sort_key))


def _dissimilar_pair_sort_key(
    pair: tuple[dict[str, object], dict[str, object]],
    *,
    mutation_tokens_by_id: dict[str, frozenset[str]],
    mutation_positions_by_id: dict[str, frozenset[int]],
) -> tuple[object, ...]:
    left, right = pair
    left_id = str(left["candidate_id"])
    right_id = str(right["candidate_id"])
    position_distance = jaccard_distance(
        mutation_positions_by_id.get(left_id, frozenset()),
        mutation_positions_by_id.get(right_id, frozenset()),
    )
    token_distance = jaccard_distance(
        mutation_tokens_by_id.get(left_id, frozenset()),
        mutation_tokens_by_id.get(right_id, frozenset()),
    )
    evidence_keys = tuple(sorted((_late_evidence_sort_key(left), _late_evidence_sort_key(right))))
    return (-position_distance, -token_distance, evidence_keys)


def _late_evidence_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    values: list[object] = []
    for _stage_id, field_name, direction in _RANK_FIELDS[2:]:
        value = row.get(field_name)
        if direction == "lexicographic":
            values.append(str(value or ""))
        elif direction == "lower":
            values.append(_float_value(value, default=9999.0))
        elif direction == "higher":
            values.append(-_float_value(value, default=-9999.0))
        else:
            raise ValueError(f"Unknown selected-panel rank direction: {direction}")
    return tuple(values)


def with_nearest_mutation_audit(
    row: dict[str, object],
    *,
    peer_rows: Sequence[dict[str, object]],
    mutation_tokens_by_id: dict[str, frozenset[str]],
    mutation_positions_by_id: dict[str, frozenset[int]],
) -> dict[str, object]:
    """Attach nearest within-policy mutation overlap fields to a selected row."""

    audited = dict(row)
    candidate_id = str(row["candidate_id"])
    peer_ids = [str(peer["candidate_id"]) for peer in peer_rows]
    candidate_tokens = mutation_tokens_by_id.get(candidate_id, frozenset())
    candidate_positions = mutation_positions_by_id.get(candidate_id, frozenset())
    peer_tokens = [mutation_tokens_by_id.get(peer_id, frozenset()) for peer_id in peer_ids]
    peer_positions = [mutation_positions_by_id.get(peer_id, frozenset()) for peer_id in peer_ids]
    audited["within_group_nearest_exact_substitution_jaccard_distance"] = nearest_jaccard_distance(
        candidate_tokens, peer_tokens
    )
    audited["within_group_nearest_mutated_position_jaccard_distance"] = nearest_jaccard_distance(
        candidate_positions, peer_positions
    )
    audited["within_group_nearest_exact_substitution_shared_count"] = nearest_shared_count(
        candidate_tokens, peer_tokens
    )
    audited["within_group_nearest_mutated_position_shared_count"] = nearest_shared_count(
        candidate_positions, peer_positions
    )
    return audited


def nearest_sequence_distance(
    row: dict[str, object],
    *,
    peer_rows: Sequence[dict[str, object]],
    sequence_by_id: dict[str, str],
) -> int | None:
    """Return the nearest full-sequence Hamming distance within a policy."""

    candidate_id = str(row["candidate_id"])
    peer_sequences = [sequence_by_id.get(str(peer["candidate_id"]), "") for peer in peer_rows]
    return _nearest_distance(sequence_by_id.get(candidate_id, ""), peer_sequences)


def choose_farthest_candidate(
    *,
    candidate_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    sequence_by_id: dict[str, str],
    mutation_tokens_by_id: dict[str, frozenset[str]] | None = None,
    mutation_positions_by_id: dict[str, frozenset[int]] | None = None,
) -> tuple[dict[str, object], int | None]:
    """Choose the next row by mutation-set distance and declared late evidence."""

    selected_sequences = [sequence_by_id[str(row["candidate_id"])] for row in selected_rows]
    mutation_tokens_by_id = mutation_tokens_by_id or {}
    mutation_positions_by_id = mutation_positions_by_id or {}
    selected_token_sets = [mutation_tokens_by_id.get(str(row["candidate_id"]), frozenset()) for row in selected_rows]
    selected_position_sets = [
        mutation_positions_by_id.get(str(row["candidate_id"]), frozenset()) for row in selected_rows
    ]
    selected_token_union = frozenset().union(*selected_token_sets) if selected_token_sets else frozenset()
    selected_position_union = frozenset().union(*selected_position_sets) if selected_position_sets else frozenset()

    audit_by_id = {
        str(row["candidate_id"]): _candidate_distance_audit(
            row,
            selected_sequences=selected_sequences,
            selected_token_sets=selected_token_sets,
            selected_position_sets=selected_position_sets,
            selected_token_union=selected_token_union,
            selected_position_union=selected_position_union,
            sequence_by_id=sequence_by_id,
            mutation_tokens_by_id=mutation_tokens_by_id,
            mutation_positions_by_id=mutation_positions_by_id,
        )
        for row in candidate_rows
    }
    chosen = dict(
        min(
            candidate_rows,
            key=lambda row: _farthest_candidate_sort_key(
                row,
                audit=audit_by_id[str(row["candidate_id"])],
            ),
        )
    )
    audit = audit_by_id[str(chosen["candidate_id"])]
    chosen.update(
        {
            "new_exact_substitution_count_vs_panel": audit["new_mutation_token_count"],
            "new_mutated_position_count_vs_panel": audit["new_mutation_position_count"],
            "shared_exact_substitution_count_vs_panel": audit["shared_mutation_token_count"],
            "shared_mutated_position_count_vs_panel": audit["shared_mutation_position_count"],
            "nearest_selected_mutation_token_jaccard_distance": audit["nearest_mutation_token_jaccard"],
            "nearest_selected_mutation_position_jaccard_distance": audit["nearest_mutation_position_jaccard"],
            "nearest_selected_mutation_token_shared_count": audit["nearest_mutation_token_shared"],
            "nearest_selected_mutation_position_shared_count": audit["nearest_mutation_position_shared"],
        }
    )
    return chosen, audit["nearest_distance"]


def _candidate_distance_audit(
    row: dict[str, object],
    *,
    selected_sequences: list[str],
    selected_token_sets: list[frozenset[str]],
    selected_position_sets: list[frozenset[int]],
    selected_token_union: frozenset[str],
    selected_position_union: frozenset[int],
    sequence_by_id: dict[str, str],
    mutation_tokens_by_id: dict[str, frozenset[str]],
    mutation_positions_by_id: dict[str, frozenset[int]],
) -> dict[str, int | float | None]:
    candidate_id = str(row["candidate_id"])
    tokens = mutation_tokens_by_id.get(candidate_id, frozenset())
    positions = mutation_positions_by_id.get(candidate_id, frozenset())
    return {
        "nearest_distance": _nearest_distance(sequence_by_id.get(candidate_id, ""), selected_sequences),
        "nearest_mutation_token_jaccard": nearest_jaccard_distance(tokens, selected_token_sets),
        "nearest_mutation_position_jaccard": nearest_jaccard_distance(positions, selected_position_sets),
        "nearest_mutation_token_shared": nearest_shared_count(tokens, selected_token_sets),
        "nearest_mutation_position_shared": nearest_shared_count(positions, selected_position_sets),
        "new_mutation_token_count": len(tokens - selected_token_union),
        "new_mutation_position_count": len(positions - selected_position_union),
        "shared_mutation_token_count": len(tokens & selected_token_union),
        "shared_mutation_position_count": len(positions & selected_position_union),
    }


def _farthest_candidate_sort_key(
    row: dict[str, object],
    *,
    audit: dict[str, int | float | None],
) -> tuple[object, ...]:
    dynamic_fields = {
        "nearest_selected_distance_aa": audit["nearest_distance"] or 0,
        "nearest_selected_mutation_token_jaccard_distance": audit["nearest_mutation_token_jaccard"] or 0.0,
        "nearest_selected_mutation_position_jaccard_distance": audit["nearest_mutation_position_jaccard"] or 0.0,
        "nearest_selected_mutation_token_shared_count": audit["nearest_mutation_token_shared"] or 0,
        "nearest_selected_mutation_position_shared_count": audit["nearest_mutation_position_shared"] or 0,
        "new_exact_substitution_count_vs_panel": audit["new_mutation_token_count"] or 0,
        "new_mutated_position_count_vs_panel": audit["new_mutation_position_count"] or 0,
        "shared_exact_substitution_count_vs_panel": audit["shared_mutation_token_count"] or 0,
        "shared_mutated_position_count_vs_panel": audit["shared_mutation_position_count"] or 0,
    }
    values: list[object] = []
    for _stage_id, field_name, direction in _RANK_FIELDS:
        value = dynamic_fields.get(field_name, row.get(field_name))
        if direction == "lexicographic":
            values.append(str(value or ""))
        elif direction == "lower":
            values.append(_float_value(value, default=9999.0))
        elif direction == "higher":
            values.append(-_float_value(value, default=-9999.0))
        else:
            raise ValueError(f"Unknown selected-panel rank direction: {direction}")
    return tuple(values)


def _nearest_distance(sequence: str, selected_sequences: list[str]) -> int | None:
    if not selected_sequences:
        return None
    return min(_hamming_distance(sequence, selected) for selected in selected_sequences)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


def _float_value(value: object, *, default: float = -1.0) -> float:
    return default if value is None else float(value)
