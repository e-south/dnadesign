"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/pairing_qa.py

Pairing QA helpers for backend-neutral secondary-structure predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.folding import SecondaryStructurePredictionV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import SecondaryStructureIntendedPairingQaV1
from dnadesign.contracts.visual import SequenceEvidenceMapV1


def cross_copy_pairing_annotations(
    prediction: SecondaryStructurePredictionV1,
    *,
    unit_copy_spans: tuple[dict[str, int | str], ...],
) -> list[dict[str, object]]:
    if prediction.result is None:
        return []
    pairings: list[dict[str, object]] = []
    for pair in prediction.result.pair_map:
        left_copy = copy_for_index(pair.left, unit_copy_spans)
        right_copy = copy_for_index(pair.right, unit_copy_spans)
        if left_copy is None or right_copy is None:
            continue
        if (left_copy["unit_id"], left_copy["copy_index"]) == (right_copy["unit_id"], right_copy["copy_index"]):
            continue
        pairings.append(
            {
                "left_index_0": pair.left,
                "right_index_0": pair.right,
                "pair": pair.pair,
                "left_unit_id": left_copy["unit_id"],
                "left_copy_index": left_copy["copy_index"],
                "right_unit_id": right_copy["unit_id"],
                "right_copy_index": right_copy["copy_index"],
            }
        )
    return pairings


def intended_pairing_qa(
    visual_contract: SequenceEvidenceMapV1,
    *,
    predicted_pairs: set[tuple[int, int]],
) -> list[SecondaryStructureIntendedPairingQaV1]:
    intended: list[SecondaryStructureIntendedPairingQaV1] = []
    for pairing in visual_contract.pairings:
        expected_pairs = set(expected_pair_keys(pairing))
        recovered_count = len(expected_pairs.intersection(predicted_pairs))
        if recovered_count == len(expected_pairs):
            status = "fully_recovered"
        elif recovered_count:
            status = "partially_recovered"
        else:
            status = "missed"
        intended.append(
            SecondaryStructureIntendedPairingQaV1(
                pairing_id=pairing.pairing_id,
                primary_start=pairing.primary_start,
                primary_end=pairing.primary_end,
                complement_start=pairing.complement_start,
                complement_end=pairing.complement_end,
                expected_pair_count=len(expected_pairs),
                predicted_pair_count=recovered_count,
                status=status,
            )
        )
    return intended


def intended_pair_lookup(visual_contract: SequenceEvidenceMapV1 | None) -> dict[tuple[int, int], tuple[str, ...]]:
    if visual_contract is None:
        return {}
    mutable: dict[tuple[int, int], list[str]] = {}
    for pairing in visual_contract.pairings:
        for key in expected_pair_keys(pairing):
            mutable.setdefault(key, []).append(pairing.pairing_id)
    return {key: tuple(value) for key, value in mutable.items()}


def expected_pair_keys(pairing: object) -> tuple[tuple[int, int], ...]:
    primary_start = int(getattr(pairing, "primary_start"))
    primary_end = int(getattr(pairing, "primary_end"))
    complement_start = int(getattr(pairing, "complement_start"))
    complement_end = int(getattr(pairing, "complement_end"))
    span_length = min(primary_end - primary_start, complement_end - complement_start)
    return tuple(pair_key(primary_start + offset, complement_end - 1 - offset) for offset in range(span_length))


def pair_key(left: int, right: int) -> tuple[int, int]:
    if right < left:
        return right, left
    return left, right


def unit_copy_spans(visual_contract: SequenceEvidenceMapV1 | None) -> tuple[dict[str, int | str], ...]:
    if visual_contract is None:
        return ()
    raw = visual_contract.meta.get("unit_copies")
    if not isinstance(raw, list):
        return ()
    spans: list[dict[str, int | str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        span = item.get("span")
        if not isinstance(span, dict):
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
            copy_index = int(item["copy_index"])
            unit_id = str(item["unit_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append({"unit_id": unit_id, "copy_index": copy_index, "start": start, "end": end})
    return tuple(spans)


def copy_for_index(index: int, unit_copy_spans: tuple[dict[str, int | str], ...]) -> dict[str, int | str] | None:
    for span in unit_copy_spans:
        if int(span["start"]) <= index < int(span["end"]):
            return span
    return None


__all__ = [
    "copy_for_index",
    "cross_copy_pairing_annotations",
    "expected_pair_keys",
    "intended_pair_lookup",
    "intended_pairing_qa",
    "pair_key",
    "unit_copy_spans",
]
