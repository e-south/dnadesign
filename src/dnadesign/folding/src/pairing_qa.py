"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/pairing_qa.py

Pairing QA helpers for backend-neutral secondary-structure predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from dnadesign.contracts.folding import SecondaryStructurePredictionV2
from dnadesign.contracts.folding.secondary_structure_prediction_v2 import (
    SecondaryStructureContiguousWatsonCrickStemRunV1,
    SecondaryStructureIntendedPairingQaV1,
)
from dnadesign.contracts.visual import SequenceEvidenceMapV1

_WATSON_CRICK_PAIRS = frozenset({"AU", "UA", "AT", "TA", "GC", "CG"})
_STEM_METRIC_EXCLUDED_TAG_KINDS = frozenset({"snapback_cap"})


def cross_copy_pairing_annotations(
    prediction: SecondaryStructurePredictionV2,
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
    predicted_pair_labels: Mapping[tuple[int, int], str],
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
        stem_runs = contiguous_watson_crick_stem_runs(
            pairing,
            predicted_pair_labels=predicted_pair_labels,
            sequence_length=len(visual_contract.primary_sequence),
            excluded_indices=stem_metric_excluded_indices(visual_contract),
        )
        predicted_watson_crick_count = sum(run.length_bp for run in stem_runs)
        intended.append(
            SecondaryStructureIntendedPairingQaV1(
                pairing_id=pairing.pairing_id,
                primary_start=pairing.primary_start,
                primary_end=pairing.primary_end,
                complement_start=pairing.complement_start,
                complement_end=pairing.complement_end,
                expected_pair_count=len(expected_pairs),
                predicted_pair_count=recovered_count,
                predicted_watson_crick_pair_count=predicted_watson_crick_count,
                contiguous_watson_crick_stem_bp=max((run.length_bp for run in stem_runs), default=0),
                contiguous_watson_crick_stem_runs=stem_runs,
                status=status,
            )
        )
    return intended


def contiguous_watson_crick_stem_runs(
    pairing: object,
    *,
    predicted_pair_labels: Mapping[tuple[int, int], str],
    sequence_length: int | None = None,
    excluded_indices: set[int] | frozenset[int] = frozenset(),
) -> list[SecondaryStructureContiguousWatsonCrickStemRunV1]:
    primary_start = int(getattr(pairing, "primary_start"))
    primary_end = int(getattr(pairing, "primary_end"))
    complement_start = int(getattr(pairing, "complement_start"))
    complement_end = int(getattr(pairing, "complement_end"))
    span_length = min(primary_end - primary_start, complement_end - complement_start)
    offset_start = 0
    offset_end = span_length
    if sequence_length is not None:
        offset = -1
        while _is_watson_crick_offset(
            offset,
            primary_start=primary_start,
            complement_end=complement_end,
            sequence_length=sequence_length,
            excluded_indices=excluded_indices,
            predicted_pair_labels=predicted_pair_labels,
        ):
            offset_start = offset
            offset -= 1
        offset = span_length
        while _is_watson_crick_offset(
            offset,
            primary_start=primary_start,
            complement_end=complement_end,
            sequence_length=sequence_length,
            excluded_indices=excluded_indices,
            predicted_pair_labels=predicted_pair_labels,
        ):
            offset_end = offset + 1
            offset += 1
    matching_offsets = [
        offset
        for offset in range(offset_start, offset_end)
        if _is_watson_crick_offset(
            offset,
            primary_start=primary_start,
            complement_end=complement_end,
            sequence_length=sequence_length,
            excluded_indices=excluded_indices,
            predicted_pair_labels=predicted_pair_labels,
        )
    ]
    if not matching_offsets:
        return []

    runs: list[SecondaryStructureContiguousWatsonCrickStemRunV1] = []
    run_start = matching_offsets[0]
    previous = matching_offsets[0]
    for offset in matching_offsets[1:]:
        if offset == previous + 1:
            previous = offset
            continue
        _append_payload_anchored_stem_run(
            runs,
            run_start=run_start,
            run_end=previous + 1,
            span_length=span_length,
            primary_start=primary_start,
            complement_end=complement_end,
        )
        run_start = offset
        previous = offset
    _append_payload_anchored_stem_run(
        runs,
        run_start=run_start,
        run_end=previous + 1,
        span_length=span_length,
        primary_start=primary_start,
        complement_end=complement_end,
    )
    return runs


def stem_metric_excluded_indices(visual_contract: SequenceEvidenceMapV1) -> frozenset[int]:
    indices: set[int] = set()
    for tag in visual_contract.effect_tags:
        if tag.row_id != "primary" or tag.tag_kind not in _STEM_METRIC_EXCLUDED_TAG_KINDS:
            continue
        indices.update(range(tag.start, tag.end))
    return frozenset(indices)


def is_watson_crick_pair(pair: str) -> bool:
    return str(pair or "").strip().upper() in _WATSON_CRICK_PAIRS


def _is_watson_crick_offset(
    offset: int,
    *,
    primary_start: int,
    complement_end: int,
    sequence_length: int | None,
    excluded_indices: set[int] | frozenset[int],
    predicted_pair_labels: Mapping[tuple[int, int], str],
) -> bool:
    key = _offset_pair_key(
        offset,
        primary_start=primary_start,
        complement_end=complement_end,
        sequence_length=sequence_length,
        excluded_indices=excluded_indices,
    )
    return key is not None and is_watson_crick_pair(predicted_pair_labels.get(key, ""))


def _offset_pair_key(
    offset: int,
    *,
    primary_start: int,
    complement_end: int,
    sequence_length: int | None,
    excluded_indices: set[int] | frozenset[int],
) -> tuple[int, int] | None:
    left = primary_start + offset
    right = complement_end - 1 - offset
    if left < 0 or right < 0 or left >= right:
        return None
    if sequence_length is not None and (left >= sequence_length or right >= sequence_length):
        return None
    if left in excluded_indices or right in excluded_indices:
        return None
    return pair_key(left, right)


def _append_payload_anchored_stem_run(
    runs: list[SecondaryStructureContiguousWatsonCrickStemRunV1],
    *,
    run_start: int,
    run_end: int,
    span_length: int,
    primary_start: int,
    complement_end: int,
) -> None:
    if run_end <= 0 or run_start >= span_length:
        return
    runs.append(
        _stem_run(
            run_start=run_start,
            run_end=run_end,
            primary_start=primary_start,
            complement_end=complement_end,
        )
    )


def _stem_run(
    *,
    run_start: int,
    run_end: int,
    primary_start: int,
    complement_end: int,
) -> SecondaryStructureContiguousWatsonCrickStemRunV1:
    return SecondaryStructureContiguousWatsonCrickStemRunV1(
        start_offset=run_start,
        end_offset=run_end,
        length_bp=run_end - run_start,
        primary_start=primary_start + run_start,
        primary_end=primary_start + run_end,
        complement_start=complement_end - run_end,
        complement_end=complement_end - run_start,
    )


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
    "contiguous_watson_crick_stem_runs",
    "copy_for_index",
    "cross_copy_pairing_annotations",
    "expected_pair_keys",
    "intended_pair_lookup",
    "intended_pairing_qa",
    "is_watson_crick_pair",
    "pair_key",
    "stem_metric_excluded_indices",
    "unit_copy_spans",
]
