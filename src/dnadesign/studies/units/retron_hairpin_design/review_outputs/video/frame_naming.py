"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/video/frame_naming.py

Review-frame naming for Retron sequence montage outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError
from ..sequence.index import SequenceReviewFrame
from ..sequence.variant_identity import identity_for_frame, payload_window, variant_id


def frame_filename_stem(frame: SequenceReviewFrame, *, review_variant_ids: Mapping[str, str]) -> str:
    construct_id = review_construct_id(frame, review_variant_ids=review_variant_ids)
    return f"{frame.order:02d}_{construct_id}_{payload_window_slug(frame)}"


def frame_evidence_label(frame: SequenceReviewFrame, *, review_variant_ids: Mapping[str, str]) -> str:
    identity = identity_for_frame(frame)
    construct_id = review_construct_id(frame, review_variant_ids=review_variant_ids)
    return f"{construct_id} | {identity.payload_label} | {identity.scaffold} scaffold | {identity.insert_nt} nt"


def payload_window_slug(frame: SequenceReviewFrame) -> str:
    start, end = payload_window(frame.payload_trim_id)
    return f"tetO-w{start:02d}-{end:02d}"


def review_construct_id(frame: SequenceReviewFrame, *, review_variant_ids: Mapping[str, str]) -> str:
    compact_variant_id = variant_id(frame)
    try:
        return review_variant_ids[compact_variant_id]
    except KeyError as exc:
        raise RetronMsdCompilerError(
            f"Retron review_variant_ids has no review construct id for {compact_variant_id}"
        ) from exc


__all__ = ["frame_evidence_label", "frame_filename_stem", "payload_window_slug", "review_construct_id"]
