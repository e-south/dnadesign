"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence/variant_identity.py

Compact reviewer-facing variant identity for Retron review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...compiler.exceptions import RetronMsdCompilerError
from .index import SequenceReviewFrame

CONSTRUCT_PREFIX = "pES-tetr-"
PAYLOAD_WINDOW_RE = re.compile(r"^TetR_w(?P<start>\d{2})_(?P<end>\d{2})$")


@dataclass(frozen=True)
class ReviewVariantIdentity:
    variant_id: str
    scaffold: str
    retained_window: str
    insert_nt: int
    payload_label: str
    role: str


def identity_for_frame(frame: SequenceReviewFrame) -> ReviewVariantIdentity:
    window = payload_window(frame.payload_trim_id)
    retained_window = f"[{window[0]},{window[1]})"
    scaffold = scaffold_label(frame.scaffold_context)
    return ReviewVariantIdentity(
        variant_id=variant_id(frame),
        scaffold=scaffold,
        retained_window=retained_window,
        insert_nt=window[1] - window[0],
        payload_label=f"tetO PWM {retained_window}",
        role=role_label(frame.variant_role),
    )


def variant_id(frame: SequenceReviewFrame) -> str:
    if frame.construct_id.startswith(CONSTRUCT_PREFIX):
        return frame.construct_id.removeprefix(CONSTRUCT_PREFIX)
    start, end = payload_window(frame.payload_trim_id)
    return f"{scaffold_label(frame.scaffold_context)}-w{start:02d}-{end:02d}"


def payload_window(payload_trim_id: str) -> tuple[int, int]:
    match = PAYLOAD_WINDOW_RE.match(payload_trim_id)
    if match is None:
        raise RetronMsdCompilerError(
            f"Retron review payload_trim_id must use TetR_w00_19-style window form: {payload_trim_id}"
        )
    return int(match.group("start")), int(match.group("end"))


def scaffold_label(scaffold_context: str) -> str:
    return {
        "retron26": "r26",
        "retron43": "r43",
        "retron180": "r180",
    }.get(scaffold_context, _slug(scaffold_context))


def role_label(variant_role: str) -> str:
    return {
        "scaffold_target": "target",
        "trim_candidate": "candidate",
    }.get(variant_role, variant_role)


def _slug(value: str) -> str:
    return "-".join(part for part in re.split(r"[^A-Za-z0-9]+", value.lower()) if part)


__all__ = [
    "ReviewVariantIdentity",
    "identity_for_frame",
    "payload_window",
    "role_label",
    "scaffold_label",
    "variant_id",
]
