"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/materialize/identities.py

Reader identity selection for reporter-response materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from ....reader_evidence import ReaderEvidenceBindingSet


def _observed_reader_identities(frame: pd.DataFrame) -> set[tuple[str, str | None]]:
    """Return exact design and optional assay-subject identities in a frame."""

    assay_subjects = (
        frame["assay_subject_id"]
        if "assay_subject_id" in frame.columns
        else pd.Series((None,) * len(frame), index=frame.index, dtype=object)
    )
    return {
        (str(design_id), None if pd.isna(assay_subject_id) else str(assay_subject_id))
        for design_id, assay_subject_id in zip(frame["design_id"], assay_subjects, strict=True)
    }


def _has_ambiguous_partial_identity(
    observed_identities: set[tuple[str, str | None]],
    *,
    bindings: ReaderEvidenceBindingSet,
) -> bool:
    """Return whether a design-only identity matches multiple source bindings."""

    binding_design_ids = tuple(row.raw_design_id for row in bindings.rows)
    return any(
        assay_subject_id is None and binding_design_ids.count(design_id) > 1
        for design_id, assay_subject_id in observed_identities
    )


def _reader_identity_mask(frame: pd.DataFrame, identity: tuple[str, str | None]) -> pd.Series:
    """Select rows matching one exact Reader identity."""

    design_id, assay_subject_id = identity
    mask = frame["design_id"].astype(str).eq(design_id)
    if "assay_subject_id" not in frame.columns:
        return mask if assay_subject_id is None else mask & False
    if assay_subject_id is None:
        return mask & frame["assay_subject_id"].isna()
    return mask & frame["assay_subject_id"].astype(str).eq(assay_subject_id)
