"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/manifest.py

Vendor-neutral synthesis manifest construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

from dnadesign.opal import scan_restriction_sites

from .contracts import (
    CloningStrategy,
    SelectedCandidate,
    SelectionMembership,
    optional_nonnegative_integer,
    require_nonnegative_integer,
    require_positive_integer,
    validate_promoter_core,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _candidate_from(value: SelectedCandidate | Mapping[str, Any]) -> SelectedCandidate:
    if isinstance(value, SelectedCandidate):
        return value
    if isinstance(value, Mapping):
        return SelectedCandidate(
            campaign_slug=str(value["campaign_slug"]),
            selection_memberships=tuple(
                SelectionMembership.from_mapping(row) for row in value["selection_memberships"]
            ),
            as_of_round=require_nonnegative_integer(value["as_of_round"], field="as_of_round"),
            run_id=str(value["run_id"]),
            selection_rank=require_positive_integer(value["selection_rank"], field="selection_rank"),
            id=str(value["id"]),
            sequence=str(value["sequence"]),
            synthesis_name=str(value["synthesis_name"]),
            selection_source=str(value.get("selection_source", "selected_csv")),
            selection_epoch=str(value.get("selection_epoch", "opal_model_round")),
            assay_batch_index=optional_nonnegative_integer(
                value.get("assay_batch_index"),
                field="assay_batch_index",
            ),
            model_as_of_round=optional_nonnegative_integer(
                value.get("model_as_of_round"),
                field="model_as_of_round",
            ),
        )
    if is_dataclass(value):
        return _candidate_from(asdict(value))
    raise TypeError(f"unsupported selected candidate row: {type(value).__name__}")


def _validate_uniqueness(candidates: list[SelectedCandidate]) -> None:
    seen_ids: set[str] = set()
    seen_aliases: set[str] = set()
    seen_sequences: set[str] = set()
    for candidate in candidates:
        if candidate.id in seen_ids:
            raise ValueError(f"duplicate candidate id in synthesis batch: {candidate.id}")
        seen_ids.add(candidate.id)
        if candidate.synthesis_name in seen_aliases:
            raise ValueError(f"duplicate synthesis_name in synthesis batch: {candidate.synthesis_name}")
        seen_aliases.add(candidate.synthesis_name)
        sequence = candidate.sequence.upper()
        if sequence in seen_sequences:
            raise ValueError(f"duplicate promoter sequence in synthesis batch: {candidate.id}")
        seen_sequences.add(sequence)


def _restriction_site_summary(report: Any) -> str:
    return "; ".join(
        f"{hit.enzyme}:{hit.motif}@{hit.start_0}-{hit.end_0}:{hit.region}" for hit in report.unexpected_hits
    )


def build_synthesis_manifest(
    *,
    selected: Iterable[SelectedCandidate | Mapping[str, Any]],
    strategy: CloningStrategy,
    batch_id: str,
) -> pd.DataFrame:
    """Build a vendor-neutral synthesis manifest from selected OPAL rows."""

    batch = str(batch_id).strip()
    if not batch:
        raise ValueError("batch_id must be non-empty")
    candidates = [_candidate_from(row) for row in selected]
    if not candidates:
        raise ValueError("synthesis batch requires at least one selected candidate")
    _validate_uniqueness(candidates)

    rows: list[dict[str, Any]] = []
    core_start = len(strategy.left_flank)
    for candidate in candidates:
        core_sequence = validate_promoter_core(
            candidate.sequence,
            expected_length=strategy.expected_core_length,
            candidate_id=candidate.id,
        )
        final_sequence = f"{strategy.left_flank}{core_sequence}{strategy.right_flank}"
        core_end = core_start + len(core_sequence)
        if strategy.restriction_sites:
            restriction_report = scan_restriction_sites(
                candidate_id=candidate.id,
                core_sequence=core_sequence,
                left_flank=strategy.left_flank,
                right_flank=strategy.right_flank,
                expected_core_length=strategy.expected_core_length,
                forbidden_sites=strategy.restriction_sites,
            )
            if restriction_report.unexpected_hits:
                raise ValueError(
                    f"candidate {candidate.id} has unexpected restriction site(s) in assembled insert: "
                    f"{_restriction_site_summary(restriction_report)}"
                )
        rows.append(
            {
                "batch_id": batch,
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "strategy_version": strategy.version,
                "campaign_slug": candidate.campaign_slug,
                "selection_view_ids": json.dumps(candidate.selection_view_ids, separators=(",", ":")),
                "selection_memberships": json.dumps(
                    [asdict(row) for row in candidate.selection_memberships],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "as_of_round": candidate.as_of_round,
                "run_id": candidate.run_id,
                "selection_source": candidate.selection_source,
                "selection_epoch": candidate.selection_epoch,
                "assay_batch_index": candidate.assay_batch_index,
                "model_as_of_round": candidate.model_as_of_round,
                "selection_rank": candidate.selection_rank,
                "id": candidate.id,
                "synthesis_name": candidate.synthesis_name,
                "core_sequence": core_sequence,
                "left_flank": strategy.left_flank,
                "right_flank": strategy.right_flank,
                "core_start": core_start,
                "core_end": core_end,
                "core_length": len(core_sequence),
                "final_sequence": final_sequence,
                "final_length": len(final_sequence),
                "expected_final_length": strategy.expected_final_length,
                "core_sha256": _sha256_text(core_sequence),
                "final_sha256": _sha256_text(final_sequence),
                "validation_status": "pass",
                "restriction_site_validation_status": "pass" if strategy.restriction_sites else "not_configured",
            }
        )

    manifest = pd.DataFrame(rows)
    if not (manifest["final_length"] == manifest["expected_final_length"]).all():
        raise ValueError("final sequence length does not match cloning strategy expectation")
    return manifest
