"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/evaluation_baseline.py

Public verification facade for the frozen round-0 MSRB evaluation baseline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .evaluation_baseline_artifacts import file_sha256, verify_baseline_sources
from .evaluation_baseline_contracts import (
    BASELINE_PATH,
    CAMPAIGN_SLUG,
    COMPARISON_ROLE,
    COMPARISON_STATEMENT,
    OBJECTIVE_ID,
    PROTOCOL_ID,
    RUN_ID,
    SCHEMA_ID,
    SCHEMA_VERSION,
    FrozenAllocation,
    FrozenArtifact,
    MsrbEvaluationBaseline,
    MsrbEvaluationBaselineError,
)
from .evaluation_baseline_parser import parse_baseline


def load_msrb_evaluation_baseline(
    repo_root: str | Path,
    *,
    baseline_path: str | Path = BASELINE_PATH,
) -> MsrbEvaluationBaseline:
    """Load the baseline and verify every bound artifact against its frozen claims."""

    root = Path(repo_root).expanduser().resolve()
    source_path = Path(baseline_path).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise MsrbEvaluationBaselineError(f"MSRB evaluation baseline is missing: {source_path}")
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MsrbEvaluationBaselineError(f"MSRB evaluation baseline YAML is invalid: {exc}") from exc

    parsed = parse_baseline(payload, root=root)
    selection_replay = verify_baseline_sources(root, parsed)
    return MsrbEvaluationBaseline(
        schema_id=SCHEMA_ID,
        baseline_id="secg_msrb_round0_evaluation_v1",
        campaign_slug=CAMPAIGN_SLUG,
        run_id=RUN_ID,
        round_index=0,
        campaign_config=parsed.campaign_config,
        selection_allocation_api_version=parsed.selection_allocation_api_version,
        selection_replay=selection_replay,
        prediction_ledger=parsed.prediction_ledger,
        selection_batch=parsed.selection_batch,
        labels_used=parsed.labels_used,
        allocations=parsed.allocations,
        comparison_role=COMPARISON_ROLE,
        comparison_candidate_ids=parsed.comparison_candidate_ids,
        comparison_method=parsed.comparison_method,
        comparison_subset_size=parsed.comparison_subset_size,
        comparison_subset_count=parsed.comparison_subset_count,
        physical_random_control=False,
        comparison_statement=COMPARISON_STATEMENT,
        endpoint_ids=parsed.endpoint_ids,
        acquisition_efficacy_claim=parsed.acquisition_efficacy_claim,
        hill_climb_claim=parsed.hill_climb_claim,
        synthesis_authorization=parsed.synthesis_authorization,
        claim_limit_statement=parsed.claim_limit_statement,
        source_path=source_path,
        source_sha256=file_sha256(source_path),
    )


__all__ = [
    "BASELINE_PATH",
    "CAMPAIGN_SLUG",
    "OBJECTIVE_ID",
    "PROTOCOL_ID",
    "RUN_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "FrozenAllocation",
    "FrozenArtifact",
    "MsrbEvaluationBaseline",
    "MsrbEvaluationBaselineError",
    "load_msrb_evaluation_baseline",
]
