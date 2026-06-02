"""Top-level Stage B execution runner."""

from __future__ import annotations

from .campaigns import run_campaign
from .contracts import TfbsStageBExecutionConfig, TfbsStageBExecutionResult
from .manifest import (
    build_execution_manifest,
    normalize_execution_config,
    read_json,
    selected_campaign_rows,
    validate_config_manifest,
    write_json,
)


def run_tfbs_stage_b_sentinel_campaigns(config: TfbsStageBExecutionConfig) -> TfbsStageBExecutionResult:
    """Run Stage B OPAL campaigns from a validated config manifest."""

    cfg = normalize_execution_config(config)
    manifest = read_json(cfg.config_manifest_path)
    validate_config_manifest(manifest)
    campaigns = selected_campaign_rows(manifest, cfg.campaign_keys)
    if not campaigns:
        raise ValueError("Stage B execution selected zero campaigns")
    round_count = int(cfg.rounds if cfg.rounds is not None else manifest["rounds"])
    if round_count <= 0:
        raise ValueError("Stage B execution rounds must be positive")

    results = [
        run_campaign(campaign, repo_root=cfg.repo_root, rounds=round_count, resume=cfg.resume_existing)
        for campaign in campaigns
    ]
    execution_manifest = build_execution_manifest(
        source_manifest_path=cfg.config_manifest_path,
        source_manifest=manifest,
        campaign_results=results,
        rounds=round_count,
    )
    out_path = cfg.config_manifest_path.parent / "stage_b_sentinel_execution_manifest.json"
    write_json(out_path, execution_manifest)
    return TfbsStageBExecutionResult(
        status="PASS",
        execution_manifest_path=out_path,
        campaign_count=len(results),
        round_count=round_count,
    )
