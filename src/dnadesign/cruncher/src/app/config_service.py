"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/config_service.py

Load and summarize resolved Cruncher configurations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from dnadesign.cruncher.config.schema_v3 import CruncherConfig


def summarize_config(cfg: CruncherConfig) -> Dict[str, Any]:
    regulators = [tf for group in cfg.regulator_sets for tf in group]
    campaign_names = [campaign.name for campaign in cfg.campaigns]
    local_sources = []
    for src in cfg.ingest.local_sources:
        local_sources.append(
            {
                "source_id": src.source_id,
                "description": src.description,
                "root": str(src.root),
                "patterns": src.patterns,
                "recursive": src.recursive,
                "format_map": src.format_map,
                "default_format": src.default_format,
                "tf_name_strategy": src.tf_name_strategy,
                "matrix_semantics": src.matrix_semantics,
                "extract_sites": src.extract_sites,
                "meme_motif_selector": src.meme_motif_selector,
                "tags": src.tags,
                "citation": src.citation,
                "license": src.license,
                "source_url": src.source_url,
                "source_version": src.source_version,
                "organism": src.organism.model_dump() if src.organism else None,
            }
        )
    site_sources = []
    for src in cfg.ingest.site_sources:
        site_sources.append(
            {
                "source_id": src.source_id,
                "description": src.description,
                "path": str(src.path),
                "tf_name": src.tf_name,
                "record_kind": src.record_kind,
                "tags": src.tags,
                "citation": src.citation,
                "license": src.license,
                "source_url": src.source_url,
                "source_version": src.source_version,
                "organism": src.organism.model_dump() if src.organism else None,
            }
        )
    summary: Dict[str, Any] = {
        "out_dir": str(cfg.out_dir),
        "regulator_sets": cfg.regulator_sets,
        "regulators_flat": regulators,
        "regulator_categories": cfg.regulator_categories,
        "campaigns": [campaign.model_dump(mode="json") for campaign in cfg.campaigns],
        "campaign_names": campaign_names,
        "campaign": cfg.campaign.model_dump(mode="json") if cfg.campaign else None,
        "io": {"parsers": {"extra_modules": cfg.io.parsers.extra_modules}},
        "catalog": {
            "root": str(cfg.catalog.root),
            "pwm_source": cfg.catalog.pwm_source,
            "site_kinds": cfg.catalog.site_kinds,
            "combine_sites": cfg.catalog.combine_sites,
            "pseudocounts": cfg.catalog.pseudocounts,
            "dataset_preference": cfg.catalog.dataset_preference,
            "dataset_map": cfg.catalog.dataset_map,
            "site_window_lengths": cfg.catalog.site_window_lengths,
            "site_window_center": cfg.catalog.site_window_center,
            "pwm_window_lengths": cfg.catalog.pwm_window_lengths,
            "pwm_window_strategy": cfg.catalog.pwm_window_strategy,
            "min_sites_for_pwm": cfg.catalog.min_sites_for_pwm,
            "allow_low_sites": cfg.catalog.allow_low_sites,
            "source_preference": cfg.catalog.source_preference,
            "allow_ambiguous": cfg.catalog.allow_ambiguous,
        },
        "discover": {
            "enabled": cfg.discover.enabled,
            "tool": cfg.discover.tool,
            "tool_path": str(cfg.discover.tool_path) if cfg.discover.tool_path else None,
            "minw": cfg.discover.minw,
            "maxw": cfg.discover.maxw,
            "nmotifs": cfg.discover.nmotifs,
            "min_sequences_for_streme": cfg.discover.min_sequences_for_streme,
            "source_id": cfg.discover.source_id,
        },
        "ingest": {
            "genome_source": cfg.ingest.genome_source,
            "genome_fasta": str(cfg.ingest.genome_fasta) if cfg.ingest.genome_fasta else None,
            "genome_cache": str(cfg.ingest.genome_cache),
            "genome_assembly": cfg.ingest.genome_assembly,
            "contig_aliases": cfg.ingest.contig_aliases,
            "ncbi_email": cfg.ingest.ncbi_email,
            "ncbi_tool": cfg.ingest.ncbi_tool,
            "ncbi_timeout_seconds": cfg.ingest.ncbi_timeout_seconds,
            "http": {
                "retries": cfg.ingest.http.retries,
                "backoff_seconds": cfg.ingest.http.backoff_seconds,
                "max_backoff_seconds": cfg.ingest.http.max_backoff_seconds,
                "retry_statuses": cfg.ingest.http.retry_statuses,
                "respect_retry_after": cfg.ingest.http.respect_retry_after,
            },
            "regulondb": {
                "base_url": cfg.ingest.regulondb.base_url,
                "motif_matrix_source": cfg.ingest.regulondb.motif_matrix_source,
                "alignment_matrix_semantics": cfg.ingest.regulondb.alignment_matrix_semantics,
                "min_sites_for_pwm": cfg.ingest.regulondb.min_sites_for_pwm,
                "pseudocounts": cfg.ingest.regulondb.pseudocounts,
                "allow_low_sites": cfg.ingest.regulondb.allow_low_sites,
                "curated_sites": cfg.ingest.regulondb.curated_sites,
                "ht_sites": cfg.ingest.regulondb.ht_sites,
                "ht_dataset_type": cfg.ingest.regulondb.ht_dataset_type,
                "ht_binding_mode": cfg.ingest.regulondb.ht_binding_mode,
            },
            "local_sources": local_sources,
            "site_sources": site_sources,
        },
    }
    summary["sample"] = cfg.sample.model_dump(mode="json") if cfg.sample else None
    summary["analysis"] = cfg.analysis.model_dump(mode="json") if cfg.analysis else None
    return summary
