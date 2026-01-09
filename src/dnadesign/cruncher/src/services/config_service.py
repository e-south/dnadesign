"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/services/config_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

from dnadesign.cruncher.config.schema_v2 import CruncherConfig


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
    summary: Dict[str, Any] = {
        "out_dir": str(cfg.out_dir),
        "regulator_sets": cfg.regulator_sets,
        "regulators_flat": regulators,
        "regulator_categories": cfg.regulator_categories,
        "campaigns": [campaign.model_dump(mode="json") for campaign in cfg.campaigns],
        "campaign_names": campaign_names,
        "campaign": cfg.campaign.model_dump(mode="json") if cfg.campaign else None,
        "io": {"parsers": {"extra_modules": cfg.io.parsers.extra_modules}},
        "motif_store": {
            "catalog_root": str(cfg.motif_store.catalog_root),
            "pwm_source": cfg.motif_store.pwm_source,
            "site_kinds": cfg.motif_store.site_kinds,
            "combine_sites": cfg.motif_store.combine_sites,
            "dataset_preference": cfg.motif_store.dataset_preference,
            "dataset_map": cfg.motif_store.dataset_map,
            "site_window_lengths": cfg.motif_store.site_window_lengths,
            "site_window_center": cfg.motif_store.site_window_center,
            "min_sites_for_pwm": cfg.motif_store.min_sites_for_pwm,
            "allow_low_sites": cfg.motif_store.allow_low_sites,
            "source_preference": cfg.motif_store.source_preference,
            "allow_ambiguous": cfg.motif_store.allow_ambiguous,
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
                "allow_low_sites": cfg.ingest.regulondb.allow_low_sites,
                "curated_sites": cfg.ingest.regulondb.curated_sites,
                "ht_sites": cfg.ingest.regulondb.ht_sites,
                "ht_dataset_type": cfg.ingest.regulondb.ht_dataset_type,
                "ht_binding_mode": cfg.ingest.regulondb.ht_binding_mode,
            },
            "local_sources": local_sources,
        },
    }
    if cfg.sample is None:
        summary["sample"] = None
    else:
        sample = cfg.sample
        summary["sample"] = {
            "init": {
                "kind": sample.init.kind,
                "length": sample.init.length,
                "regulator": sample.init.regulator,
                "pad_with": sample.init.pad_with,
            },
            "draws": sample.draws,
            "tune": sample.tune,
            "chains": sample.chains,
            "top_k": sample.top_k,
            "min_dist": sample.min_dist,
            "bidirectional": sample.bidirectional,
            "seed": sample.seed,
            "record_tune": sample.record_tune,
            "progress_bar": sample.progress_bar,
            "progress_every": sample.progress_every,
            "save_trace": sample.save_trace,
            "save_sequences": sample.save_sequences,
            "pwm_sum_threshold": sample.pwm_sum_threshold,
            "include_consensus_in_elites": sample.include_consensus_in_elites,
            "optimiser": {
                "kind": sample.optimiser.kind,
                "scorer_scale": sample.optimiser.scorer_scale,
                "cooling": sample.optimiser.cooling.model_dump(),
                "swap_prob": sample.optimiser.swap_prob,
            },
            "moves": {
                "block_len_range": sample.moves.block_len_range,
                "multi_k_range": sample.moves.multi_k_range,
                "slide_max_shift": sample.moves.slide_max_shift,
                "swap_len_range": sample.moves.swap_len_range,
                "move_probs": sample.moves.move_probs,
            },
        }
    if cfg.analysis is None:
        summary["analysis"] = None
    else:
        summary["analysis"] = {
            "runs": cfg.analysis.runs,
            "plots": cfg.analysis.plots.model_dump(),
            "scatter_scale": cfg.analysis.scatter_scale,
            "subsampling_epsilon": cfg.analysis.subsampling_epsilon,
            "scatter_style": cfg.analysis.scatter_style,
            "tf_pair": cfg.analysis.tf_pair,
            "archive": cfg.analysis.archive,
        }
    return summary
