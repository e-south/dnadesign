"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/foldcheck_fixtures.py

Fold-review fixtures for Eco1 review-deliverable materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .foldcheck_structure_set_fixtures import write_foldcheck_full_structure_set

_FOLDCHECK_RANKING_SOURCE = ["foldcheck_review/foldcheck_candidate_ranking.parquet"]
_STRUCTURE_PANEL_SOURCE = [
    "foldcheck_review/foldcheck_structure_panel.yaml",
    "foldcheck_review/foldcheck_candidate_ranking.parquet",
]
_REVIEW_CLASS_COUNTS_TITLE = "Fold bins summarize structural triage"


def write_foldcheck_review_manifest(review_root: Path) -> None:
    """Write a compact fold-review manifest and ranking table."""

    plot_root = review_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    _write_foldcheck_candidate_ranking(review_root / "foldcheck_candidate_ranking.parquet")
    write_foldcheck_full_structure_set(review_root)
    plot_root.joinpath("review_class_counts.svg").write_text(
        (
            f'<svg role="img"><title>{_REVIEW_CLASS_COUNTS_TITLE}</title>'
            "<desc>Fixture review-class counts.</desc></svg>\n"
        ),
        encoding="utf-8",
    )
    plot_root.joinpath("fold_metric_scatter.svg").write_text(
        '<svg role="img"><title>Fold metrics</title><desc>Fixture fold metrics.</desc></svg>\n',
        encoding="utf-8",
    )
    plot_root.joinpath("biohub_esmc_sae_coverage.svg").write_text(
        '<svg role="img"><title>SAE coverage</title><desc>Fixture SAE coverage.</desc></svg>\n',
        encoding="utf-8",
    )
    plot_root.joinpath("structure_overlay_panel.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    plots = [
        _plot_row(
            plot_id="review_class_counts",
            path="plots/review_class_counts.svg",
            title=_REVIEW_CLASS_COUNTS_TITLE,
            alt_text="Fixture review-class count plot.",
            description="Fixture fold-review class-count plot.",
            interpretation_limit="Review labels are triage summaries, not candidate acceptance decisions.",
            data_sources=_FOLDCHECK_RANKING_SOURCE,
        ),
        _plot_row(
            plot_id="fold_metric_scatter",
            path="plots/fold_metric_scatter.svg",
            title="ColabFold fold-check metric scatter",
            alt_text="Fixture fold metric scatter.",
            description="Fixture fold-review plot.",
            interpretation_limit="Fold metrics do not measure activity.",
            data_sources=_FOLDCHECK_RANKING_SOURCE,
        ),
        _plot_row(
            plot_id="biohub_esmc_sae_coverage",
            path="plots/biohub_esmc_sae_coverage.svg",
            title="Biohub ESMC SAE coverage is complete for fold-reviewed sequences",
            alt_text="Fixture Biohub ESMC SAE coverage plot.",
            description="Fixture Biohub ESMC SAE coverage plot.",
            interpretation_limit="SAE coverage is annotation coverage, not activity evidence.",
            data_sources=["biohub_esmc_sae_profile.parquet"],
        ),
        _plot_row(
            plot_id="structure_overlay_panel",
            path="plots/structure_overlay_panel.png",
            title="Selected ColabFold structures align to the cryoEM reference",
            alt_text="Fixture ChimeraX overlay panel.",
            description="Fixture fold-review ChimeraX render.",
            interpretation_limit="Structure overlays are visual review aids.",
            data_sources=_STRUCTURE_PANEL_SOURCE,
        ),
        _plot_row(
            plot_id="structure_overlay_skipped",
            status="skipped_runtime_unavailable",
            path="plots/structure_overlay_skipped.png",
            title="Skipped structure overlay fixture",
            alt_text="Skipped overlay fixture.",
            description="Fixture skipped fold-review ChimeraX render.",
            interpretation_limit="Skipped renders do not support review.",
            data_sources=["foldcheck_review/foldcheck_structure_panel.yaml"],
            skip_reason="ChimeraX unavailable in fixture.",
        ),
    ]
    review_root.joinpath("review_visual_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.foldcheck_review_visual_manifest",
                "status": "materialized",
                "plot_count": len(plots),
                "plots": plots,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _plot_row(
    *,
    plot_id: str,
    path: str,
    title: str,
    alt_text: str,
    description: str,
    interpretation_limit: str,
    data_sources: list[str],
    status: str = "rendered",
    skip_reason: str = "",
) -> dict[str, object]:
    return {
        "plot_id": plot_id,
        "status": status,
        "path": path,
        "title": title,
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "data_sources": data_sources,
        "skip_reason": skip_reason,
    }


def _write_foldcheck_candidate_ranking(path: Path) -> None:
    rows = [
        {
            "candidate_id": "thread_candidate_alpha",
            "review_rank": 1,
            "plddt": 92.4,
            "wt_runtime_ca_rmsd": 0.82,
            "cryoem_mapped_ca_rmsd": 1.23,
            "seq_recovery": 0.72,
            "mutation_count": 2,
            "review_class": "strong_fold_preserved",
        },
        {
            "candidate_id": "thread_candidate_beta",
            "review_rank": 2,
            "plddt": 89.7,
            "wt_runtime_ca_rmsd": 3.12,
            "cryoem_mapped_ca_rmsd": 2.45,
            "seq_recovery": 0.55,
            "mutation_count": 3,
            "review_class": "review_band",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
