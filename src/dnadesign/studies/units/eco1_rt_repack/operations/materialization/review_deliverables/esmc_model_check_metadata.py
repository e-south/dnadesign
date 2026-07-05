"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/esmc_model_check_metadata.py

Manifest metadata for WT ESMC masked-marginal model-check panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

SECTION = SECTION_CONSTRAINT_EVIDENCE
SOURCE_TABLES = [
    "biohub_esmc/mutation_scoring/wt_mutation_scoring_mask_join.parquet",
    "biohub_esmc/mutation_scoring/wt_position_entropy.parquet",
    "biohub_esmc/mutation_scoring/wt_substitution_llr.parquet",
]
INTERPRETATION_LIMIT = (
    "ESMC masked-marginal scores are a model check of the WT sequence context. "
    "They are not experimental DMS, processivity evidence, or a current mask update."
)
METHOD_SUMMARY = (
    "Mask one WT residue, read ESMC sequence logits at that masked position, compute entropy in bits, "
    "and compute each single-substitution score as LLR = log P(alternate) - log P(WT)."
)


def missing_model_check_row(panel_root: Path, mutation_scoring_root: Path) -> dict[str, object]:
    """Return the manifest row used when WT mutation-scoring inputs are absent."""

    return make_deliverable_row(
        deliverable_id="wt_model_check",
        section=SECTION,
        artifact_kind="manifest",
        status="skipped_missing_input",
        path=panel_root / "missing_wt_model_check.txt",
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes({"mutation_scoring_root": mutation_scoring_root}),
        alt_text="WT ESMC model check was skipped because mutation-scoring inputs were missing.",
        description="The WT ESMC masked-marginal review section requires the joined mask-context table.",
        interpretation_limit=INTERPRETATION_LIMIT,
        title="WT ESMC masked-marginal checks wait for mutation-scoring inputs",
        method_summary=METHOD_SUMMARY,
        role="review_only",
        skip_reason=f"Missing input table: {mutation_scoring_root / 'wt_mutation_scoring_mask_join.parquet'}",
    )


def mutation_scoring_evidence_summary(mutation_scoring_root: Path) -> dict[str, object]:
    """Summarize row counts and source method for linked WT mutation-scoring artifacts."""

    paths = {
        "position_entropy_rows": mutation_scoring_root / "wt_position_entropy.parquet",
        "substitution_llr_rows": mutation_scoring_root / "wt_substitution_llr.parquet",
        "mask_join_rows": mutation_scoring_root / "wt_mutation_scoring_mask_join.parquet",
    }
    summary: dict[str, object] = {
        "scoring_method": "ESMC masked-marginal sequence-logit scoring",
        "source_notebook": (
            "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
            "esmc_mutation_scoring.ipynb"
        ),
    }
    for label, path in paths.items():
        if path.exists():
            summary[label] = int(pq.read_metadata(path).num_rows)
    return summary
