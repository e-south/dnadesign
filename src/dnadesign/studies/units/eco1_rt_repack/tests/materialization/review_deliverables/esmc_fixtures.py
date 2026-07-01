"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/esmc_fixtures.py

ESMC fixture helpers for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


def write_wt_mutation_scoring_outputs(
    output_root: Path,
    *,
    scoring_root: Path | None = None,
    model: str = "esmc-300m-2024-12",
    request_hash_tail: str = "6",
    llr_shift: float = 0.0,
) -> None:
    """Write compact WT ESMC masked-marginal fixture outputs."""

    scoring_root = scoring_root or output_root / "biohub_esmc" / "mutation_scoring"
    plot_root = scoring_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for plot_name in (
        "wt_entropy_by_position.svg",
        "wt_fraction_negative_alternate_llr_by_position.svg",
        "wt_substitution_llr_heatmap.svg",
    ):
        plot_root.joinpath(plot_name).write_text(
            f'<svg role="img"><title>{plot_name}</title><desc>Fixture ESMC plot.</desc></svg>\n',
            encoding="utf-8",
        )

    scoring_root.joinpath("wt_mutation_scoring_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt_repack.biohub_esmc_wt_mutation_scoring.request",
                "status": "materialized",
                "biohub_request_hash": "sha256:" + request_hash_tail * 64,
                "source_request_hash": "sha256:" + "7" * 64,
                "biohub_api_base_url": "https://biohub.ai",
                "biohub_api_version": "v1",
                "endpoint_flow": ["POST /api/v1/encode", "POST /api/v1/logits"],
                "model": model,
                "scoring_method_id": "esmc_masked_marginal_v1",
                "position_count": 6,
                "accepted_position_count": 6,
                "errored_position_count": 0,
                "changes_current_mask": False,
                "authorization": "<redacted>",
                "method_references": [
                    {
                        "title": "Biohub ESMC mutation-scoring notebook",
                        "url": (
                            "https://colab.research.google.com/github/Biohub/esm/blob/main/cookbook/tutorials/"
                            "esmc_mutation_scoring.ipynb"
                        ),
                        "role": "masked-marginal entropy and zero-shot LLR method pattern",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(_position_entropy_rows()), scoring_root / "wt_position_entropy.parquet")
    pq.write_table(
        pa.Table.from_pylist(_substitution_llr_rows(llr_shift=llr_shift)),
        scoring_root / "wt_substitution_llr.parquet",
    )
    pq.write_table(pa.Table.from_pylist(_mask_join_rows()), scoring_root / "wt_mutation_scoring_mask_join.parquet")


def _mask_join_rows() -> list[dict[str, object]]:
    entropy_values = [0.2, 1.2, 2.8, 0.7, 3.4, 1.8]
    plurality_values = [0.9, 0.7, 0.15, 0.5, 0.1, 0.3]
    best_alt_values = [-3.5, -1.2, 1.8, -0.5, 2.5, 0.2]
    fraction_negative_values = [1.0, 1.0, 0.6, 0.95, 0.4, 0.85]
    rows: list[dict[str, object]] = []
    for position, wt_aa in enumerate("MKSAYL", start=1):
        rows.append(
            {
                "sequence_id": "wild_type",
                "sequence_hash": "sha256:" + "3" * 64,
                "canonical_position": position,
                "wt_aa": wt_aa,
                "protected": position in {2, 3, 4, 5},
                "non_fixed": position in {1, 6},
                "non_fixed_missing_backbone": False,
                "protection_reasons_json": "[]",
                "motif_protected": position == 3,
                "wang_ec86_direct_contact_prior": position == 4,
                "direct_retained_dna_rna_contact_5a": position == 5,
                "evolutionarily_conserved_clade9_25pct_plurality": position in {1, 2, 4},
                "wt_plurality_frequency": plurality_values[position - 1],
                "min_distance_to_retained_dna_rna_angstrom": float(position),
                "entropy_bits": entropy_values[position - 1],
                "canonical_entropy_bits": entropy_values[position - 1],
                "fraction_negative_alternate_llr": fraction_negative_values[position - 1],
                "best_alt_aa": "A",
                "best_alt_llr": best_alt_values[position - 1],
                "worst_alt_aa": "W",
                "worst_alt_llr": -6.0 - position,
                "status": "accepted",
                "mask_context_status": "joined",
            }
        )
    return rows


def _position_entropy_rows() -> list[dict[str, object]]:
    return [
        {
            "sequence_id": row["sequence_id"],
            "sequence_hash": row["sequence_hash"],
            "canonical_position": row["canonical_position"],
            "wt_aa": row["wt_aa"],
            "status": row["status"],
            "entropy_bits": row["entropy_bits"],
            "fraction_negative_alternate_llr": row["fraction_negative_alternate_llr"],
            "best_alt_aa": row["best_alt_aa"],
            "best_alt_llr": row["best_alt_llr"],
        }
        for row in _mask_join_rows()
    ]


def _substitution_llr_rows(*, llr_shift: float = 0.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position_row in _mask_join_rows():
        wt_aa = str(position_row["wt_aa"])
        for index, alt_aa in enumerate(_CANONICAL_AMINO_ACIDS):
            if alt_aa == wt_aa:
                continue
            rows.append(
                {
                    "sequence_id": position_row["sequence_id"],
                    "sequence_hash": position_row["sequence_hash"],
                    "canonical_position": position_row["canonical_position"],
                    "wt_aa": wt_aa,
                    "alt_aa": alt_aa,
                    "llr": float(index) / 10.0 - 1.0 + llr_shift,
                    "status": position_row["status"],
                }
            )
    return rows
