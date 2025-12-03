"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/tests/test_multisite_protocol.py

Integration tests for the multisite_select protocol:

  • end-to-end run on a tiny synthetic dataset
  • invariants on epistasis, scores, diversity, and cluster caps
  • tie-breaker behaviour (prefer fewer mutations, then higher delta, etc.)

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.permuter.src.core.storage import read_parquet
from dnadesign.permuter.src.protocols.multisite_select.geometry import (
    l2_normalize_rows,
    pairwise_angular,
)
from dnadesign.permuter.src.protocols.multisite_select.main import MSel
from dnadesign.permuter.src.protocols.multisite_select.utils import (
    parse_aa_combo_to_map,
)


def _write_synthetic_records(tmp_path: Path) -> Path:
    """
    Create a tiny synthetic records.parquet suitable for testing multisite_select.

    We keep embeddings 2D to keep things simple; extract_embedding_matrix does
    not assume 512-dim explicitly.

    The synthetic dataset has:
      • three clusters: A, B, C (two variants each),
      • strictly non-negative epistasis,
      • consistent AA position lists and combo strings,
      • DNA sequences that are proper coding regions (length % 3 == 0).
    """
    records = []

    # Three clusters: A, B, C; two variants each
    clusters = ["A", "A", "B", "B", "C", "C"]
    mut_counts = [2, 3, 2, 3, 2, 3]

    # Observed LLR (higher for B, then C, then A)
    llr_obs = [0.5, 0.2, 3.0, 2.5, 1.5, 1.0]
    llr_exp = [0.0] * 6  # simple baseline
    epistasis = [obs - exp for obs, exp in zip(llr_obs, llr_exp)]

    # Simple embeddings on a circle
    angles = np.deg2rad([0.0, 10.0, 120.0, 130.0, 240.0, 250.0])
    emb = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # AA positions (1-indexed, within a small coding region)
    aa_pos_lists = [
        [1, 2],
        [1, 3, 4],
        [5, 6],
        [5, 7, 8],
        [9, 10],
        [9, 11, 12],
    ]
    aa_combo_strs = [
        "A1V|B2C",
        "A1V|C3K|D4E",
        "E5K|F6R",
        "E5K|G7A|H8L",
        "I9M|J10N",
        "I9M|K11Q|L12P",
    ]

    # DNA sequence: same for all variants; length divisible by 3
    #  (12 codons → AA positions 1..12 are valid)
    sequence = "acg" * 12  # length 36

    for i in range(6):
        records.append(
            {
                "id": f"var{i}",
                "sequence": sequence,
                "permuter__observed__llr_mean": llr_obs[i],
                "permuter__expected__llr_mean": llr_exp[i],
                "epistasis": epistasis[i],
                "permuter__observed__logits_mean": emb[i].tolist(),
                "permuter__aa_pos_list": aa_pos_lists[i],
                "permuter__aa_combo_str": aa_combo_strs[i],
                "permuter__mut_count": mut_counts[i],
                "cluster__perm_v1": clusters[i],
                # optional columns used for tie-breakers
                "permuter__proposal_score": float(i),
                "permuter__var_id": f"v{i}",
            }
        )

    df = pd.DataFrame(records)
    out_path = tmp_path / "records.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def _build_default_params(records_path: Path, art_dir: Path) -> dict:
    """
    Default YAML-equivalent parameters for an end-to-end multisite_select run,
    matching the textual spec for the RT selection job.
    """
    return {
        "from_dataset": str(records_path),
        "_artifact_dir": str(art_dir),
        "select": {
            "scoring": {
                "normalize": {
                    "method": "mad",
                    "gaussian_consistent": False,
                    "winsor_mads": None,
                },
                "weights": {"llr": 1.0, "epi": 1.0},
            },
            "embedding": {
                "column": "permuter__observed__logits_mean",
                "l2_normalize": True,
                "distance": "angular",
                "representative": "medoid",
            },
            "clusters": {
                "picks_per_cluster": 1,
                "filters": {
                    # relaxed thresholds so all clusters pass without relaxation
                    "min_cluster_mean_z_llr": -10.0,
                    "min_cluster_pos_epistasis_fraction": 0.0,
                    "location_stat": "mean",
                    "trimmed_mean_frac": 0.10,
                },
            },
            "budget": {
                "total_variants": 4,
                "pool_factor": 2.5,
                "intracluster_diversity": {
                    "enabled": True,
                    # modest angular threshold so the synthetic embeddings satisfy it
                    "min_angular_distance_deg": 8.0,
                },
            },
            "tie_breakers": {
                "prefer_fewer_mutations": True,
                "then_higher_delta": True,
                "then_higher_proposal_score": True,
            },
            "diagnostics": {
                "figsize_in": 4,
                "dpi": 100,
                "random_sample_seed": 123,
                "random_sample_repeats": 1,
                "random_sample_factor": 1.0,
                "random_sample_cap": 64,
            },
            "heb": {
                "enabled": True,
                "min_cooccur_count": 1,
                "width_scale": "sqrt",
                "edge_cmap": "viridis",
                "color_by": "node_avg_k",
                "out_svg": False,
                "figsize_in": 6,
            },
            "reproducibility": {
                "rng_seed": 12345,
            },
        },
    }


def test_multisite_select_end_to_end_invariants(tmp_path):
    records_path = _write_synthetic_records(tmp_path)
    art_dir = tmp_path / "artifacts"
    params = _build_default_params(records_path, art_dir)

    proto = MSel()
    proto.validate_cfg(params=params)

    # Run selection
    picks = list(proto.generate(ref_entry={"name": "REF"}, params=params))

    # Basic count invariant
    assert 1 <= len(picks) <= params["select"]["budget"]["total_variants"]

    # Core artifacts are created
    select_parquet = art_dir / "MULTISITE_SELECT.parquet"
    select_csv = art_dir / "MULTISITE_SELECT.csv"
    cluster_summary = art_dir / "CLUSTER_SUMMARY.parquet"
    summary_md = art_dir / "SELECT_SUMMARY.md"

    assert select_parquet.exists()
    assert select_csv.exists()
    assert cluster_summary.exists()
    assert summary_md.exists()

    sel_df = read_parquet(select_parquet)
    assert len(sel_df) == len(picks)

    # ------------------------------------------------------------------
    # Epistasis invariants
    # ------------------------------------------------------------------
    # delta = llr_obs - llr_exp, all non-negative, and equal to source epistasis
    diff = sel_df["llr_obs"] - sel_df["llr_exp"]
    np.testing.assert_allclose(diff.to_numpy(), sel_df["delta"].to_numpy())
    assert (sel_df["delta"] >= 0.0).all()

    src_df = read_parquet(records_path)
    assert "epistasis" in src_df.columns

    src_epi = src_df.set_index("id")["epistasis"]
    epi_selected_source = src_epi.loc[sel_df["source_id"]].to_numpy(dtype=float)
    assert (epi_selected_source >= 0.0).all()
    np.testing.assert_allclose(
        epi_selected_source, sel_df["delta"].to_numpy(dtype=float)
    )

    # ------------------------------------------------------------------
    # Score invariants
    # ------------------------------------------------------------------
    weights = params["select"]["scoring"]["weights"]
    expected_score = sel_df["z_llr"] * float(weights["llr"]) + sel_df["z_epi"] * float(
        weights["epi"]
    )
    np.testing.assert_allclose(expected_score.to_numpy(), sel_df["score"].to_numpy())

    # ------------------------------------------------------------------
    # Cluster representation and caps
    # ------------------------------------------------------------------
    src_clusters = set(src_df["cluster__perm_v1"])
    assert set(sel_df["cluster_id"]).issubset(src_clusters)

    # picks_per_cluster=1, so no cluster should contribute more than one pick
    max_per_cluster = sel_df["cluster_id"].value_counts().max()
    assert max_per_cluster <= params["select"]["clusters"]["picks_per_cluster"]

    # We should hit at least min(3, len(sel_df)) distinct clusters
    assert len(sel_df["cluster_id"].unique()) >= min(3, len(sel_df))

    # ------------------------------------------------------------------
    # Mutation combination uniqueness and consistency
    # ------------------------------------------------------------------
    assert "aa_combo_str" in sel_df.columns
    aa_combos = sel_df["aa_combo_str"].astype(str)
    # No duplicate multi-mutant combinations in the final selection artifacts
    assert aa_combos.is_unique

    aa_pos_tuples = []
    for _, row in sel_df.iterrows():
        combo_map = parse_aa_combo_to_map(row["aa_combo_str"])
        pos_from_combo = sorted(combo_map.keys())
        pos_from_list = sorted(int(p) for p in row["aa_pos_list"])
        assert pos_from_combo == pos_from_list
        aa_pos_tuples.append(tuple(pos_from_list))

    # Enforce uniqueness of AA position lists as well.
    assert len(aa_pos_tuples) == len(set(aa_pos_tuples))

    # The emitted variant stream should mirror artifact-level uniqueness.
    aa_combo_stream = [str(p["aa_combo_str"]) for p in picks]
    aa_pos_stream = [tuple(int(p) for p in p["aa_pos_list"]) for p in picks]
    assert len(aa_combo_stream) == len(set(aa_combo_stream))
    assert len(aa_pos_stream) == len(set(aa_pos_stream))

    # ------------------------------------------------------------------
    # Decorated DNA sequence consistency (uppercase mutated codons)
    # ------------------------------------------------------------------
    for _, row in sel_df.iterrows():
        seq = row["sequence"]
        aa_positions = list(row["aa_pos_list"])
        L = len(seq)
        assert L % 3 == 0
        n_codons = L // 3

        # All AA positions must map into the sequence
        assert all(1 <= int(p) <= n_codons for p in aa_positions)

        mutated_codons = set(int(p) for p in aa_positions)
        for codon_idx in range(1, n_codons + 1):
            start = 3 * (codon_idx - 1)
            codon = seq[start : start + 3]
            if codon_idx in mutated_codons:
                # At least one base in the codon should be uppercase
                assert any(ch.isupper() for ch in codon)
            else:
                # Unmutated codons should be entirely lowercase
                assert codon.lower() == codon

    # ------------------------------------------------------------------
    # sequence_window: per-variant trimming of the global mutation window
    # ------------------------------------------------------------------
    assert "sequence_window" in sel_df.columns

    # Derive the global AA window directly from selected variants.
    all_positions = []
    for aa_list in sel_df["aa_pos_list"]:
        all_positions.extend(int(p) for p in aa_list)
    assert all_positions, "Selected variants must carry at least one mutation"
    start_aa = min(all_positions)
    end_aa = max(all_positions)
    expected_nt_len = 3 * (end_aa - start_aa + 1)
    nt_start_idx = 3 * (start_aa - 1)
    nt_end_idx = 3 * end_aa

    window_lengths = sel_df["sequence_window"].astype(str).str.len().unique().tolist()
    assert len(window_lengths) == 1
    assert int(window_lengths[0]) == expected_nt_len

    for _, row in sel_df.iterrows():
        full = str(row["sequence"])
        window = str(row["sequence_window"])
        # Window must be a contiguous slice in nucleotide space, aligned
        # to codon boundaries and shared across all selected variants.
        assert full[nt_start_idx:nt_end_idx] == window

    # ------------------------------------------------------------------
    # Diversity invariants when intracluster_diversity is enabled
    # ------------------------------------------------------------------
    budget_cfg = params["select"]["budget"]
    div_cfg = budget_cfg["intracluster_diversity"]
    if div_cfg.get("enabled", False) and len(sel_df) >= 2:
        min_angle_deg = float(div_cfg["min_angular_distance_deg"])

        # Reconstruct embeddings for selected variants in source order
        src_by_id = src_df.set_index("id")
        emb_list = (
            src_by_id.loc[sel_df["source_id"]]["permuter__observed__logits_mean"]
            .apply(lambda v: np.asarray(v, dtype=float))
            .to_list()
        )
        emb_sel = np.stack(emb_list, axis=0)
        U_sel = l2_normalize_rows(emb_sel)
        A = pairwise_angular(U_sel)

        # Minimum off-diagonal angle in degrees should respect the threshold
        if A.shape[0] > 1:
            mask = ~np.eye(A.shape[0], dtype=bool)
            min_offdiag_deg = float(np.rad2deg(A[mask]).min())
            assert min_offdiag_deg + 1e-6 >= min_angle_deg


def test_tie_breakers_respected(tmp_path):
    """
    Tie-breakers should obey the documented order:

      1) composite score (here: llr only),
      2) mut_count (fewer mutations first),
      3) delta (higher epistasis),
      4) proposal_score (higher is better),
      5) var_id (lexicographic).

    This test constructs two variants with identical composite score but different
    mut_count, delta, and proposal_score, and checks that the intended ordering
    is honoured.
    """
    records_path = _write_synthetic_records(tmp_path)
    art_dir = tmp_path / "artifacts"

    # Construct a minimal dataset that actually exercises tie-breakers:
    #
    #   • keep only the two A-cluster variants (var0, var1),
    #   • force them to have identical composite scores (same llr),
    #   • keep their differing mut_count / delta / proposal_score.
    #
    # With pool_factor=1.0 and total_variants=1, the score-gated pool
    # contains both rows, and their *relative order* is determined purely
    # by the configured tie-breakers.
    df_full = read_parquet(records_path)
    df = df_full[df_full["cluster__perm_v1"] == "A"].copy()
    # Sanity: we expect exactly two rows here (var0, var1).
    assert df.shape[0] == 2

    # Force identical llr so composite score is tied; we do not touch
    # epistasis or proposal_score so that differences in delta and proposal
    # remain available for downstream tie-breakers.
    df["permuter__observed__llr_mean"] = 1.0

    df.to_parquet(records_path, index=False)

    params = {
        "from_dataset": str(records_path),
        "_artifact_dir": str(art_dir),
        "select": {
            "scoring": {
                "normalize": {
                    "method": "none",  # raw scores for simpler reasoning
                    "gaussian_consistent": False,
                    "winsor_mads": None,
                },
                # Score is purely llr_obs
                "weights": {"llr": 1.0, "epi": 0.0},
            },
            "embedding": {
                "column": "permuter__observed__logits_mean",
                "l2_normalize": True,
                "distance": "angular",
                "representative": "medoid",
            },
            "clusters": {
                "picks_per_cluster": 1,
                "filters": {
                    "min_cluster_mean_z_llr": None,
                    "min_cluster_pos_epistasis_fraction": None,
                    "location_stat": "mean",
                    "trimmed_mean_frac": 0.10,
                },
            },
            "budget": {
                "total_variants": 1,
                "pool_factor": 1.0,
                "intracluster_diversity": {
                    "enabled": False,
                    "min_angular_distance_deg": 0.0,
                },
            },
            "tie_breakers": {
                "prefer_fewer_mutations": True,
                "then_higher_delta": True,
                "then_higher_proposal_score": True,
            },
            "diagnostics": {
                "figsize_in": 4,
                "dpi": 100,
                "random_sample_seed": 123,
                "random_sample_repeats": 1,
            },
            "heb": {
                "enabled": False,
            },
            "reproducibility": {"rng_seed": 42},
        },
    }

    proto = MSel()
    proto.validate_cfg(params=params)
    picks = list(proto.generate(ref_entry={"name": "REF"}, params=params))

    assert len(picks) == 1

    # With identical scores, prefer fewer mutations (k), then higher delta, then proposal_score.
    # In this reduced synthetic dataset, var0 has mut_count=2 and var1 has mut_count=3.
    assert picks[0]["mut_count"] == 2
    assert picks[0]["source_id"] == "var0"
