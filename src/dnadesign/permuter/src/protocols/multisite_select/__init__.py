"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/__init__.py

Multi-site mutant variant selection protocol.

Implements a score-gated, diversity-constrained selection over multi-mutant
variants using:

  • robust median/MAD scaling of observed fitness (LLR) and epistasis,
  • a composite per-variant score (alpha · z_llr + β · z_epi),
  • a high-score candidate pool,
  • greedy score-first, diversity-second selection in Evo-2 logits space,
  • optional per-cluster capacity caps,
  • strict row-level validation and deterministic tie-breaking.

This protocol expects a source dataset (records.parquet) produced upstream by
combine_aa + evaluate, with:

  • observed fitness:    permuter__observed__<metric_id>
  • expected baseline:   permuter__expected__<metric_id>
  • epistasis:           epistasis = observed - expected
  • logits embedding:    permuter__observed__logits_mean (list<item: double>)

Run via:

    permuter run \
      --job jobs/rt_multisite_select.yaml \
      --ref retron_Eco1_RT_wt

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""
