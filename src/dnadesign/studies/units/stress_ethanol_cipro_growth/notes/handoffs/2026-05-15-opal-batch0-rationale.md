# OPAL Batch 0 Handoff Rationale

Batch 0 is a pre-assay seed set. It is not OPAL model-based selection because
no measured labels exist yet.

The study owns this first sampling rule because it combines three study-local
surfaces: DenseGen plan/regulator metadata, the LatentDNA representation choice
(`intermediate_embedding_7b_context_anchor_mean_bidir_concat`), and the OPAL
campaign split across ethanol factor, ciprofloxacin factor, and AND objectives.

After batch 0 labels are measured, OPAL owns ingest, label history, training,
scoring, active-learning selection, and ledgers. The intended handoff is:

```bash
uv run opal ingest-y -c <campaign>/configs/campaign.yaml --observed-round 0 --in <labels.xlsx> --apply
uv run opal run -c <campaign>/configs/campaign.yaml --labels-as-of 0
```

The candidate table is `usr_prom_eth_cip_opal_candidates`, an OPAL candidate
feature table with one fixed-length vector-valued X column. It should not be
called just a matrix. The selected X is the Fwd+RC 1 kb context-anchor Evo2 7B
intermediate embedding vector, not a UMAP coordinate, centroid-distance scalar,
assay label, or phenotype claim.

Terminology boundaries:

- DenseGen plans: `background_only`, `ethanol`, `ciprofloxacin`, and
  `ethanol_ciprofloxacin`
- OPAL campaigns: ethanol factor, ciprofloxacin factor, and AND
- SFXI state order: `[00, 10, 01, 11]`
- Current study phase: `latentdna_reference_normalization_audit`

Use `ciprofloxacin` in data values; reserve `cipro` for display abbreviations
and compact campaign slugs. Treat AND as a campaign objective, not as a synonym
for every dual-plan row. Preserve the Sigma-35 ladder semantics
`f > e > d > c > b`; plot labels may appear uppercase.
