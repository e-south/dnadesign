# Candidate X Selection Scorecard

This deliverable narrows Stress LatentDNA review to three biologically motivated
Evo2 7B intermediate candidates:

- **Anchor-source causal mean:** Evo2 is run on the emitted source insert, and
  Infer averages the token-position hidden states across that emitted insert.
  This is the conservative DenseGen-plan baseline.
- **Forward 1 kb context anchor-span causal mean:** Evo2 is run on the full
  forward 1 kb expression-vector context, and Infer averages only the
  Construct-provided anchor span. This keeps vector context available to later
  anchor positions while avoiding whole-context dilution.
- **Equal-block forward/RC 1 kb context anchor-span concat:** Evo2 is run on the
  forward 1 kb context and on the matched reverse-complement 1 kb context.
  LatentDNA normalizes each `anchor_mean` block and concatenates them. This is a
  bidirectional summary in the row-level sense, not a native bidirectional Evo2
  hidden state.

Rank is treated as a collapse gate, not the selector. The selection surface
balances DenseGen plan geometry, Sigma-35 ordinal structure, reference-to-plan
centroid margins, and collection-specific Anderson/W strength audits.

Claims stay pre-assay: these plots support representation triage for a working
`X`; they do not estimate promoter activity or growth phenotype.

The word "bidirectional" is used as study shorthand for an external
two-orientation summary. Evo2 still supplies causal token states in each emitted
sequence. The bidirectional candidate concatenates equal-weight forward and
reverse-complement 1 kb context `anchor_mean` blocks; it is not a native
bidirectional Evo2 hidden state.

### candidate_x_selection_scorecard | Candidate X selection scorecard

#### Plot details

**Data.** This plot compares the three predeclared candidate-X views:
anchor-source causal mean, forward 1 kb context anchor-span causal mean, and
equal-block forward/RC 1 kb context anchor-span causal-mean concat. All three
are row-level summaries built before assay labels exist.

**Decision use.** Treat effective rank as a health gate. The decision signal comes from the combination of DenseGen plan separation, Sigma-35 ordinal structure, reference-to-plan centroid margin, and collection-specific reference-standard strength ordering.

**Limits.** A strong scorecard row does not prove activity or growth. It only says the representation preserves the intended pre-assay latent geometry better under the declared checks.

### reference_to_plan_centroid_heatmap | Reference-to-plan centroid heatmap

#### Plot details

**Data.** Reference rows or reference-set centroids are compared with background, ethanol, ciprofloxacin, and dual-stress DenseGen plan centroids in high-dimensional cosine space.

**Decision use.** Read the nearest-plan margin as the primary evidence. Native-length and core60 references remain separate sequence-scope contracts, so a core60 improvement supports the manifold-comparability hypothesis only within that paired reference logic.

**Limits.** Diffuse margins mean the references are weak landmarks under the tested representation; they are not a phenotype failure.

### reference_standard_strength_audit | Reference standard strength audit

#### Plot details

**Data.** Anderson iGEM and W collection standards are audited within their own collection-specific numeric strength scales.

**Decision use.** Use the high-dimensional Spearman/Kendall statistics as the evidence for latent ordering. Projection views are visual support only.

**Limits.** Anderson and W strength values should not be pooled into one biological scale. Weak ordering in one collection does not invalidate DenseGen plan separation or the Sigma-35 ladder.
