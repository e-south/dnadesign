# Candidate X Selection Scorecard

This deliverable narrows Stress LatentDNA review to three biologically motivated
Evo 2 7B intermediate candidates:

- **Anchor-source causal mean:** Evo 2 is run on the emitted source insert, and
  Infer averages the token-position hidden states across that emitted insert.
  This is the conservative DenseGen-plan baseline.
- **Forward 1 kb context anchor-span causal mean:** Evo 2 is run on the full
  forward 1 kb expression-vector context, and Infer averages only the
  Construct-provided anchor span. This keeps vector context available to later
  anchor positions while avoiding whole-context dilution.
- **Equal-block forward/RC 1 kb context anchor-span concat:** Evo 2 is run on the
  forward 1 kb context and on the matched reverse-complement 1 kb context.
  LatentDNA standardizes and row-L2-normalizes each `anchor_mean` block before
  feature-axis concatenation. This is a bidirectional summary in the row-level
  sense, not a native bidirectional Evo 2 hidden state.

Rank is treated as a collapse gate, not the selector. The selection surface
balances DenseGen plan geometry, Sigma-35 ordinal structure, reference-to-plan
centroid margins, and collection-specific Anderson/W strength audits.

Claims stay pre-assay: these plots support representation triage for a working
`X`; they do not estimate promoter activity or growth phenotype.

The word "bidirectional" is used as study shorthand for an external
two-orientation summary. Evo 2 still supplies causal token states in each emitted
sequence. The bidirectional candidate concatenates separately standardized and
row-L2-normalized forward and reverse-complement 1 kb context `anchor_mean`
blocks; it is not a native bidirectional Evo 2 hidden state.

## Current Synthesis

The current working pre-assay representation is
`intermediate_embedding_7b_context_anchor_mean_bidir_concat`. This is not
because it wins every scalar. It is the least contrived compromise across the
declared questions: it passes the health gate most strongly, lowers PC1
dominance, gives the highest global Sigma-35 ordinal signal, has the strongest
median reference-to-plan margin, and retains near-best DenseGen plan separation.

The conservative baseline remains `intermediate_embedding_7b_anchor_60bp`,
because it gives the strongest DenseGen plan separation. The interpretation
lens for known synthetic strength standards remains
`intermediate_embedding_7b_full_context_anchor_mean`, because the W collection
has its strongest high-dimensional ordinal signal there
(`Spearman = 0.658`, permutation `p = 0.0099`). This is an argument for using
the bidirectional candidate as the working active-learning `X`, not an argument
that every biological question has one universal winner.

| Candidate view | Health read | Main positive signal | Main caveat |
|---|---|---|---|
| Anchor-source causal mean | effective rank `10.99`, PC1 fraction `0.261` | strongest DenseGen plan separation, `1.494` | mixed source includes variable-length native/reference rows |
| Forward 1 kb context anchor-span causal mean | effective rank `7.85`, PC1 fraction `0.395` | strongest W collection strength ordering, `Spearman = 0.658` | weaker DenseGen plan separation than anchor-source or bidirectional |
| Equal-block forward/RC 1 kb context anchor-span concat | effective rank `12.56`, PC1 fraction `0.215` | best global Sigma-35 Spearman, `0.496`, and best median reference margin, `0.095` | not the strongest synthetic-standard strength lens |

The most defensible narrative is therefore:

> Carry the equal-block forward/RC 1 kb context anchor-span concat as the
> working pre-assay `X` for active-learning planning. Keep anchor-source mean
> as the DenseGen-plan baseline, and keep forward context anchor mean as the
> strength-standard interpretation lens.

## Secondary Story Surfaces

The current plots rightly retain the full LatentDNA audit surface. The study
story should be narrower and evidence-led:

- **W collection strength ordering.** The W collection has a clear
  collection-specific high-dimensional ordinal signal in the forward 1 kb
  context anchor-span view. A useful story figure would plot W standard numeric
  strength against the high-dimensional ordinal-axis projection, faceted by
  candidate view, with Spearman/Kendall reported in the panel title. This would
  show the statistic as more than a bar height. Do not pool W and Anderson
  strength values into one scale.
- **spyP/sulAp reference normalization.** The core60 landmarks behave more
  plausibly than native-length landmarks. Across all three candidate-X views,
  `spyp_core60` maps nearest the Ethanol centroid and `sulAp_core60` maps
  nearest the Ciprofloxacin centroid. Native-length `spyP`/`sulAp` are less
  coherent, which supports the core60 normalization rationale. This should be
  shown as a paired native-to-core60 reference-to-plan heatmap or slope plot.
- **Native MG1655 promoters.** The configured Native MG1655 set centroid shifts
  toward the Ethanol centroid after core60 normalization, especially in the
  bidirectional candidate, but individual promoters remain heterogeneous. That
  configured set currently contains 16 native rows; `spyP`/`sulAp` are exposed
  as a separate landmark set and `soxSp` is not included in the configured
  Native MG1655 regex. Current LatentDNA metadata does not yet encode an
  expected stress-response class for each native MG1655 promoter, so individual
  native-promoter claims should stay exploratory until that ontology is added.
- **DenseGen plan identity.** The anchor-source view is still the strongest
  plan-partitioning baseline, while bidirectional remains close enough to carry
  forward. This is the main reason not to overfit the story to Sigma-35 or W
  collection alone.

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
