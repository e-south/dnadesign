# 2026-05-10 Candidate-X Rationale And Story Surfaces

## Purpose

This note records the study-level rationale for carrying
`intermediate_embedding_7b_context_anchor_mean_bidir_concat` forward as the
working pre-assay `X` for active-learning planning.

The purpose is not to find a scalar that declares a universal winner. The
purpose is to choose a representation that is healthy enough to trust, preserves
the synthetic design structure the study already knows, keeps strength-like
grammar visible, and gives reference landmarks a plausible but non-dominating
role.

## Decision

Use the equal-block forward/reverse-complement 1 kb context anchor-span concat
as the current working `X`.

Keep two companion interpretations:

- `intermediate_embedding_7b_anchor_60bp` remains the conservative DenseGen-plan
  baseline.
- `intermediate_embedding_7b_full_context_anchor_mean` remains the
  strength-standard lens, especially for the W collection.

This is a pragmatic active-learning decision, not a phenotype claim. The study
has not yet measured growth, ethanol response, ciprofloxacin response, or
promoter strength for the DenseGen designs.

## Evidence Summary

| Candidate view | Effective rank | PC1 fraction | DenseGen separation | Sigma-35 Spearman | Reference margin median | Strength Spearman median |
|---|---:|---:|---:|---:|---:|---:|
| Anchor-source causal mean | 10.99 | 0.261 | 1.494 | 0.451 | 0.087 | 0.321 |
| Forward 1 kb context anchor-span causal mean | 7.85 | 0.395 | 1.353 | 0.483 | 0.092 | 0.352 |
| Equal-block forward/RC 1 kb context anchor-span concat | 12.56 | 0.215 | 1.448 | 0.496 | 0.095 | 0.233 |

The bidirectional candidate is the best balanced working representation. It is
the healthiest candidate by effective rank, has the lowest PC1 dominance, gives
the strongest global Sigma-35 ordinal signal, and has the strongest median
reference-to-plan margin. It does not dominate every biological audit. The
anchor-source view still gives the strongest DenseGen plan separation, and the
forward context anchor-span view gives the strongest W collection strength
ordering.

That mixed result is useful. It argues against a contrived story in which every
metric is forced to support one view. The working decision is instead:
bidirectional is the best single representation to carry into downstream active
learning, while the companion views remain important controls for interpreting
what the model is seeing.

## Five Study Questions

### 1. DenseGen Plan Identity

The anchor-source causal mean remains the strongest DenseGen plan geometry
view. Its balanced design-family separation is `1.494`, compared with `1.448`
for the bidirectional candidate and `1.353` for the forward context anchor-span
view.

Interpretation: the anchor-source view is the cleanest way to ask whether
background, ethanol-regulator, ciprofloxacin/LexA/SOS, and dual-stress plans
partition. The bidirectional candidate is close enough on this axis that it
does not sacrifice plan identity for orientation robustness.

### 2. Strength-Like Grammar

The Sigma-35 B-through-F ladder is visible but moderate. The global Sigma-35
Spearman is highest for the bidirectional candidate at `0.496`. The forward
context anchor-span view is close at `0.483` and has the strongest balanced
Sigma-35 Spearman, `0.528`.

The W collection gives a separate strength-standard story. Its
collection-specific high-dimensional ordinal Spearman is strongest in the
forward context anchor-span view:

| Candidate view | W collection Spearman | Permutation p-value |
|---|---:|---:|
| Anchor-source causal mean | 0.419 | 0.0396 |
| Forward 1 kb context anchor-span causal mean | 0.658 | 0.0099 |
| Equal-block forward/RC 1 kb context anchor-span concat | 0.449 | 0.0099 |

Interpretation: the forward 1 kb context anchor-span view is the best lens for
known W-collection ordering. The bidirectional candidate remains stronger as a
general-purpose `X` because it balances this against health, Sigma-35, and
reference-to-plan behavior.

### 3. Stress-Reference Semantics

The user's `cell A` landmark is represented in this study as the sulA promoter
row, `sulAp`.

The core60 landmark check supports the intended qualitative story:

| Candidate view | `spyp_core60` nearest centroid | `sulAp_core60` nearest centroid |
|---|---|---|
| Anchor-source causal mean | Ethanol | Ciprofloxacin |
| Forward 1 kb context anchor-span causal mean | Ethanol | Ciprofloxacin |
| Equal-block forward/RC 1 kb context anchor-span concat | Ethanol | Ciprofloxacin |

The native-length rows do not behave as cleanly. Native `spyP` maps nearest
Ciprofloxacin in all three candidate views, and native `sulAp` maps nearest
Ciprofloxacin in the context views. That is not a failure of the bidirectional
concat. It is evidence that native-length references and core60 references are
different sequence-scope contracts, and that the core60 normalization is the
better reference-manifold check for this question.

The configured Native MG1655 reference set behaves as a broader, more
heterogeneous reference collection. This configured set contains 16 native
rows; it excludes the dedicated `spyP`/`sulAp` landmark set and currently also
excludes `soxSp`. For the bidirectional candidate, the configured set centroid
is nearest Ethanol with a weak native-length margin (`0.0245`) and a stronger
core60 margin (`0.0879`). At the individual-promoter level,
background-relative centroid margins show:

| Native MG1655 scope | Background | Ethanol | Ciprofloxacin | Dual |
|---|---:|---:|---:|---:|
| Native-length rows | 6 | 3 | 6 | 1 |
| Core60 rows | 2 | 8 | 3 | 3 |

Including all 19 native GenBank-backed rows changes the bidirectional core60
margin counts to `2 background / 9 ethanol / 5 ciprofloxacin / 3 dual`.

Interpretation: core60 shifts the configured Native MG1655 collection away
from a mostly diffuse/native-length view and toward stress-plan axes, but it
does not turn the collection into one clean ethanol-only or ciprofloxacin-only
landmark set. Individual promoter claims should wait for an explicit
expected-response metadata axis in the reference ontology.

### 4. Context And Orientation Robustness

The bidirectional candidate is an externally constructed two-orientation
summary. Evo2 is still causal in each emitted sequence. LatentDNA standardizes
and row-L2-normalizes the forward and reverse-complement anchor-span blocks
separately, then concatenates them along the feature axis.

Interpretation: this gives active learning a balanced representation of the
anchor from both emitted orientations without claiming a native bidirectional
Evo2 hidden state.

### 5. Non-Collapse

Rank and spread remain gates. They do not decide the biology by themselves.
The bidirectional candidate passes the gate most strongly: effective rank
`12.56`, PC1 fraction `0.215`, and input dimensionality `8192`. The forward
context anchor-span view passes but is more PC1-dominated. Output-layer means
remain mostly diagnostic and should not drive candidate-X selection.

## How To Read The Margin And Centroid Galleries

Several plots are intentionally related, but they are not duplicates. They are
different ways to ask whether the same candidate representation preserves
design-family identity, promoter-grammar structure, and reference-landmark
semantics.

### Balanced design-family margin gallery

This is the fairer DenseGen plan-identity scatter. Each plotted point is still a
full-population row, but the ethanol, ciprofloxacin, dual, and background
centroid directions are built from matched synthetic reference subsets.

In this study, "balanced" means the centroid builder stratifies rows by
`sig35_variant` and `spacer_length`. Within each retained stratum, it requires
the configured design families to be present, then samples the same number of
rows from each family. Strata that cannot support the matched comparison are
discarded. Because `balance_reference_only` is enabled, only the centroid
reference rows are balanced; the plotted scatter still shows the full
population.

The intuition is: if ethanol designs accidentally contain more of one
Sigma-35/spacer combination than background designs, a direct centroid might
partly measure that composition difference. The balanced version asks whether
ethanol/ciprofloxacin/background separation remains after matching those
obvious design axes.

### Design centroid margin gallery

This is the direct semantic map. It asks where each row sits relative to the
study-internal ethanol, ciprofloxacin, and background centroids:

```text
m_eth(x) = cos(z_x, c_eth) - cos(z_x, c_bg)
m_cipro(x) = cos(z_x, c_cipro) - cos(z_x, c_bg)
```

Positive values mean the row is closer to the target centroid than to
background on that axis. This plot is easy to explain and is useful for
reference overlays, such as asking whether `spyp_core60` or `sulAp_core60`
lands nearer the expected stress-plan side. Its limitation is that the direct
centroids can reflect design-family composition as well as family identity.

### Sigma-35 versus stress margin gallery

This plot asks whether one candidate space can carry two kinds of structure at
once.

The x axis is a Sigma-35 ladder margin:

```text
m_sigma35(x) = cos(z_x, c_f) - cos(z_x, c_b)
```

Higher values mean the row is more similar to the `F` Sigma-35 centroid than to
the `B` centroid. Lower or negative values mean the row is closer to the `B`
side. This is a geometric strength-like proxy, not measured expression.

The y axis is:

```text
m_stress(x) = max(m_eth(x), m_cipro(x))
```

This asks whether the row is closer to at least one stress-family centroid than
to background. It does not decide whether the row is specifically ethanol-like
or specifically ciprofloxacin-like; the separate ethanol/cipro margins are used
for that.

The desired pattern is not diffuse spread for its own sake. A useful candidate
shows interpretable structure in both directions: Sigma-35 classes should have
some x-axis ordering, stress-family rows should separate from background on the
y axis, and the two axes should not collapse into one indistinguishable trend.

### Sigma-35 centroid distance gallery

This is a group-level heatmap, not a row-level scatter. For each Sigma-35 group
`g`, the code averages the normalized row vectors in that group and normalizes
the result:

```text
c_g = normalize(mean(z_i for rows i in group g))
```

For two groups `g` and `h`, the heatmap value is:

```text
d_emb(g, h) = 1 - cos(c_g, c_h)
```

The diagonal is each centroid compared with itself and should be near zero. The
off-diagonal cells compare different variant centroids, such as `B` versus `F`
or `D` versus `E`. A coherent ladder does not require a perfect linear color
gradient, but it should avoid total collapse: adjacent variants may be closer,
far-apart variants should often be more separated, and the candidate view
should not place all B-F centroids in one indistinguishable block.

## Story Figures To Build Or Highlight

1. W collection ordinal-axis scatter.

   Plot W standard numeric strength on one axis and the high-dimensional
   ordinal-axis projection on the other, faceted by candidate view. Put
   Spearman, Kendall, and permutation p-value in each panel title. This shows
   why the forward context anchor-span view is the strength-standard lens
   without turning that one lens into the selected `X`.

2. spyP/sulAp native-to-core60 reference map.

   Show each landmark's similarity to Ethanol and Ciprofloxacin centroids
   before and after core60 normalization, with arrows from native-length to
   core60. The important story is not that all references behave perfectly. It
   is that the length-matched core60 versions recover the expected spyP/Ethanol
   and sulAp/Ciprofloxacin nearest-centroid relationships.

3. Native MG1655 individual reference-margin panel.

   The current reference-to-plan heatmap uses the Native MG1655 set centroid.
   A useful follow-up figure would show individual Native MG1655 promoters as
   rows and stress-plan margins as columns, with native and core60 rows paired.
   This would expose heterogeneity instead of hiding it behind one set
   centroid.

4. Candidate-X role card.

   Use one compact visual with three cards: working X, DenseGen-plan baseline,
   and strength-standard lens. Each card should name what that view is good for
   and what it should not be used to claim.

## Guardrails

- Do not say the bidirectional candidate is a native bidirectional Evo2
  embedding.
- Do not say rank selected the candidate. Rank only keeps collapsed views out.
- Do not pool Anderson and W collection strength values into one scale.
- Do not claim that Native MG1655 individual promoters validate ethanol or
  ciprofloxacin biology until expected-response labels are encoded and audited.
- Do not use UMAP position as the statistical evidence. Use high-dimensional
  cosine, centroid, and ordinal-axis metrics as the evidence; use UMAP as the
  visual orientation layer.
