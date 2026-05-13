# 2026-05-09 Candidate-View Language Prose Audit

## Review Scope

This audit reviews the study-facing language used for LatentDNA candidate views,
plot accordions, alt text, and deliverable notes. It is an author-side
manuscript-style prose review, not a new data analysis.

The checked prose must let a technical reader answer four questions without
guessing:

1. What sequence was passed to Evo2?
2. Which token positions were averaged?
3. Which emitted orientation was used?
4. What claim is allowed from the resulting row vector?

## Reviewer-Style Findings

### Finding 1: "Bidirectional" needed a positive definition and a boundary.

Previous wording correctly warned that the concat is not a native
bidirectional Evo2 state, but it sometimes sounded like a negative caveat
without explaining what the view does represent. The revised language defines
the candidate as an **equal-block forward/RC 1 kb context anchor-span summary**.

Allowed phrasing:

> The bidirectional candidate concatenates normalized forward and
> reverse-complement causal anchor-span means from matched 1 kb context passes.

Avoided phrasing:

> Evo2 produced a bidirectional promoter embedding.

### Finding 2: "Mean pooling" needed to say what axis is averaged.

The revised prose now states that Infer averages token-position vectors, not
embedding dimensions. This matters because the row vector keeps the embedding
dimension and collapses the sequence-position axis.

Allowed phrasing:

> `anchor_mean` averages the Evo2 token-position vectors over the
> Construct-provided anchor span.

Avoided phrasing:

> The embedding dimensions are averaged.

### Finding 3: "Full context" needed the causal caveat.

The revised prose now distinguishes "Evo2 received the full 1 kb emitted
sequence" from "each token state had downstream access." In a causal model, a
token state at position `t` is prefix-conditioned in the emitted orientation.

Allowed phrasing:

> The full 1 kb sequence is provided to Evo2, but each forward token state is
> still conditioned only on its prefix in that emitted sequence.

Avoided phrasing:

> Every anchor token sees the whole 1 kb context.

### Finding 4: Reference and standard rows needed scope boundaries.

The revised prose keeps native-length references, core60 windows, and
context-derived reference rows as separate sequence-scope contracts. This avoids
overclaiming that all reference rows live on a single perfectly matched
manifold.

Allowed phrasing:

> Native-length rows, derived core60 rows, and 1 kb context rows are compared as
> separate sequence-scope contracts.

Avoided phrasing:

> All reference promoters are directly comparable after pooling.

## House Style For Candidate Views

Use this order when describing a view:

1. Sequence: source insert, 1 kb context, reference core60, or reference
   context.
2. Orientation: forward or reverse-complement.
3. Pooling span: full emitted sequence, anchor span, or exact 60 bp core
   window.
4. Feature family: intermediate embedding or output-layer mean.
5. Claim boundary: triage geometry, QC diagnostic, reference landmark, or
   appendix orientation.

Example:

> The forward 1 kb context `anchor_mean` view runs Evo2 on the full emitted
> forward context, then averages the causal token-position hidden states over
> the Construct-provided anchor span. It is used to test whether vector context
> improves promoter-local geometry without averaging the whole 1 kb construct.

## Residual Risks

- Long plot labels may still force abbreviated panel titles. The accordion and
  alt text should carry the full technical description when the title must stay
  compact.
- "Bidirectional" remains useful shorthand, but each document should expand it
  once as "forward plus reverse-complement causal summaries."
- If a future view uses last-token pooling, attention pooling, or token-grid
  features, the current glossary must be updated rather than stretched to cover
  a different operation.
