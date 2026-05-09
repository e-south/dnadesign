# 2026-05-09 Bidirectional Context-Anchor Mean Confidence Audit

## Decision

The current working candidate,
`intermediate_embedding_7b_context_anchor_mean_bidir_concat`, is defensible as
an externally constructed two-orientation 1 kb context-anchor representation.
It is analogous to the usual bidirectional workaround for a causal sequence
model: run the model once in the forward orientation, run it again on the
reverse-complement sequence, summarize the same biological span in each emitted
orientation, then concatenate the two summaries.

The precise claim is:

> This view combines left-to-right and right-to-left evidence around the anchor
> by concatenating equal-weight forward and reverse-complement causal
> mean-pooled anchor-span embeddings from matched 1 kb contexts.

The claim that should not be made is:

> Evo2 itself produced a native bidirectional hidden state for each anchor
> token.

That is not the implementation and not the model contract.

## Evidence Chain

1. Construct emits paired 1 kb context records.

   Forward rows preserve the forward context sequence and forward anchor bounds.
   Reverse-complement rows store the reverse-complement sequence and
   reverse-complement-orientation anchor bounds. The reverse row is therefore a
   real second emitted sequence, not a coordinate transform applied after
   inference.

2. Infer pools over the explicit emitted-orientation span.

   For `anchor_mean` and `core60_mean`, Infer validates `pooling_start_0` and
   `pooling_end_0`, then computes `tensor[start:end].mean(dim=0)`. The mean is
   over token positions. `core60_mean` additionally rejects spans that are not
   exactly 60 bp.

3. Evo2 token states are causal in the emitted orientation.

   For a forward emitted sequence \(x_{1:T}\), the layer-\(\ell\) state at
   position \(t\) should be read as

   $$
   \overrightarrow{h}^{(\ell)}_t =
   f_\theta^{(\ell)}(x_{\le t}).
   $$

   A forward anchor-span mean is therefore

   $$
   \overrightarrow{z}^{(\ell)}_I =
   |I|^{-1}\sum_{t \in I}\overrightarrow{h}^{(\ell)}_t.
   $$

   It summarizes the anchor with progressively more within-span prefix context,
   but early anchor positions do not see later anchor or downstream-context
   bases.

4. The reverse-complement pass supplies the complementary causal direction.

   On the reverse-complement emitted sequence, positions that are downstream in
   the forward biological orientation become prefix-side context in the reverse
   emitted orientation. Its anchor-span mean is

   $$
   \overleftarrow{z}^{(\ell)}_I =
   |I|^{-1}\sum_{t \in I_\mathrm{rc}}\overleftarrow{h}^{(\ell)}_t.
   $$

   This gives the study an external right-context summary without pretending
   that the forward token states had future access.

5. LatentDNA concatenates the two blocks with equal-block normalization.

   The derived view loads the forward and reverse-complement `anchor_mean`
   matrices, aligns rows, centers/scales each block, L2-normalizes each row
   within each block, and column-stacks the blocks:

   $$
   X_\mathrm{bidir} =
   [\mathrm{L2}(Z_\mathrm{fwd});\mathrm{L2}(Z_\mathrm{rc})].
   $$

   Equal-block normalization matters because otherwise the chosen candidate
   could be dominated by whichever orientation block has larger raw variance or
   norm.

## How To Describe The Views

Use these plain-language names:

- `anchor_60bp_mean`: anchor-source causal mean. This is the conservative
  DenseGen-plan baseline because it keeps the representation closest to the
  inserted promoter sequence. The name is convenient, but the merged source
  also carries variable-length native/reference/control rows.
- `context_anchor_mean_fwd`: forward 1 kb context anchor-span causal mean. This
  asks whether placing the anchor inside expression-vector context improves
  grammar-like signals such as the Sigma-35 ladder.
- `context_anchor_mean_bidir_concat`: two-orientation 1 kb context anchor-span
  mean. This is the working `X` because it combines forward and
  reverse-complement causal summaries while preserving a compact candidate set.

Use these technical names when precision matters:

- prefix-conditioned mean-pooled span embedding
- causal mean-pooled anchor-span embedding
- equal-block forward/RC anchor-mean concat
- externally constructed two-orientation representation

Avoid these phrases unless they are immediately qualified:

- "bidirectional Evo2 embedding"
- "fully bidirectional hidden state"
- "the anchor sees the whole 1 kb context"
- "60 bp candidate" when referring to every row in the merged anchor-source
  candidate

## Confidence Read

The implementation matches the intended bidirectional workaround in principle:
provide the full 1 kb forward context, provide the full 1 kb reverse-complement
context, pool the same biological anchor/core span in each emitted orientation,
then concatenate the two orientation summaries.

The main caveat is semantic, not mechanical. The concat gives the row a
bidirectional *summary*, not bidirectional token states. It cannot recover a
single token-level representation where every forward token directly attended
to both upstream and downstream bases. For the study's current purpose,
pre-assay candidate-X selection, that caveat is acceptable because the scorecard
uses row-level geometry, not token-level mechanistic claims.

The right shorthand is therefore:

> bidirectional 1 kb context-anchor summary

not:

> native bidirectional Evo2 promoter embedding.

## Residual Risks

- Mean pooling still discards position-specific structure. A later assay-era
  model may want last-token, endpoint, attention-style, or token-grid features
  if row-level pooled geometry is insufficient.
- The merged anchor-source candidate is not literally same-length for all rows;
  native references and controls can remain off the strict 60 bp manifold.
- Reference-to-plan margins remain weak enough that spyP/sulAp and other
  natural promoters should stay landmark sanity checks, not selection criteria.
- "Bidirectional" in figure captions should always be expanded once per
  document as "forward plus reverse-complement causal summaries."
