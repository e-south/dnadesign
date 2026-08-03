## RT-lnRNA Trait-Axis Projection

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-30

This surface asks whether LatentDNA geometry contains a source-scoped
abundance direction for the current RT-lnRNA construct population. Crawford and
Khan remain separate evidence sources; their numeric abundance values are not
pooled into a shared target.

### Evidence Scope

`rt_lnrna_trait_axis_projection_rows` fits signed high-vs-low axes from
declared endpoint cohorts, then projects configured fit, eval, reference, and
sensitivity populations onto each axis. `rt_lnrna_trait_axis_projection_summary`
summarizes source-value correlations, endpoint separation, endpoint stability,
and direct Crawford-vs-Khan axis concordance when fitted-axis sidecars are
available. `rt_lnrna_trait_axis_projection_concordance` exposes that
concordance sidecar as a scalar table so the notebook can render it without a
trait-specific plotting path.

The first pass uses four intermediate candidate-X views:

- `intermediate_embedding_7b_dual_cassette_2000bp_fwd_rc_concat`
- `intermediate_embedding_7b_lnrna_fixed_384bp_window_in_construct_anchor_mean_bidir_concat`
- `intermediate_embedding_7b_rt_cds_fixed_1600bp_window_in_construct_anchor_mean_bidir_concat`
- `intermediate_embedding_7b_lnrna_384bp_rt_cds_1600bp_anchor_window_pair_concat`

### Source Semantics

Crawford abundance is the dense Eco1 msDNA abundance axis for the Crawford
source rows. Khan abundance is a sparse RT-DNA abundance audit for Khan source
rows. A strong Crawford axis does not imply Khan support, and a weak Khan axis
does not invalidate Crawford support unless the summary concordance and
source-value checks show that relationship explicitly.

GenBank catalog rows and compiler MSD rows are unlabeled reference or candidate
overlays. Their scores can support triage, but they are not labels and do not
define either abundance axis.

RT-CDS DMS rows are configured as held-out sensitivity rows. They must not fit
the abundance axes unless a future config change makes that override explicit.
Parent-relative `axis_delta` and `orthogonal_delta` interpretation is deferred
until parent mapping is complete and reliable in the workspace metadata.

Current Reader-backed reporter-response profiles support a 6-10 h provisional
descriptive comparison. They are not LatentDNA labels: biological-replicate
uncertainty and a constrained objective remain unresolved. This surface must
not claim assay-response predictive validity or use immutable historical
snapshots as labels.

### First-Pass Limits

- No RT-specific LatentDNA runtime branch owns these claims; RT meaning lives in
  workspace config and study docs.
- The surface is scalar-first. Plot and notebook presentation consume row,
  summary, and concordance scalar artifacts without loading raw representation
  matrices.
- Endpoint sensitivity is available for the configured endpoint definitions,
  and the primary review path now includes a stability panel.
- DMS parent-relative movement is deferred because the current config does not
  declare a parent key and parent candidate mapping.
