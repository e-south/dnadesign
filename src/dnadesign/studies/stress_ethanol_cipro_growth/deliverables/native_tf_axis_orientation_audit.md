# Native TF-Axis Orientation Audit

This deliverable is a narrow calibration layer for the stress promoter study.
It asks whether native promoters associated with BaeR/CpxR and LexA point in
the DenseGen ethanol-like and ciprofloxacin/SOS-like latent directions before
batch-zero selection.

It is not an OPAL input, not an assay-label substitute, and not a broad
RegulonDB exploratory analysis. The intended row table is native-promoter only;
synthetic DenseGen rows provide the centroids used to define the two margins.

## Contract

- DenseGen centroid view: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Native audit rows: RegulonDB core60 rows appended through
  `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context`, then filtered
  by `derived__parent_dataset == usr_regulondb_native_promoters`
- Native bridge: reuse the existing pDual10 1 kb context handoff and existing
  forward/reverse-complement anchor-mean summaries; do not create a separate
  native context dataset
- Regulatory overlay: USR `_relations/regulatory_interactions.parquet`
- TF bins: BaeR/CpxR as `ethanol_TF`, LexA as `lexA_TF`, both as `mixed`,
  neither as `neither`

The companion test table runs only two one-sided checks:

- `ethanolness(BaeR or CpxR) > ethanolness(neither)`
- `ciproness(LexA) > ciproness(neither)`

Mixed rows are plotted but excluded from the two primary tests.

### native_tf_axis_orientation_audit | Native TF-axis orientation audit

#### Plot details

**Data.** Native RegulonDB core60 rows that have been appended through the
study anchor/context quota, embedded in the same pDual10 forward/RC anchor-mean
view as the DenseGen candidate surface, and joined to BaeR/CpxR/LexA
associations through the native promoter parent ID.

**Preprocessing.** DenseGen centroids are computed from the shared
bidirectional context-anchor view. Native rows are filtered by
`derived__parent_dataset == usr_regulondb_native_promoters`; TF flags are
derived at render time from `regulatory_interactions.parquet`.

**Definition.** The x-axis is ethanolness and the y-axis is ciproness, each
defined as cosine similarity to the relevant DenseGen centroid minus similarity
to the background centroid.

**Decision use.** The plot is an axis-orientation audit before batch-zero
selection. It can support or weaken trust in the DenseGen semantic axis names,
but it does not select OPAL candidates and does not act as assay evidence.

**Limits.** The plot is unavailable until the RegulonDB relations, anchor/context
append, and new 7B sidecars are complete. Native genomic context is not part of
this contract.

## Current State

The LatentDNA workspace now declares this as a planned first-class deliverable
and notebook plot option. It will render only after the RegulonDB regulatory
interaction sidecar is populated, native core60 rows are appended through the
existing study anchor/context quota, and matching 7B Infer feature sidecars are
filled for those appended rows.
