---
doc_id: study-eco1-rt-repack-msa-method
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-20
---

## MSA Method

This page explains how the Eco1 RT conservation profile should be built. It is
not an MSA, and it does not satisfy `conservation_profile.parquet`.

The machine-readable source contract is:

```text
docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml
```

The source-discovery note is:

```text
docs/studies/eco1_rt_repack/workbench/provenance/conservation-source-discovery.md
```

### Current Authority

Eco1/Ec86 RT is treated as the ec86kit-pinned reference sequence used by the
selected structure and residue map. The conservation target row must match:

```text
sha256:429a9c9894501e04f48803b96307cea45955f63b85f1461dc25c017e94b7eaeb
```

Mestre et al. 2020 Supplementary Table S1 is the roster authority, not a
finished alignment. The Eco1/Ec86 row is Node 1550, RT clade 9, retron subtype
II-A3, cluster/domain `42_1`, accession `WP_099010551.1`.

The selected Phase 1 conservation profiles are:

```text
broad_tao_homolog_rt  Tao-like target-centered bounded homolog panel drawn from Mestre S1
eco1_like_retron_rt   Mestre II-A3 cluster 42_1 after declared filters
```

The full Mestre roster is also retained as context:

```text
full_mestre_retron_rt  classification/candidate-pool context, not the Phase 1 denominator
```

### Procedure

1. Start from the Mestre S1 roster declared in `conservation-sources.yaml`.
2. Treat the full Mestre roster as a candidate pool and context surface, not
   as the conservation-scoring denominator.
3. Split the active source authority into `broad_tao_homolog_rt` and
   `eco1_like_retron_rt`.
4. Fetch candidate protein sequences through declared providers only:
   `ncbi_protein_efetch` for NCBI Protein accessions in S1, including
   `WP_*` and GenBank-style protein ids such as `EIJ70524.1`, and
   `bv_brc_feature_protein_fasta` for `fig|*` feature ids.
5. Exclude unresolved provider rows only with an explicit reason; do not
   silently drop them.
6. Materialize provider FASTA source files from the hash-pinned Mestre roster
   table. Provider-missing accessions must be written to an explicit failure
   ledger before they can become excluded source records.
7. For `broad_tao_homolog_rt`, run a bounded homolog selector over provider
   sequences before source-record materialization. The selector must compute
   target-centered coverage, identity, motif support, and deterministic
   diversity/cap metadata; it must not fall back to raw roster order.
8. Materialize the local roster/source cache from the bounded broad selector,
   the Eco1-like roster selector, and explicit provider FASTA sources. This
   writes `source_records.yaml`, filtered provider cache FASTAs, and a cache
   manifest.
9. Materialize unaligned source FASTA bundles from the local provider caches
   and `source_records.yaml`; each bundle must insert the ec86kit Eco1 RT
   sequence as the explicit target FASTA row.
10. Reject `WP_099010551.1` as the target row unless the T301/A301 discrepancy
   is explicitly adjudicated.
11. Run the source-sequence sufficiency gate; reject missing cache roots,
   placeholder accessions, undersized profile bundles, missing source hashes,
   provider hash drift, and exclusions without reasons.
12. Apply the declared filters: query coverage, identity range, length range,
   required RT/retron motifs, and excluded RT families.
13. Align proteins with the declared MSA backend command from the source contract
   through `dnadesign.aligner.msa`.
14. Map alignment columns back to `residue_map.parquet` through canonical Eco1
   positions, not raw PDB residue ids.
15. Compute conservation using non-gap rows as the denominator.
16. Emit `conservation_profile.parquet` only after every row has source hashes,
    target-row provenance, profile id, WT amino acid, plurality amino acid, WT
    frequency, non-gap count, and pass/fail status.

### Tao-Style Conservation Rule

The Eco1 profile follows the Tao et al. rule shape:

```text
fixed_by_conservation =
  wt_aa_is_plurality_aa
  AND wt_frequency >= conservation_threshold
```

For the first conservative profile, `conservation_threshold` is `0.25`.

The MSA is evidence for masking, not an activity model. It cannot make a
residue designable. Missing MSA evidence fails closed until a later operator
explicitly changes the policy.

### T301/A301 Handling

The selected ec86kit/structure authority has T301. A direct NCBI fetch of
`WP_099010551.1` observed A301. Position 301 is near the C terminus but is
resolved in the selected structure and already contact-proximal under the 20 A
retained-context policy.

This is a source-authority mismatch, not a biological conclusion. The MSA
target must be the ec86kit sequence unless a future contract explicitly
declares a substitution and updates all linked hashes.

### Fail-Fast Rules

- No conservation profile without `conservation-sources.yaml`.
- No target row inferred from a public accession with a sequence mismatch.
- No provider fallback outside the declared provider ids.
- No MAFFT alignment from a source bundle that fails the sufficiency gate.
- No figure-level or prose-only MSA used as materialized evidence.
- No conservation count with gaps in the denominator.
- No fixed-position conservation rule unless WT is the plurality amino acid.
- No designability from missing conservation evidence.

### Provider-Source Materializer

The provider-source materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/provider_sources/
```

It consumes the hash-pinned Mestre S1 roster table, derives declared NCBI and
BV-BRC provider accessions from the selected source groups and candidate-pool
rules, resolves provider identity through `sequence_providers[*].accession_patterns`
in `conservation-sources.yaml`, and writes explicit provider FASTA source files:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/ncbi_protein_efetch.fasta
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/bv_brc_feature_protein_fasta.fasta
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/provider_source_manifest.yaml
```

If a declared provider does not return requested records, those records may
only be carried forward through an explicit failure ledger:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_provider_sources/provider_source_failures.yaml
```

Current local real-data counts:

```text
ncbi_protein_efetch requested 350, returned 350
bv_brc_feature_protein_fasta requested 1577, returned 1464, unresolved 113
```

These provider-source files are candidate-pool inputs for the bounded selector.
They do not by themselves define the broad conservation denominator.

Command shape:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources \
  --repo-root . \
  --roster-table <mestre-s1-roster.xlsx> \
  --write-unresolved-ledger
```

### Roster-Cache Materializer

The roster-cache materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/
```

It consumes selected source records plus explicit provider FASTA sources:

```text
<roster-table>.csv|tsv|xlsx
<provider-source-root>/ncbi_protein_efetch.fasta
<provider-source-root>/bv_brc_feature_protein_fasta.fasta
```

Roster tables may carry optional `source_cache_status` and
`exclusion_reason` columns. Rows default to `included`; rows marked
`excluded` must include a reason and do not require a provider FASTA sequence.
Provider accession shapes are not hard-coded in roster-cache; they are compiled
from the checked-in conservation source contract and reused by the sufficiency
gate.

Under the revised source contract, `broad_tao_homolog_rt` cannot be materialized
directly from the full Mestre roster. Roster-cache materialization now fails
for that profile until `conservation-bounded-homolog-selector-v1` emits
bounded source records with selector metadata. This is intentional: the full
Mestre roster is a candidate pool and display context, not the scoring
denominator.

By default it requires the roster-table hash to match
`conservation-sources.yaml`. Test fixtures may use
`--allow-uncontracted-roster-hash`, but real study data should not. It writes
the local source cache:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/source_records.yaml
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/provider_caches/
outputs/thread/eco1_rt_conservative_v1/conservation_source_cache/source_cache_manifest.yaml
```

Command shape:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache \
  --repo-root . \
  --roster-table <mestre-s1-roster.csv-or-xlsx> \
  --provider-source-root <provider-fasta-source-root> \
  --provider-failure-ledger <provider-source-root>/provider_source_failures.yaml
```

The materializer does not perform live NCBI or BV-BRC network retrieval. It
ingests explicit provider FASTA source files so provider drift remains visible.

### Source-Sequence Bundle Materializer

The source-sequence bundle materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/
```

It consumes explicit local source caches produced by the roster-cache layer:

```text
<source-cache-root>/source_records.yaml
<source-cache-root>/provider_caches/ncbi_protein_efetch.fasta
<source-cache-root>/provider_caches/bv_brc_feature_protein_fasta.fasta
```

The ledger records `profile_id`, `record_id`, `provider_id`, `accession`,
`status`, and an `exclusion_reason` for excluded rows. The materializer inserts
`eco1_rt_ec86kit_reference` itself, rejects operator-supplied target rows, and
writes unaligned source FASTA plus manifests:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_sources/
```

It does not fetch live provider records and it does not run MAFFT.

Before alignment, run the sufficiency preflight:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency --repo-root .
```

This command must pass before MAFFT. The previous full-roster broad source
bundle passed with:

```text
broad_tao_homolog_rt included 1814, excluded 114
eco1_like_retron_rt included 46, excluded 1
```

That broad bundle is now superseded candidate-pool context. It should not be
aligned or scored as the active `broad_tao_homolog_rt` denominator.

It rejects source bundles that are fixture-like, under-supported relative to
`min_non_gap_count`, not hash-linked to `source_records.yaml` and provider
caches, or populated with placeholder accessions such as synthetic `WP_BROAD`
or `fig|BROAD` records.

### Alignment Bundle Materializer

The study-owned alignment materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/
```

It validates the source-sequence sufficiency gate, reads the declared MAFFT
command from `conservation-sources.yaml`, and delegates generic alignment
execution to `dnadesign.aligner.msa`. The generic aligner wrapper writes MAFFT
stdout to a temporary FASTA, validates that aligned FASTA, and publishes the
final FASTA plus manifest only after validation. Stderr is recorded as an
explicit sidecar so interrupted or timed-out runs do not masquerade as accepted
aligned bundles.

Run it through Pixi so MAFFT comes from the repository native-tool environment:

```bash
pixi run uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments --repo-root .
```

To operate one profile at a time, repeat `--profile-id` for the intended
declared profile ids:

```bash
pixi run uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments \
  --repo-root . \
  --profile-id eco1_like_retron_rt
```

The current declared command is:

```text
mafft --globalpair --maxiterate 1000 --reorder <input_fasta> > <output_fasta>
```

An interactive real-data run of the former full-roster broad profile ran for
roughly four hours of active CPU before being interrupted without producing an
accepted `broad_tao_homolog_rt.aligned.fasta`. Do not switch to a faster MSA
backend or preset silently. First materialize the bounded broad homolog source
records; then either run the declared backend policy or update the source
contract with an explicit profile-specific alignment policy and benchmark
evidence.

A selected-profile local run of `eco1_like_retron_rt` completed through the
declared command and published:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_alignments/eco1_like_retron_rt.aligned.fasta
outputs/thread/eco1_rt_conservative_v1/conservation_alignments/eco1_like_retron_rt.aligned.manifest.yaml
```

The accepted profile has 47 aligned records, one aligned length of 560 aa, the
`eco1_rt_ec86kit_reference` target row, MAFFT v7.526, return code 0, and
hash-linked stdout/stderr provenance. Treat this as accepted profile-level
evidence only; conservation-profile materialization still requires the broad
profile alignment as well.

### MSA Visualization Sidecars

The generic visualization API is:

```text
src/dnadesign/aligner/msa/visualization/
```

It consumes accepted aligned FASTA files and writes diagnostic sidecars only.
Its internal ontology is:

```text
contracts/        generic request/result models and YAML readers
materialization/  orchestration, QC calculations, manifests, CSV, and HTML
renderers/        SVG panel renderers and label placement
```

Eco1 supplies profile IDs, the target row ID/hash, optional study-owned
target-position annotations, and the output location:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_visualizations/
```

The optional Eco1 annotation track is:

```text
docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml
```

The optional Eco1 exemplar-row selection is:

```text
docs/studies/eco1_rt_repack/workbench/ontology/msa-exemplar-rows.yaml
```

The optional Eco1 panel spec is:

```text
docs/studies/eco1_rt_repack/workbench/ontology/msa-panel-spec.yaml
```

It follows the visualization precedent from validated retron RT MSAs: Simon et
al. boxed conserved RT regions and highlighted retron-specific X/Y and
catalytic motif anchors, while Mestre et al. used RT0-RT7 alignments to define
retron RT clades. The current Eco1 track renders only audited motif anchors
(`NAxxH`, `YADD`, and `VTG`). Full RT0-RT7 interval boxes should be added only
after a dedicated Eco1 residue-numbering/motif audit.

The Eco1 annotation track uses two display layers:

1. light bordered context spans around Region X, the catalytic YADD context,
   and Region Y; and
2. stronger filled motif anchors for `NAxxH`, `YADD`, and `VTG`.

Context-span labels are declared above the spans, while compact motif-anchor
labels are declared below the anchors. That placement is an Eco1 display
choice in `rt-annotation-tracks.yaml`, not a renderer assumption.

Those border/fill styles are figure grammar only. They are useful because they
make masking-relevant neighborhoods visible, but the actual mask source remains
`mask_set.yaml` after contact and conservation evidence are materialized.

The visual report should not be a single opaque heatmap. The generic renderer
therefore emits four complementary layers:

1. a global target-position QC track for gap/plurality inspection; and
2. local exemplar-row windows around annotated motif anchors;
3. selected-row whole-alignment overview panels; and
4. target-position plurality/gap histograms.

The exemplar rows and panel spec make motif-local and whole-alignment variation
visible without changing the conservation denominator. Rows are not selected
automatically from FASTA order, because that would introduce avoidable
footguns: hidden row-order bias, over-representation of near-duplicates,
accidental inclusion of a target duplicate, and cherry-picked examples that
make plurality look stronger than it is. The panel spec can declare
display-only high-gap trimming for publication views, but conservation scoring
must use the untrimmed accepted aligned FASTA. Any publication figure should
label exemplar-row source and selection rule explicitly.

Simon-style cross-family controls such as RT-Mxa1, RT-Sen2, Group II RT-RI, or
DGR bRT are not silently inserted into the current conservation MSA. They would
be valuable publication references, but they need a separate declared
cross-family FASTA/reference bundle because they are display controls, not part
of the retron RT conservation denominator used for masking.

Run a strict complete report after both required profiles exist:

```bash
uv run python -m dnadesign.aligner.msa.visualization \
  --alignment-root outputs/thread/eco1_rt_conservative_v1/conservation_alignments \
  --output-root outputs/thread/eco1_rt_conservative_v1/conservation_visualizations \
  --profile-id broad_tao_homolog_rt \
  --profile-id eco1_like_retron_rt \
  --target-row-id eco1_rt_ec86kit_reference \
  --target-sequence-hash sha256:429a9c9894501e04f48803b96307cea45955f63b85f1461dc25c017e94b7eaeb \
  --annotation-tracks-yaml docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml \
  --exemplar-rows-yaml docs/studies/eco1_rt_repack/workbench/ontology/msa-exemplar-rows.yaml \
  --panel-spec-yaml docs/studies/eco1_rt_repack/workbench/ontology/msa-panel-spec.yaml
```

While only one profile is accepted, an explicit partial report can be generated:

```bash
uv run python -m dnadesign.aligner.msa.visualization \
  --alignment-root outputs/thread/eco1_rt_conservative_v1/conservation_alignments \
  --output-root outputs/thread/eco1_rt_conservative_v1/conservation_visualizations \
  --profile-id broad_tao_homolog_rt \
  --profile-id eco1_like_retron_rt \
  --target-row-id eco1_rt_ec86kit_reference \
  --target-sequence-hash sha256:429a9c9894501e04f48803b96307cea45955f63b85f1461dc25c017e94b7eaeb \
  --annotation-tracks-yaml docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml \
  --exemplar-rows-yaml docs/studies/eco1_rt_repack/workbench/ontology/msa-exemplar-rows.yaml \
  --panel-spec-yaml docs/studies/eco1_rt_repack/workbench/ontology/msa-panel-spec.yaml \
  --allow-missing-profiles
```

The sidecars include per-profile MSA QC YAML, per-position QC CSV, SVG tracks,
selected-row overview panels, plurality/gap histograms, an HTML summary, and an
index manifest. They are inspection aids only. They do not decide designability
and do not replace `conservation_profile.parquet` or `mask_set.yaml`. Eco1
source authority, provider ledgers, conservation scoring, and mask policy must
not move into `aligner`.

### MAFFT Runtime Log Interpretation

MAFFT progress lines such as `STEP 001`, `STEP 002`, `accepted`, `rejected`,
and `identical` are aligner-internal iterative-refinement messages. They are
not Eco1 pipeline phases and they are not artifact acceptance states. In this
study, an aligned FASTA becomes accepted only after the generic
`dnadesign.aligner.msa` backend exits with return code 0, validates equal
aligned lengths and the pinned target row, hash-links the inputs and outputs,
and atomically publishes the final FASTA plus manifest.

### Conservation Profile Materializer

The study-owned materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/
```

It consumes explicit aligned FASTA files, one per selected profile id:

```text
<alignment-root>/broad_tao_homolog_rt.aligned.fasta
<alignment-root>/eco1_like_retron_rt.aligned.fasta
```

Each aligned FASTA must include the target row:

```text
eco1_rt_ec86kit_reference
```

The materializer writes:

```text
outputs/thread/eco1_rt_conservative_v1/conservation_profile.parquet
```

It validates the target row against `residue_map.parquet`, records aligned
FASTA source hashes, and emits long-form rows keyed by
`profile_id + canonical_position`.

This materializer does not fetch provider sequences or run MAFFT. It requires
an accepted aligned FASTA bundle from the alignment materializer before it can
create real conservation evidence.

### Next Slice

The next data slice is `conservation-bounded-homolog-selector-v1`: materialize
bounded `broad_tao_homolog_rt` source records from the full Mestre
candidate-pool provider cache using target-centered coverage, identity, motif
support, and deterministic diversity/cap metadata. After that, rerun source
FASTA sufficiency, align the bounded broad profile through `dnadesign.aligner.msa`,
run the conservation materializer, and confirm Phase 1 advances to the
`mask_set.yaml` blocker only.
