---
doc_id: study-eco1-rt-repack-msa-method
surface: study-context
study_id: eco1_rt_repack
owner: dnadesign-maintainers
last_verified: 2026-06-22
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
ec86_clade9_conservation_v1               Mestre RT clade 9 homolog panel after QC
ec86_iia3_cluster42_1_conservation_v1     Mestre II-A3 cluster 42_1 family panel after QC
```

The full Mestre roster is also retained as context:

```text
mestre_all_retron_rt_context  classification/candidate-pool context, not the Phase 1 denominator
```

### Reviewer-Facing Method Logic

Evolutionary conservation is estimated from Mestre-derived homolog panels, not
from the entire retron RT census. Mestre S1 is the accession and classification authority.
For Eco1/Ec86, the broad homolog panel is RT clade 9, the
Mestre-defined RT clade containing Eco1/Ec86; the Eco1-family panel is the
narrower subtype II-A3, cluster/domain `42_1`. Both panels are filtered for
coverage, length, source availability, hard-reject structural obviousness, and
motif QC before alignment. Conservation masking then follows Tao et al.'s
plurality/frequency rule: an Eco1 residue is fixed when the Eco1 amino acid is
the plurality amino acid at the aligned column and its non-gap frequency meets
the declared threshold. The full Mestre roster is retained for context and
visualization, but is not the Phase 1 conservation denominator.

This is the key anti-footgun for reviewers and future agents: the study is
running homolog MSA conservation scoring, not a whole-database census alignment.
The full Mestre table can contextualize clades and figures, while the two
selected panels define the denominators used by
`conservation_profile.parquet`.

### Procedure

1. Start from the Mestre S1 roster declared in `conservation-sources.yaml`.
2. Treat the full Mestre roster as a candidate pool and context surface, not
   as the conservation-scoring denominator.
3. Split the active source authority into `ec86_clade9_conservation_v1` and
   `ec86_iia3_cluster42_1_conservation_v1`.
4. Fetch candidate protein sequences through declared providers only:
   `ncbi_protein_efetch` for NCBI Protein accessions in S1, including
   `WP_*` and GenBank-style protein ids such as `EIJ70524.1`, and
   `bv_brc_feature_protein_fasta` for `fig|*` feature ids.
5. Exclude unresolved provider rows only with an explicit reason; do not
   silently drop them.
6. Materialize provider FASTA source files from the hash-pinned Mestre roster
   table. Provider-missing accessions must be written to an explicit failure
   ledger before they can become excluded source records.
7. For `ec86_clade9_conservation_v1`, select Mestre RT clade 9 rows after
   declared QC. Clade 9 is the natural broad homolog unit because it is the
   Mestre-defined RT clade containing Eco1/Ec86; do not replace it with a
   cap-first subset or the full Mestre census.
8. Materialize the local roster/source cache from the clade 9 selector, the
   II-A3/`42_1` selector, and explicit provider FASTA sources. This
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
   hard-reject filters, motif-QC markers, and excluded RT families. Motif
   deviations are recorded as QC evidence; they are not silently converted into
   regex-only inclusion/exclusion decisions.
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
- No MSA alignment from a source bundle that fails the sufficiency gate.
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
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_provider_sources/ncbi_protein_efetch.fasta
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_provider_sources/bv_brc_feature_protein_fasta.fasta
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_provider_sources/provider_source_manifest.yaml
```

If a declared provider does not return requested records, those records may
only be carried forward through an explicit failure ledger:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_provider_sources/provider_source_failures.yaml
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

Under the revised source contract, `ec86_clade9_conservation_v1` is
materialized from Mestre RT clade 9 after QC, not from the full Mestre roster.
This is intentional: the full Mestre roster is a candidate pool and display
context, not the scoring denominator. A runtime cap may be introduced only by a
future benchmarked policy update; it is not part of the biological source-set
definition.

The source-record QC is a pre-MSA gate. It computes pairwise target coverage,
pairwise identity-to-target, sequence length status, motif-marker calls, and
hard-reject filters. Identity is evaluated with a target-vs-provider global
pairwise alignment rather than raw index-wise comparison, because raw
positional identity rejects legitimate RT homologs with insertions or terminal
extensions. Motif deviations are recorded as QC markers; only the declared
hard-reject filters exclude rows.

By default it requires the roster-table hash to match
`conservation-sources.yaml`. Test fixtures may use
`--allow-uncontracted-roster-hash`, but real study data should not. It writes
the local source cache:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_source_cache/source_records.yaml
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_source_cache/provider_caches/
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_source_cache/source_cache_manifest.yaml
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
writes unaligned source FASTA plus manifests. Included-row `sequence_qc`
metadata is preserved into the profile manifests so the sufficiency gate can
reject hand-authored or stale bundles before alignment:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_sources/
```

It does not fetch live provider records and it does not run the MSA backend.

Before alignment, run the sufficiency preflight:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency --repo-root .
```

This command must pass before alignment. The current selected source bundles
pass locally with:

```text
ec86_clade9_conservation_v1 included 302, excluded 22
ec86_iia3_cluster42_1_conservation_v1 included 44, excluded 3
```

The older 1814-row full-roster broad bundle is now superseded candidate-pool
context. It should not be aligned or scored as the active
`ec86_clade9_conservation_v1` denominator.

It rejects source bundles that are fixture-like, under-supported relative to
`min_non_gap_count`, not hash-linked to `source_records.yaml` and provider
caches, or populated with placeholder accessions such as synthetic `WP_BROAD`
or `fig|BROAD` records.

### Alignment Bundle Materializer

The study-owned alignment materializer is:

```text
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/
```

It validates the source-sequence sufficiency gate, reads the declared MSA
backend command from `conservation-sources.yaml`, and delegates generic
alignment execution to `dnadesign.aligner.msa`. The generic aligner wrapper
stages backend output, validates that aligned FASTA, and publishes the final
FASTA plus manifest only after validation. Stderr is recorded as an explicit
sidecar so interrupted or timed-out runs do not masquerade as accepted aligned
bundles.

Run it through Pixi so the declared native alignment backend comes from the
repository tool environment:

```bash
pixi run uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments --repo-root .
```

To operate one profile at a time, repeat `--profile-id` for the intended
declared profile ids:

```bash
pixi run uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments \
  --repo-root . \
  --profile-id ec86_iia3_cluster42_1_conservation_v1
```

The current declared command is:

```text
clustalo --force --outfmt=fasta --threads=1 -i <input_fasta> -o <output_fasta>
```

An interactive real-data run of the former full-roster broad profile ran for
roughly four hours of active CPU before being interrupted without producing an
accepted broad-profile aligned FASTA. That run used the previous
high-sensitivity MAFFT policy and the full-roster denominator; it is historical
runtime evidence, not the selected Phase 1 alignment policy. Do not switch MSA
backends or presets silently. Clustal Omega is now selected by contract for the
Mestre clade 9 and II-A3/`42_1` homolog panels.

A complete local run of both selected profiles completed through the declared
Clustal Omega command and published:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_alignments/ec86_clade9_conservation_v1.aligned.fasta
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_alignments/ec86_clade9_conservation_v1.aligned.manifest.yaml
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_alignments/ec86_iia3_cluster42_1_conservation_v1.aligned.fasta
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_alignments/ec86_iia3_cluster42_1_conservation_v1.aligned.manifest.yaml
```

The accepted clade-9 alignment has 303 records and aligned length 853. The
accepted II-A3/`42_1` alignment has 45 records and aligned length 527. Both
include the pinned `eco1_rt_ec86kit_reference` target row and hash-linked
manifests.

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
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_visualizations/
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
retron RT clades. The current Eco1 track renders the audited RT1-RT7 interval
spans plus motif anchors (`NAxxH`, `YADD`, and `VTG`). Those coordinates are
declared in the ontology; they are not renderer constants.

The Eco1 annotation track uses three display layers:

1. light bordered context spans around Region X, the catalytic YADD context,
   and Region Y;
2. RT1-RT7 interval boxes that mirror `manual-mask-authority.yaml`; and
3. stronger filled motif anchors for `NAxxH`, `YADD`, and `VTG`.

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
  --alignment-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_alignments \
  --output-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_visualizations \
  --profile-id ec86_clade9_conservation_v1 \
  --profile-id ec86_iia3_cluster42_1_conservation_v1 \
  --target-row-id eco1_rt_ec86kit_reference \
  --target-sequence-hash sha256:429a9c9894501e04f48803b96307cea45955f63b85f1461dc25c017e94b7eaeb \
  --annotation-tracks-yaml docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml \
  --exemplar-rows-yaml docs/studies/eco1_rt_repack/workbench/ontology/msa-exemplar-rows.yaml \
  --panel-spec-yaml docs/studies/eco1_rt_repack/workbench/ontology/msa-panel-spec.yaml
```

While only one profile is accepted, an explicit partial report can be generated:

```bash
uv run python -m dnadesign.aligner.msa.visualization \
  --alignment-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_alignments \
  --output-root src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_visualizations \
  --profile-id ec86_clade9_conservation_v1 \
  --profile-id ec86_iia3_cluster42_1_conservation_v1 \
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
<alignment-root>/ec86_clade9_conservation_v1.aligned.fasta
<alignment-root>/ec86_iia3_cluster42_1_conservation_v1.aligned.fasta
```

Each aligned FASTA must include the target row:

```text
eco1_rt_ec86kit_reference
```

The materializer writes:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/conservation_profile.parquet
```

It validates the target row against `residue_map.parquet`, records aligned
FASTA source hashes, and emits long-form rows keyed by
`profile_id + canonical_position`. The current local materialized profile has
640 rows: 320 per selected profile.

This materializer does not fetch provider sequences or run the MSA backend. It
requires an accepted aligned FASTA bundle from the alignment materializer before
it can create real conservation evidence.

### Current Mask Use

The clade-9 source cache, source FASTA sufficiency gate, accepted Clustal Omega
alignment bundle, generic MSA visualization sidecars, and
`conservation_profile.parquet` are now available for the selected profile IDs.
The current mask rule uses the Ec86 clade 9 profile through a hard
WT-plurality rule: an Eco1 amino acid is protected when it is evolutionarily
conserved at `>=25%` WT plurality in the clade 9 homolog MSA.

The current mask is `eco1_rt_clade9_plurality25_direct_contact5a_v1`: protect
NAxxH/YADD/VTG, Wang/Ec86 direct substrate-contact priors, Ec86 clade 9 >=25%
WT-plurality conservation calls, and mapped residues within 5 A of retained
DNA/RNA.
Terminal residues `1`, `2`, and `312-320` are `non_fixed_missing_backbone`.
Paired-protomer dimerization is not a retention objective for this profile, so
pre-RT1 residues are not fixed solely to preserve the dimer.
