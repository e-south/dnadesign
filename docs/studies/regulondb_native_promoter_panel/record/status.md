## regulondb_native_promoter_panel

- Last verified: 2026-05-04
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Lifecycle posture: inactive source-intake lane; local native USR and TSS-upstream core60 datasets are materialized, the standard local Evo2 7B Infer sidecars are complete for the native/full and derived core60 lanes, and the current checked-in phase is the LatentDNA native/core60 audit

### Evo2 DNA case incident

- On 2026-05-05, a cross-study GPU sentinel audit showed that Evo2 treats
  lowercase and uppercase DNA as distinct tokenizer symbols. Lowercase native
  RegulonDB promoter records therefore invalidate any rank, projection, or
  nearest-neighbor diagnostics derived from those model inputs.
- RegulonDB USR and Construct DNA model inputs are now canonicalized to
  uppercase A/C/G/T before Evo2 adapter calls and feature-key construction.
  Existing `_derived/infer` sidecars generated from lowercase records must be
  regenerated, not reused.

### Current Datasets

- Native promoter source: `usr_regulondb_native_promoters` (`local validated`, generated/untracked)
- Native/full 7B vector/scalar sidecars: `infer_regulondb_native_promoter_views_7b` (`local complete`)
- Core60 view: `usr_regulondb_native_promoter_core60` (`local validated`, generated/untracked)
- Core60 7B vector/scalar sidecars: `infer_regulondb_native_promoter_core60_views_7b` (`local complete`)
- Native/full plus in-context TSS-upstream core60 sidecars:
  `config.sequence_views.native_full_plus_tss_upstream_core60.evo2_7b.yaml`
  (`local complete`, additive dogfood lane, not part of the default fill quota)

### Current Phase

- Declared phase: `latentdna_native_audit`
- Source export status: local Cruncher superset export validated; export artifacts are not checked in
- USR dataset status: `src/dnadesign/usr/datasets/usr_regulondb_native_promoters` is materialized locally and passes strict USR validation as of 2026-04-29
- Sequence-view status: write mode now emits one `source_record` sequence view per retained native promoter sequence, plus mutable view semantics for `source_family`, `selection_basis`, `view_collections`, and `role_tags`
- Preferred first infer family: `evo2_7b`
- Current batch ergonomics: `ops runbook fill-infer` inspects the checked-in
  Infer runbooks, skips complete sequence-view vector/scalar lanes, and now
  plans zero RegulonDB GPU submissions from this checkout.

### Source Model

Durable provenance, inclusion rules, and TSS/core60 semantics live in
`../contexts/source-model/README.md`. This status note keeps only current
state, row counts, downstream posture, and next actions.

### Current Row Counts

- `usr_regulondb_native_promoters`: 3,182 base rows (`local validated`, untracked)
- `usr_regulondb_native_promoters/_views/sequence_views.parquet`: 3,182 `source_record` views
- `usr_regulondb_native_promoters/_views/view_semantics.parquet`: 3,182 mutable semantics rows
- `infer_regulondb_native_promoter_views_7b`: 3,182 source-record
  `seq_mean` views complete; 6,364 vector keys and 6,364 scalar keys are
  reusable from canonical native `_derived/infer` sidecars
- `usr_regulondb_native_promoter_core60`: 3,181 canonical 60 bp sequence rows plus 3,182 `analysis_window` sequence views (`local validated`, generated/untracked)
- `usr_regulondb_native_promoters/_derived/infer`: 12,728 feature alias rows,
  12,728 feature vector rows, 6,364 scalar alias rows, and 6,364 scalar rows
  after the native/full plus in-context core60 dogfood run. The checked-in
  native/full runbook quota reuses the 6,364 `seq_mean` vector rows and 6,364
  scalar rows from this sidecar set.
- `usr_regulondb_native_promoter_core60/_derived/infer`: 6,364 feature alias
  rows and 6,364 scalar alias rows. The payload sidecars have 6,362 vector
  rows and 6,362 scalar rows because one duplicate 60 bp sequence is reused by
  two sequence views; alias rows preserve the view-level identities.

### Evidence

Probe evidence, fidelity checks, and end-to-end readiness evidence live in
`evidence/readiness-audit.md`.

### Current Downstream Posture

- Construct: materialized `usr_regulondb_native_promoter_core60` through
  `native_tss_upstream_core60` on 2026-04-29. The run wrote 3,182 sequence-view
  rows and 3,181 canonical 60 bp sequence rows; the difference is expected USR
  sequence deduplication for duplicate derived windows.
- Infer: the standard native source-record `seq_mean` and derived core60
  `core60_mean` Evo2 7B lanes are locally complete. Both lanes request
  intermediate block means, output-layer means, mean-per-token log likelihoods,
  and total log likelihoods. An additive dogfood config also extracts
  `core60_mean` over `[0,60)` from the full 81 bp native context in the same
  forward pass group as native `seq_mean`.
- Notify: native and core60 Infer event-path/profile smoke checks succeeded
  during local dogfood. Watcher cursors consumed the terminal Infer events with
  no remaining spool files.
- Ops: `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel`
  now discovers the native/full and core60 Infer runbooks, blocks stale or
  missing-product lanes, and plans only lanes with missing vectors/scalars. The
  2026-04-30 local plan marks both checked-in 7B lanes `skip_complete`, with
  zero blocked lanes, zero missing products, zero missing vectors, and zero
  missing scalars.
- LatentDNA: configured with a workspace and study binding. The workspace
  validates against the completed native/full and core60 Infer sidecars, and
  the local downstream snapshot now reports the primary decision deliverables
  `representation_health_summary`, `native_core60_shift_summary`, and
  `sigma_factor_structure_summary` as current.
  In LatentDNA prose, native `seq_mean` means a sequence-position mean over
  Evo2 token states from the emitted native 81 bp source-record sequence, and
  core60 mean means the same operation over the derived 60 bp TSS-upstream
  analysis window. Because Evo2 is causal/autoregressive, these pooled token
  states are prefix-conditioned in the emitted forward orientation; they should
  not be described as bidirectional promoter encodings.
- Cluster: submit-ready runbooks exist, but no SCC submission is currently
  needed for the local RegulonDB 7B standard lanes because sidecars are
  complete.
- OPAL: not configured.

### Next Actions

- Review or sync the local generated USR datasets and Infer sidecars through the
  normal USR data sync path. Do not add generated dataset roots to git.
- Do not submit another standard RegulonDB Infer batch from this checkout unless
  completion inventory reports missing or stale vectors/scalars. The current
  handoff command is `uv run ops runbook fill-infer --study-dir
  docs/studies/regulondb_native_promoter_panel --repo-root . --plan-only`.
- Decide whether the additive full-native-context `[0,60)` pooled core60 values
  should become part of the official study quota. If yes, promote the additive
  config into a checked-in runbook; until then it remains dogfood evidence and
  a reusable config, not a default batch lane.
- Keep the LatentDNA snapshot and notebook controls refreshed after any new
  sidecar, plot, or deliverable run.
- Keep live RegulonDB 14.5, sigmulon, HT, prediction, and EcoCyc strata as explicit future reconciliation work rather than silently widening the current base table.
