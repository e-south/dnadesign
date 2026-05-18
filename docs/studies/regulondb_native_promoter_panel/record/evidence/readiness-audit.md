## RegulonDB Readiness Evidence

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-04

This file holds durable evidence for the checked-in status note. Keep current
phase and next actions in `../status.md`.

### Read-Only Probe Evidence

Latest local source probe on 2026-04-27 produced a temporary Cruncher superset
export outside the repo. It parsed base-row-capable curated PromoterSet sources
from sibling `dnadesign-data` and inventoried supplemental source strata without
turning them into sequence rows:

- Normalized curated source records: 7,914 from RegulonDB 13 and RegulonDB 11
  PromoterSet files.
- Skipped curated source rows with row-level provenance: 184 total, all
  `missing_sequence`.
- Strict USR base rows after sigma-required canonical sequence deduplication:
  3,182.
- Retained source rows: 6,629.
- Duplicate sequence collapses among retained rows: 3,447.
- Sequence-bearing source rows excluded for missing sigma: 1,285 source rows,
  representing 645 sequence groups.
- Same-release promoter-id sequence conflicts: 0.
- Supplemental strata recorded but deferred from base-row creation: RegulonDB
  13 sigmulon, RegulonDB 11 RACE/454, RegulonDB 11 prediction rows, and EcoCyc
  28 promoter windows.

### Superset Fidelity Checks

- Duplicate canonical base sequences: 0.
- Invalid non-ACGT base sequences: 0.
- Base sequence length mismatches: 0.
- Required retained `regulondb__*` overlay metadata null counts: none.
- Orphan relation rows with `usr_id`: 0.
- Duplicate relation rows after alias and sigma deduplication: 0 for retained
  relation sidecars.
- Missing sigma annotation in retained base rows: 0.
- Fuzzy promoter-name collision candidates after strict filtering: 16; these
  remain manual-review signals, not automatic duplicate calls.

A bounded live RegulonDB 14.5 GraphQL probe on 2026-04-27 returned 20 promoter
records with 95% sequence coverage, 95% TSS coverage, and 55% sigma coverage.
The live route remains a modern overlay/completeness check rather than the
current completeness base.

### End-To-End Readiness Audit

Record-backed evidence through 2026-05-04:

- The checked-in record reports `latentdna_native_audit` after the native/full
  and core60 Infer sidecars are complete.
- `datasets.yaml` reports `exists=true` and `rows=3182` for
  `usr_regulondb_native_promoters`.
- `uv run usr validate usr_regulondb_native_promoters --strict` passes locally.
- `uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel --project native_tss_upstream_core60 --format json`
  returned `records_total=3182`, `records_written=3182`, and `dry_run=false`.
- `uv run usr validate usr_regulondb_native_promoter_core60 --strict` passes
  locally.
- Native/full and core60 Evo2 7B completion inventory reports zero missing
  vectors and zero missing scalars.
- Native and core60 Infer event-path/profile smoke checks resolved the USR
  `.events.log` files.
- `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel --repo-root . --plan-only`
  reports `lanes_total=2`, `skip_complete_lanes=2`, `runnable_lanes=0`,
  `blocked_lanes=0`, and zero missing or stale vector/scalar products.
- Local Evo2 7B dogfood completed the additive native/full plus in-context
  core60 config in 58.74 seconds wall time with about 18.8 GiB peak GPU memory,
  then completed the derived core60 standard lane in 49.57 seconds wall time
  with about 17.9 GiB peak GPU memory.
- A read-only empirical recompute check sampled 128 native source records and
  128 derived core60 views; fresh outputs matched persisted sidecars exactly
  (`max_abs_diff=0.0`, missing payloads `0`).
- A direct USR congruence check reports native records are all 81 bp, derived
  core60 records/views are all 60 bp, every core60 view has a parent native
  sequence, and `core60_sequence == parent_native_sequence[0:60]` for all
  3,182 views.
- No study-owned OPS preflight provider is registered for
  `regulondb_native_promoter_panel`; use owner-tool validation commands until
  this study owns a concrete provider.
- `MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna validate workspace --workspace regulondb_native_promoter_panel --deep --json`
  returns `status=ok`.
