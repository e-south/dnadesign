# Reporter-response window meta-study

This package owns retrospective, provisional selection of one standard
RT-lnRNA reporter-response time window for descriptive comparison. It consumes
only validated `ReporterResponseProfile` instances (or payloads first parsed by
the authoritative `profile_from_dict` API). It never reads assay workbooks,
notebook state, images, legacy condition tables, or unverified dataframes. The
protocol is fixed before regeneration, but the current cohort was not collected
as a prospective validation study.

The public surface is intentionally small:

- `contracts/` owns the fixed protocol and typed `selected|blocked` decision.
- `evaluation/readiness.py` validates bridge receipts and readiness-only
  blocked decisions.
- `evaluation/selection.py` verifies comparable profiles, evaluates the primary
  500 uM cohort, and applies lexicographic selection plus leave-one-acquisition-
  out stability. The 5/50 uM cohort remains sensitivity evidence only.
- `evaluation/evidence.py` closes attempts, profile coordinates, Reader source
  identity, and publication evidence digests.
- `sensitivity.py` emits digest-bound endpoint, alternate-window, and optional
  dose-cohort summaries that are structurally non-selectable.
- `acquisition_projection.py` keeps every acquisition profile immutable and projects the selected
  reduction across unique Reader acquisition IDs. Candidate windows remain
  selection evidence instead of becoming parallel downstream outputs.
  Experiment, plate, sheet, well, and position remain acquisition provenance;
  biological-replicate identity comes only from an explicit source field.
- `sensitivity_coverage.py` binds every ready attempt to the exact
  subject-by-reduction profile-or-omission Cartesian ledger.
- `operator.py` owns the single live regeneration path. Its status check
  requires exact parity between that regeneration and the source-controlled
  decision, compact coverage receipts, and sibling sensitivity summaries; an
  internally valid but stale profile or sensitivity digest cannot pass.
- `publication.py` creates one deterministic meta-study envelope in a new
  directory. Every bundle contains `manifest.json`, `report.md`, and
  `sensitivity.json`; every evaluated bundle, whether selected or
  scientifically blocked, additionally contains primary `evidence.json`.
  A selected bundle also contains `acquisition.json`, which is restricted to the
  selected reduction and rederived from those exact profiles during offline
  verification.
  Only a readiness-only blocked bundle is evidence-free. Verification
  independently repeats the primary decision and sensitivity evaluation from
  bundled profile/audit projections before atomic installation. Existing
  publications are immutable.

Offline bundle verification proves that typed profile content, audit
assertions, attempt receipts, evaluations, and selection agree. Its content
projection deliberately cannot mint Reader source-closure or canonical-audit
closure. Authenticity and freshness of the complete bundle still depend on
checking its recorded Reader identities and digests against their external
owners; digest-consistent bundled audit widths are not a substitute for live
raw-observation rederivation.

Within-acquisition observation range and observation quality share one derivation-
closed, self-digesting audit artifact bound to the complete canonical profile
and the pinned condition ontology. The descriptive v3 profile does not treat
within-acquisition dispersion as biological-replicate uncertainty. Reader
experiment, plate, sheet, well, and position remain acquisition provenance.
Only a declared replicate field supplies biological-replicate identity; absent
that field, identity is unknown. The selected acquisition projection reports
median and leave-one-acquisition-out values without confidence intervals.
Range above the endpoint reference remains an explicit limitation; a candidate
fails closed when no required observations exist or overflow/clipping is
reported.

Endpoint reductions, alternate 2 h/6 h centered windows, and optional 5/50 uM
cohorts produce typed sensitivity summaries that cannot enter primary
selection. The lean source-controlled state stores compact exact-identity and
coverage-digest receipts beside, not inside, `MetastudyDecision`; the immutable
publication carries the full offline-verifiable profiles and Cartesian
coverage ledgers. No weighted score, optimization label, or objective is
implemented here.

`materialize/` derives profiles and audit artifacts from one source-closed
Reader dataframe record, its source-closed subject bindings, and the exact
pinned ontology and observation policy. Its service owns orchestration and
blocked receipts, profiles own identity joins and descriptive uncertainty, and
temporal owns trace selection and condition summaries. Evidence insufficiency
returns typed blockers; only those derivation-closed outputs can enter selection.

Run the focused contract:

```bash
.venv/bin/pytest -q \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy
```
