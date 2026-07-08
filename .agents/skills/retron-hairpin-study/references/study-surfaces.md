# Study Surfaces

Keep the retron hairpin study boundaries explicit. The canonical checked-in study id is
`retron_hairpin_design`, and the route map separates cap/shortening work from
base-junction scar-nick work.

## Checked-in study surfaces

- `docs/studies/retron_hairpin_design/record/status.md`: the short factual study
  note and entrypoint into the next route
- `docs/studies/retron_hairpin_design/routes/README.md`: the one-hop study-owned
  handoff for compiler, workbench, primitive, and status/readiness routes
- `docs/studies/retron_hairpin_design/routes/`: focused owner-surface route
  details for MSD references, released-product Snapback, scar-nick,
  linear-ssDNA composition, and YIU contrast
- `docs/studies/retron_hairpin_design/compiler/`: study-owned compiler
  inputs and normalization metadata
- `docs/studies/retron_hairpin_design/workbench/`: persistent hypotheses,
  effect tags, design-set membership, compiler-run provenance, and
  materialization provenance
- `docs/studies/retron_hairpin_design/workbench/ontology/`: controlled
  direction and effect-tag vocabulary for workbench records
- `docs/studies/retron_hairpin_design/workbench/design_sets/`: authoritative
  persistent design cohorts for experimental meaning
- `docs/studies/retron_hairpin_design/workbench/design_sets/teto_pwm_trim_rescue_v1.yaml`:
  authoritative cargo-shortening cohort for retron26 control, retron43 target,
  and the pES-retron-180 C172/AGTG/CATG/XWMM context
- `docs/studies/retron_hairpin_design/workbench/design_sets/teto_payload_trim_retest_v1.yaml`:
  tetO payload-prior retest cohort for pES-retron-201 through pES-retron-204;
  it keeps the 15 nt and 13 nt retained-span extents from the 195-200 pilot
  while changing the payload prior
- `docs/studies/retron_hairpin_design/workbench/deliverables/`: persistent
  review and handoff deliverable contracts for study hypotheses
- `docs/studies/retron_hairpin_design/workbench/deliverables/teto_pwm_trim_rescue_v1.yaml`:
  tetO trim deliverable plan for PWM triptych, nine sequence review stills,
  sequence montage video, review manifest, nine-row GenBank sequence handoff,
  six-file Benchling import folder, and future Reader outcome overlay routing
- `docs/studies/retron_hairpin_design/workbench/deliverables/teto_payload_trim_retest_v1.yaml`:
  tetO payload-prior retest deliverable plan for pES-retron-201 through
  pES-retron-204, with plan-owned PWM review, sequence montage, handoff, and
  four-file Benchling import expectations
- `docs/studies/retron_hairpin_design/workbench/outputs/`: ignored generated
  output roots for local materialized bundles and review packages; preferred
  tetO trim root is `workbench/outputs/teto_pwm_trim_rescue_v1/`
- `docs/studies/retron_hairpin_design/workbench/provenance/`: compiler-run and
  materialization records that cite workbench design sets without storing bulky
  generated artifacts
- `docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/reader_spop_msd_structure_panel_v1/`:
  active Reader SPOP MSD-region source records, with one-variant GenBank inputs
  under `source_inputs/variants/`, a `variant_sources.yaml` source manifest,
  decomposed variant YAML records, and pairing-segment facts. Retired bulk
  source metadata is provenance only.
- `docs/studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction.md`: the
  base-junction context for B26/B43 profile logic, strict terminal nick policy,
  retained scar families, and scar-nick schema implications
- `docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md`: the
  study-owned handoff for whole-product multicopy ssDNA composition, including
  scar-nick projection boundaries
- `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`: study-owned
  registry metadata for payloads, caps, construct route notes, nickase, and
  nick orientation used by the MSD design-reference compiler
- `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`: concise
  cap source lookup for known `C###` ids and 5'->3' sequences
- `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`:
  convenience lab-facing MSD labels for compiler input; workbench design sets
  are authoritative for persistent cohort meaning
- `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml`:
  full checked-in materialization spec that supplies selected cap/foldback
  segments as explicit 5'->3' sequences
- `docs/studies/retron_hairpin_design/compiler/inputs/teto_pwm_trim_rescue_v1.spec.yaml`:
  nine-design tetO PWM trim compiler spec with literal payload
  sequences, payload-trim metadata, WT Eco1-only variant metadata, and explicit
  cap/stem-base choices
- `docs/studies/retron_hairpin_design/compiler/inputs/teto_payload_trim_retest_v1.spec.yaml`:
  four-design tetO payload-prior retest compiler spec for pES-retron-201
  through pES-retron-204 handoff generation
- `docs/studies/retron_hairpin_design/operations/runtime/command-groups/pipeline.yaml`: the exact command
  groups and automation bootstrap support when machine-readable detail is the
  real need
- `docs/studies/retron_hairpin_design/operations/ops.study.yaml`: nonsequential track map,
  artifacts, execution surfaces, and preflight grouping
- `docs/studies/retron_hairpin_design/record/campaign.yaml`: tracked status and
  preflight procedure bundle

## Repo-local skill surface

- `.agents/skills/retron-hairpin-study/SKILL.md`: study-specific shortcut that
  recovers the cap/shortening and base-junction context without rebuilding it by
  hand
- `.agents/skills/retron-hairpin-study/references/msd-design-references.md`:
  progressive-disclosure reference for the ID-to-catalog route

## Tool-owned detail

- `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md` owns the
  released-product lane behavior.
- `src/dnadesign/cruncher/src/scar_nick/` and
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/` own current scar-nick code
  and workspace behavior.
- `src/dnadesign/cruncher/docs/guides/yiu_workflow.md` owns the YIU contract.
- `src/dnadesign/cruncher/docs/dev/audits/2026-04-19-retron-p4-hairpin-variant.md`
  owns the retron/P4 framing note.
- `docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md` owns
  the generic Construct/folding/BaseRender/USR dev spec.
- `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`
  owns the completed implementation checklist and validation evidence.
- `docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md`
  owns remaining USR/source-ref/module-split follow-up work.
- `src/dnadesign/studies/units/retron_hairpin_design/compiler/references.py` owns
  label-to-reference compilation.
- `src/dnadesign/studies/units/retron_hairpin_design/compiler/catalog_bundle.py` owns
  catalog/reference bundle writing.
- `src/dnadesign/studies/units/retron_hairpin_design/compiler/materialization.py` owns
  Construct-backed sequence-bundle orchestration.
- `src/dnadesign/studies/units/retron_hairpin_design/compiler/exceptions.py` owns
  the fail-fast compiler exception type.
- `src/dnadesign/studies/units/retron_hairpin_design/catalog/compiler_spec.py` owns typed
  `retron_msd_compiler_spec_v1` parsing, explicit part normalization, and
  public primitive-source selector checks.
- `src/dnadesign/studies/units/retron_hairpin_design/catalog/compiler_spec_io.py` owns
  fail-fast compiler-spec file loading, including duplicate JSON/YAML mapping-key
  rejection before typed parsing.
- `src/dnadesign/studies/units/retron_hairpin_design/catalog/specs/` owns small
  typed compiler-spec support models for primitive selectors and optional
  payload-trim/variant-role metadata.
- `src/dnadesign/studies/units/retron_hairpin_design/review_outputs/` owns the
  plan-driven `review-outputs` service: deliverable-plan loading, materialized
  sequence-index validation, PWM logo triptych rendering, sequence evidence
  checks, semantic still rendering, sequence montage rendering, and
  `reviews/review_manifest.json`.
- `src/dnadesign/studies/units/retron_hairpin_design/review_outputs/service.py`
  is the review-output facade used by the CLI and tests.
- Review-output implementation is split by semantic lane:
  `contracts/` parses deliverable plans and writes review manifests, `pwm/`
  renders trim triptychs through public `dnadesign.baserender` APIs,
  `sequence/` validates materialized sequence evidence, `video/` writes stills
  and montage video, and `handoff/` writes sequence-handoff indexes plus
  Benchling import GenBanks. The deliverable contract owns
  `review_variant_ids`, and the Benchling import plan owns
  `assigned_retron_ids`, `source_precedent_ids`, included trim ids, and
  reverse-complement-only orientation. Do not add root-level `pwm_*.py`,
  `sequence_*.py`, or sequence-handoff helper files.
- `src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/` keeps
  compiler tests split by semantic lane: `test_cap_sources.py`,
  `test_msd_ids.py`, `test_cli_lint.py`, `test_cli_compile.py`,
  `test_msd_unit.py`, `test_materialization.py`, and `test_boundaries.py`.
  Typed-spec metadata tests live under
  `tests/compiler/specs/`; shared fixtures live under `tests/support/`.
- `src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/`
  mirrors the review-output semantic lanes: `cli/`, `handoff/`, `package/`,
  `pwm/`, and `video/`. Keep shared review-output fixtures in
  `tests/support/`; do not add broad root-level `tests/review_outputs/test_*.py`
  files.
- `src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/composition_payload.py`
  owns single-unit sequence artifact generation intent as linear-ssDNA
  composition payload construction.
- `src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/output_guards.py` owns
  fail-fast stale-output guards for the shallow output-bundle layout.
- `src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/materialized_outputs.py`
  owns GenBank/plot/manifest artifact publication from Construct output.
- `src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/manifests.py` owns
  catalog, index, manifest, and bundle README writers for that shallow
  output-bundle layout.
- `src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/app.py` is the thin Typer
  command service for `msd_design_reference_v1` / `msd_design_catalog_v1`
  records plus the `materialize`, `review-outputs`, and MSD-region source-ingest routes.
- `src/dnadesign/studies/units/retron_hairpin_design/source_ingest/` owns
  GenBank source normalization, per-variant source-dir ingest, annotation notes,
  review warnings, derived pairing segments, payload-binding-site semantics, and
  bundle writing for MSD-region records. Start at `msd_region_genbank.py` for
  the public API, then route by concern: `genbank_bundle.py` parses GenBank
  inputs, `variant_sources.py` owns per-variant source manifests,
  `record_normalization.py` assembles normalized records, `annotation_review.py`
  owns benign boundary notes, `pairing_segments.py` derives stem/payload pairing
  facts, `source_ingest/payload_catalog.py`,
  `source_ingest/payload_motifs.py`, and `source_ingest/payload_sites.py` own
  binding-site ontology, `comparison.py` compares older materialized outputs,
  and `bundle_writer.py` writes generated record bundles.
- `src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/review_outputs.py`
  owns the focused Typer command handler for review package generation.
- `dnadesign.cruncher.snapback` and `dnadesign.cruncher.scar_nick` expose the
  public primitive-export APIs used by compiler specs; study code must not
  import Cruncher nested `src` modules directly.

## Router rule

When the next question needs exact commands or the next human step, use the
study route map first.
When the next question says "continue the dev spec" or concerns multicopy
linear ssDNA assembly, open `contexts/composition/linear-ssdna-composition.md`, then the
dev spec, then the completed implementation record and current follow-up plan.
When the next question starts from an MSD shorthand ID or Reader-facing design
reference, use the Study route for MSD design references in `routes/README.md`, then
`routes/compiler/msd-design-references.md`, then `references/msd-design-references.md`.
When the next question asks why variants were selected or how the cohort maps to
hypotheses/effects, open `workbench/README.md` and the relevant design set.
When the next question asks where PWM plots, sequence stills, videos, GenBank
exports, or future outcome overlays belong for a hypothesis, open
`workbench/deliverables/README.md` and the relevant deliverable plan.
When a materialized review bundle already exists and the next question is to
generate the visual review package, open `workbench/deliverables/` and run
`review-outputs --deliverable-plan <plan.yaml>`.
When the next question needs machine-readable command groups or bootstrap
metadata, open `operations/runtime/command-groups/pipeline.yaml`.
When the next question needs harness or contract hardening, leave the study
surface and pair with the owning companion skill.
