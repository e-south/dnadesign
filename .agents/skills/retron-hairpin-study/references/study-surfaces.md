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
- `docs/studies/retron_hairpin_design/workbench/provenance/`: compiler-run and
  materialization records that cite workbench design sets without storing bulky
  generated artifacts
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
  records plus the `materialize` GenBank/native-structure-PNG/review-PNG
  route.
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
When the next question needs machine-readable command groups or bootstrap
metadata, open `operations/runtime/command-groups/pipeline.yaml`.
When the next question needs harness or contract hardening, leave the study
surface and pair with the owning companion skill.
