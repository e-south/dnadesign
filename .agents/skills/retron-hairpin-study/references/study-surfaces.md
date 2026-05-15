# Study Surfaces

Keep the retron hairpin study boundaries explicit. The checked-in study id is
`retron_hairpin_design`, and the route map separates cap/shortening work from
base-junction scar-nick work.

## Checked-in study surfaces

- `docs/studies/retron_hairpin_design/status.md`: the short factual study
  note and entrypoint into the next route
- `docs/studies/retron_hairpin_design/routes.md`: the canonical
  study-owned post-probe handoff for released-product Snapback, scar-nick, and
  the YIU contrast lane
- `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`: the
  base-junction context for B26/B43 profile logic, strict terminal nick policy,
  retained scar families, and scar-nick schema implications
- `docs/studies/retron_hairpin_design/linear-ssdna-composition.md`: the
  study-owned handoff for whole-product multicopy ssDNA composition, including
  scar-nick projection boundaries
- `docs/studies/retron_hairpin_design/msd_design_registry.yaml`: study-owned
  registry metadata for payloads, caps, construct route notes, nickase, and
  nick orientation used by the MSD design-reference compiler
- `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`: current
  study-selected lab-facing MSD labels for `msd_design_catalog_v1` dogfood
- `docs/studies/retron_hairpin_design/pipeline.yaml`: the exact command
  groups and automation bootstrap support when machine-readable detail is the
  real need
- `docs/studies/retron_hairpin_design/ops.study.yaml`: nonsequential track map,
  artifacts, execution surfaces, and preflight grouping
- `docs/studies/retron_hairpin_design/campaign.yaml`: tracked status and
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
- `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
  owns the retron/P4 framing note.
- `docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md` owns
  the generic Construct/folding/BaseRender/USR dev spec.
- `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`
  owns the completed implementation checklist and validation evidence.
- `docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md`
  owns remaining USR/source-ref/module-split follow-up work.
- `src/dnadesign/studies/retron_hairpin_design/compiler.py` owns the study-local
  MSD reference/catalog compiler API, shallow output-bundle layout, and
  single-unit sequence artifact generation.
- `src/dnadesign/studies/retron_hairpin_design/compiler_spec.py` owns typed
  `retron_msd_compiler_spec_v1` parsing, explicit part normalization, and
  public primitive-source selector checks.
- `src/dnadesign/studies/retron_hairpin_design/cli.py` is the thin Typer
  command adapter for `msd_design_reference_v1` / `msd_design_catalog_v1`
  records plus the `materialize` GenBank/PNG route.
- `dnadesign.cruncher.snapback` and `dnadesign.cruncher.scar_nick` expose the
  public primitive-export APIs used by compiler specs; study code must not
  import Cruncher nested `src` modules directly.

## Router rule

When the next question needs exact commands or the next human step, use the
study route map first.
When the next question says "continue the dev spec" or concerns multicopy
linear ssDNA assembly, open `linear-ssdna-composition.md`, then the dev spec,
then the completed implementation record and current follow-up plan.
When the next question starts from an MSD shorthand ID or Reader-facing design
reference, use the Study route for MSD design references in `routes.md`, then
`references/msd-design-references.md`.
When the next question needs machine-readable command groups or bootstrap
metadata, open `pipeline.yaml`.
When the next question needs harness or contract hardening, leave the study
surface and pair with the owning companion skill.
