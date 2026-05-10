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
- `docs/studies/retron_hairpin_design/pipeline.yaml`: the exact command
  groups and native-agent bootstrap support when machine-readable detail is the
  real need
- `docs/studies/retron_hairpin_design/ops.study.yaml`: lifecycle order,
  artifacts, execution surfaces, and preflight grouping
- `docs/studies/retron_hairpin_design/campaign.yaml`: tracked status and
  preflight procedure bundle

## Repo-local skill surface

- `.agents/skills/retron-hairpin-study/SKILL.md`: study-specific shortcut that
  recovers the cap/shortening and base-junction context without rebuilding it by
  hand

## Tool-owned detail

- `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md` owns the
  released-product lane behavior.
- `src/dnadesign/cruncher/src/scar_nick/` and
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/` own current scar-nick code
  and workspace behavior.
- `src/dnadesign/cruncher/docs/guides/yiu_workflow.md` owns the YIU contract.
- `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
  owns the retron/P4 framing note.

## Router rule

When the next question needs exact commands or the next human step, use the
study route map first.
When the next question needs machine-readable command groups or bootstrap
metadata, open `pipeline.yaml`.
When the next question needs harness or contract hardening, leave the study
surface and pair with the owning companion skill.
