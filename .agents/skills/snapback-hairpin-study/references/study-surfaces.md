# Study Surfaces

Keep the shortening study boundaries explicit.

## Checked-in study surfaces

- `docs/studies/snapback_shortening_effort/status.md`: the short factual study
  note and decision ladder
- `docs/studies/snapback_shortening_effort/routes.md`: the study-owned route
  map for released-product Snapback and the YIU contrast lane
- `docs/studies/snapback_shortening_effort/pipeline.yaml`: the exact command
  groups and native-agent bootstrap context
- `docs/studies/snapback_shortening_effort/ops.study.yaml`: lifecycle order,
  artifacts, execution surfaces, and preflight grouping
- `docs/studies/snapback_shortening_effort/campaign.yaml`: tracked status and
  preflight procedure bundle

## Repo-local skill surface

- `.agents/skills/snapback-hairpin-study/SKILL.md`: study-specific shortcut
  that recovers the shortening context without rebuilding it by hand

## Tool-owned detail

- `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md` owns the
  released-product lane behavior.
- `src/dnadesign/cruncher/docs/guides/yiu_workflow.md` owns the YIU contract.
- `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
  owns the retron/P4 framing note.

## Router rule

When the next question needs exact commands, use the study route map.
When the next question needs harness or contract hardening, leave the study
surface and pair with the owning companion skill.
