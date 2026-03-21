## Promoter Study Registry

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-21

Use this directory when the question is not just "how do promoter-study
workflows operate?" but "which real promoter study is active right now?"

Read `docs/studies/promoter/index.yaml` first for current-status checks. This
README explains the layout; it is not the selector.
The active study may still be in source assembly or feature generation, but the
record must still say clearly what already exists and what is still planned.
`index.yaml` selects the active study, the matching study directory holds the
record, and the promoter-study status contract explains how to refresh it.

### Required files

```text
docs/studies/promoter/
  README.md
  index.yaml
  <study-id>/
    campaign.yaml
    datasets.yaml
    status.md
    audits/
```

- `index.yaml` declares which study is active and which checked-in study
  directories are valid candidates.
- `<study-id>/campaign.yaml` tracks procedure progress and concrete artifact
  paths.
- `<study-id>/datasets.yaml` keeps the affiliated-dataset registry, including
  sync posture for shared roots and workspace-local export roots.
- `<study-id>/status.md` records the human-readable current state, row targets,
  infer slices, rollback commands, and next actions.

### Discovery rules

- If `active_study` is `null`, there is no live promoter study record yet.
- If `active_study` names a study id, that entry must exist in `studies:` and
  the corresponding `<study-id>/` directory must contain `campaign.yaml`,
  `datasets.yaml`, and `status.md`.
- If `active_study` and `studies:` disagree, fail visibly and fix the registry
  before asking agents for current-study status.

### Related docs

- [Study records index](../README.md)
- [Promoter study status contract](../../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md)
- [Promoter study index template](../../templates/promoter-study-index.yaml)
