---
doc_id: study-retron-hairpin-design-routes
surface: study-route-map
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-05-18
entrypoint: self
status_surface: studies.retron-hairpin-design.status
preflight_surface: studies.retron-hairpin-design.preflight
---

## Retron Hairpin Design Effort Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Retron MSD product work starts from the user's provided parts and desired
output, not from study phase. Status/preflight surfaces are only for explicit
progress or blocker questions.

### Navigation Header

| Need | Surface |
| --- | --- |
| Primary route | this page |
| Status | `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json` |
| Preflight | `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json` |
| Machine-readable contract | `../operations/ops.study.yaml` |

### Quick Route

| Need | Open |
| --- | --- |
| MSD label, complete parts, design catalog, GenBank, plots, or Reader join | [MSD design references](references/msd-design-references.md) |
| Persistent hypotheses, effect tags, design-set membership, or compiler/materialization provenance | [Experimental workbench](../workbench/README.md) |
| Missing cap or shortening geometry | [Released-product Snapback](product/released-product-snapback.md) |
| Missing left/right base feasibility, terminal-nick route, nickase, or `S3/S2/S1/S0` profile | [Scar-nick base-junction](product/scar-nick-base-junction.md) |
| Whole-product sequence composition boundary | [Linear ssDNA composition](composition/linear-ssdna-composition.md) |
| Mismatch/boundary contrast only | [YIU boundary check](quality/yiu-boundary-check.md) |
| Explicit status/history question | `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json` |
| Explicit blocker/readiness question | `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json` |

Repo-local study shortcut: `.agents/skills/retron-hairpin-study/SKILL.md`.

Pair with `harness-engineering` for study-surface hardening and
`code-change-discipline` for boundary or contract changes.

### Routing Contract

- If the request supplies an MSD label or explicit parts, start from the
  compiler/product route. Do not run study status or preflight first.
- If the request asks where the persistent experimental meaning lives, start
  from `workbench/README.md`, then the relevant design set or run record.
- If the request needs compiler registry or convenience label inputs, open
  `../compiler/README.md` before the YAML/TXT records.
- If a part is missing, route to the smallest primitive owner: Snapback for
  cap/shortening, scar-nick for base-junction feasibility, or YIU for contrast
  rendering only.
- Open `../operations/runtime/command-groups/pipeline.yaml` only when the task needs
  machine-readable command-group or automation bootstrap metadata.
- Open `../operations/ops.study.yaml` only when the task needs lifecycle or
  preflight declarations.

### Boundary Shorthand

- `released-product Snapback` means the BspQI-pinned dual-enzyme precursor lane
  where final geometry is evaluated on retained active top and bottom products
  and rebased so the nick boundary is origin `0` in final-geometry space.
- `preserved-site Snapback` means the older one-enzyme lane and stays a separate
  contract.
- `scar-nick` means the base-junction route for Type IIS retained scars plus
  terminal nick processing through the `scar_nick` subpackage.
- `YIU` means mismatch-centric payload rendering over a fixed 4 nt internal
  window; it is not the shortening topology engine here.
- `retron context` means biological framing from checked-in audit notes, not
  scoring hooks or implicit solver relaxations.

Keep this page as a one-hop route map. Move command-heavy or rationale-heavy
detail into `routes/`, `contexts/`, `compiler/`, or `workbench/`.
