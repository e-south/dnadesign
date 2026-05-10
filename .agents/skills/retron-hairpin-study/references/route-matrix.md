# Route Matrix

Use this matrix when the question starts near the retron hairpin study but
quickly turns into a different surface.

| User question | Primary surface | Why |
| --- | --- | --- |
| Fresh thread or skill not visible. | `docs/studies/README.md`, then `cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json` | The direct command path is canonical even when project-scope skill discovery is absent. |
| Where is the hairpin study now? | `cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json` | Cheap record-backed snapshot of the pinned checked-in study id. |
| What blocks the next study step on this host? | `cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json` | Command-level readiness for the current study phase. |
| Which command group or workspace should I open next? | `docs/studies/retron_hairpin_design/routes.md` | The study owns the human handoff; open `pipeline.yaml` only when machine-readable command-group or bootstrap metadata is the real need. |
| Which surface owns cap/shortening geometry? | `routes.md`, then the released-product Snapback route | Snapback owns the cap/shortening lane and 0/3/3 retained-active route. |
| Which surface owns base-junction or B26/B43 scar logic? | `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`, then the scar-nick route in `routes.md` | Scar-nick owns Type IIS scar plus terminal nick geometry; the context page owns the strict top/bottom policy snapshot. |
| Is YIU the shortening engine here? | `status.md`, `routes.md`, and `yiu_workflow.md` | The checked-in study keeps YIU contrast-only and mismatch-centric. |
| I need to harden the study status, preflight, or automation bootstrap. | `retron-hairpin-study` plus `harness-engineering` | This is a harness problem, rather than a prose-only edit. |
| I need to change lane boundaries, ontologies, or fail-fast behavior. | `retron-hairpin-study` plus `code-change-discipline` | This is a boundary and contract problem. |

Status-first routing boundary:

- Use study status for the record-backed answer.
- Use study preflight for blockers.
- Use `routes.md` for the next command surface.
- Use `pipeline.yaml` only for machine-readable command-group or bootstrap support.
- Do not reconstruct the study plan from generic Cruncher docs alone.
