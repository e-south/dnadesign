# Route Matrix

Use this matrix when the question starts near the shortening study but quickly
turns into a different surface.

| User question | Primary surface | Why |
| --- | --- | --- |
| I just started a fresh thread or the skill is not visible. | `docs/studies/README.md`, then `cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json` | The direct command path is canonical even when project-scope skill discovery is absent. |
| Where is the shortening study now? | `cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json` | Cheap record-backed snapshot of the pinned study. |
| What blocks the next shortening step on this host? | `cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json` | Command-level readiness for the current study phase. |
| Which command group or workspace should I open next? | `docs/studies/snapback_shortening_effort/routes.md` | The study owns the human handoff; open `pipeline.yaml` only when machine-readable command-group or bootstrap metadata is the real need. |
| Is YIU the shortening engine here? | `status.md`, `routes.md`, and `yiu_workflow.md` | The checked-in study keeps YIU contrast-only and mismatch-centric. |
| I need to harden the study status, preflight, or native-agent bootstrap. | `snapback-hairpin-study` plus `harness-engineering` | This is a harness problem, not just a prose problem. |
| I need to change lane boundaries or fail-fast behavior. | `snapback-hairpin-study` plus `pragmatic-programming-principles` | This is a boundary and contract problem. |

Status-first routing boundary:

- Use study status for the record-backed answer.
- Use study preflight for blockers.
- Use `routes.md` for the next command surface.
- Use `pipeline.yaml` only for machine-readable command-group or bootstrap support.
- Do not reconstruct the shortening plan from generic Cruncher docs alone.
