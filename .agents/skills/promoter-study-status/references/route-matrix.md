# Route Matrix

Use this matrix from a blank thread or when the question is close to the status
surface but belongs somewhere else.

| User question | Primary surface | Why |
| --- | --- | --- |
| I just started a fresh thread or the skill is not visible. | `docs/studies/README.md`, then `uv run ops progress show usr.data-plane.promoter-study-status --json` | Project-scope skill discovery is optional; the direct command path is canonical. |
| Where is the study now? | `uv run ops progress show usr.data-plane.promoter-study-status --json` | Cheap repo-backed snapshot of the study record. |
| Which DenseGen plots, LatentDNA deliverables, notebooks, or Cluster artifact paths are available? | `uv run ops progress show usr.data-plane.promoter-study-status --json`, then inspect `evidence.analysis_surfaces` | The status snapshot now carries one route inventory for exploratory-analysis discovery without turning the skill into a tool-local walkthrough. |
| What blocks execution here? | `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json` | Command-level readiness on the current host. |
| Which tool or doc should I open next? | `docs/studies/<study-id>/routes.md` | Study-owned one-hop handoff by owner surface. |
| Which study files are authoritative? | `campaign.yaml`, `datasets.yaml`, `status.md`, `ops.study.yaml`, plus `routes.md` and `pipeline.yaml` when present | The checked-in record stays authoritative. |
| Which dataset sync posture is current? | `datasets.yaml` plus `usr.data-plane.hpc-sync` evidence | Sync posture belongs to the dataset registry, not to this router. |
| Is the study still source-phase or already downstream? | Snapshot plus `status.md` | Use record-backed `source/handoff mode` language until a canonical feature dataset exists. |
| The study record is missing or inconsistent. | `docs/studies/README.md` plus `promoter-study-status-contract.md` | Fail visibly, repair the selector or record, then rerun status. |

Status-first routing boundary:

- Use `docs/studies/README.md` as the blank-thread bootstrap when project-scope
  skill discovery is unavailable.
- Use `promoter-study-status` for record-backed summary.
- Use `promoter-study-preflight` for blockers and default notify-enabled Infer
  presets.
- Use `routes.md` for DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL
  handoff.
