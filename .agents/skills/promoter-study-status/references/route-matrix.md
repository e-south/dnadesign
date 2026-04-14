# Route Matrix

Use this matrix when the user intent is adjacent to the status surface but not
identical to it.

| User question | Primary surface | Why |
| --- | --- | --- |
| Where is the study now? | `uv run ops progress show usr.data-plane.promoter-study-status --json` | Cheap repo-backed snapshot of the checked-in record. |
| What blocks execution here? | `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json` | Command-level readiness on the current host. |
| Which tool or doc should I open next? | `docs/studies/<study-id>/routes.md` | Study-owned one-hop handoff by owner surface. |
| Which study files are authoritative? | `docs/studies/<study-id>/status.md`, `ops.study.yaml`, `pipeline.yaml`, `datasets.yaml` | Record plane stays checked in and explicit. |
| Which dataset sync posture is current? | `datasets.yaml` plus `usr.data-plane.hpc-sync` evidence | Sync posture belongs to the dataset registry, not to this router. |
| Is the study still source-phase or already downstream? | Snapshot plus `status.md` | Use record-backed `source/handoff mode` language until a canonical feature dataset exists. |

Status-first routing boundary:

- Use `promoter-study-status` for record-backed summary.
- Use `promoter-study-preflight` for blockers and default notify-enabled Infer
  presets.
- Use `routes.md` for DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL
  handoff.
