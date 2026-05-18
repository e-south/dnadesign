# Route Matrix

Use this matrix from a blank thread or when the question is close to the status
surface but belongs somewhere else.

| User question | Primary surface | Why |
| --- | --- | --- |
| I just started a fresh thread or the skill is not visible. | `docs/studies/README.md`, then `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` | Project-scope skill discovery is optional; the direct command path is canonical. |
| Where is the study now? | `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` | Cheap repo-backed snapshot of the study record. |
| Which DenseGen plots, LatentDNA deliverables, notebooks, or Cluster artifact paths are available? | `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`, then inspect `evidence.analysis_surfaces` | The status snapshot now carries one route inventory for exploratory-analysis discovery without turning the skill into a tool-local walkthrough. |
| What blocks execution here? | `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json` | Command-level readiness on the current host. |
| Which tool or doc should I open next? | `docs/studies/<study-id>/routes/README.md` | Study-owned one-hop handoff by owner surface. |
| How do I check OPAL campaign status or open OPAL plots? | `docs/studies/<study-id>/routes/README.md`, then `docs/studies/<study-id>/routes/decision/opal/README.md` and `campaign-commands.md` | The one-hop route selects OPAL; command detail stays in the campaign command subpage instead of expanding this skill into an OPAL walkthrough. |
| What is blocking OPAL batch 0? | `docs/studies/<study-id>/routes/decision/opal/README.md`, then `opal validate` for the stress configs | The shared dense-plan `records.parquet` candidate table is materialized; OPAL is now pre-assay, with labels and campaign state still pending. |
| Where did this OPAL candidate ID come from? | `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml --id <candidate_id>` | Per-ID lineage is study-owned because it joins OPAL records to DenseGen sidecars, Construct views, Infer aliases, and LatentDNA rows. |
| Which study files are authoritative? | `record/campaign.yaml`, `record/datasets.yaml`, `record/status.md`, `operations/ops.study.yaml`, plus `routes/README.md` and `operations/runtime/command-groups/pipeline.yaml` when present | The checked-in record stays authoritative. |
| Which dataset sync posture is current? | `record/datasets.yaml` plus `usr.data-plane.hpc-sync` evidence | Sync posture belongs to the dataset registry, not to this router. |
| Is the study still source-phase or already downstream? | Snapshot plus `record/status.md` | Use record-backed `source/handoff mode` language until a canonical feature dataset exists. |
| The study record is missing or inconsistent. | `docs/studies/README.md` plus `operations/catalog/contracts/status.md` | Fail visibly, repair the selector or record, then rerun status. |

Status-first routing boundary:

- Use `docs/studies/README.md` as the blank-thread bootstrap when project-scope
  skill discovery is unavailable.
- Use `stress-ethanol-cipro-growth-status` for record-backed summary.
- Use `stress-ethanol-cipro-growth-preflight` for blockers and default notify-enabled Infer
  presets.
- Use `routes/README.md` for DenseGen, Construct, Infer, LatentDNA, Cluster, and OPAL
  handoff. Use `routes/decision/opal/README.md` or `routes/analysis/latentdna.md` only after the
  one-hop map selects that owner surface.
