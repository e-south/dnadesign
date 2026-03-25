## Cassette Workspace Demo (init-workspace to rendered QA)

**Owner:** dnadesign-maintainers
**Doc kind:** tutorial
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-25
**Applies to:** `uv run cruncher cassette init-workspace|solve` plus `uv run baserender job validate|run`
**Last verified:** 2026-03-25
**Primary artifacts:** `cassette_workspace_manifest.json`, `outputs/cassette_solves/<solve_id>/views/*.jsonl`, `baserender_jobs/*.job.yaml`, and workspace-local `renders/*.pdf`

### Contents
- [Overview](#overview)
- [What this demo teaches](#what-this-demo-teaches)
- [Defined demo location](#defined-demo-location)
- [End-to-end commands](#end-to-end-commands)
- [Inspect outputs](#inspect-outputs)
- [Related docs](#related-docs)

### Overview

This tutorial takes a fresh cassette workspace from scaffold to rendered QA output.

Use it when you need to:

- bootstrap a cassette workspace in the standard Cruncher `workspaces/` root
- run one of the shipped solve profiles without authoring YAML first
- validate that the emitted baserender jobs render in place under the same workspace root

Start here if you do not already have a cassette workspace.

### What this demo teaches

- why `cruncher cassette init-workspace` creates a `runbook-only` workspace instead of a full sampling workspace
- where the shipped fast, balanced, and deep MMR solve profiles live
- how solve outputs stay workspace-scoped under `outputs/cassette_solves/<solve_id>/`
- how the publication flow stays local to the solve bundle: `views/` -> `baserender_jobs/` -> `renders/`

### Defined demo location

This demo uses one fixed workspace name under the checked-in Cruncher workspaces root:

```bash
# Standard checked-in Cruncher workspaces root plus one cassette workspace name.
# Set the shared Cruncher workspaces root used by this tutorial.
WORKSPACES_ROOT=src/dnadesign/cruncher/workspaces
# Pick one cassette workspace name under that root.
DEMO_WORKSPACE=cassette_lab_demo
# Derive the full workspace path used for the remaining commands.
DEMO_ROOT="$WORKSPACES_ROOT/$DEMO_WORKSPACE"
```

Why this path works:

- it places the cassette scaffold alongside the other Cruncher workspaces
- `cruncher workspaces list` reports it as `runbook-only` because the scaffold includes `configs/runbook.yaml`
- it makes the workspace boundary obvious when you inspect `outputs/cassette_solves/...`

If you re-run this tutorial against the same root, use `--force-overwrite` only when the root was created by `cruncher cassette init-workspace`.

### End-to-end commands

Bootstrap the workspace and run the fast profile:

```bash
# Pick one cassette workspace name under the standard Cruncher workspaces root.
# Set the shared Cruncher workspaces root used by this tutorial.
WORKSPACES_ROOT=src/dnadesign/cruncher/workspaces
# Pick one cassette workspace name under that root.
DEMO_WORKSPACE=cassette_lab_demo
# Derive the full workspace path used for the remaining commands.
DEMO_ROOT="$WORKSPACES_ROOT/$DEMO_WORKSPACE"

# Scaffold the runbook-only cassette workspace in the standard root.
uv run cruncher cassette init-workspace "$DEMO_WORKSPACE"
# Optional: confirm it is discoverable next to the other workspaces.
uv run cruncher workspaces list --root "$WORKSPACES_ROOT"
# Enter the scaffold root so the shipped spec paths resolve directly.
cd "$DEMO_ROOT"

# Solve with the smallest shipped search profile first.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_fast.cassette.solve.yaml
```

The scaffold writes:

- `README.md`
- `runbook.md`
- `cassette_workspace_manifest.json`
- `configs/runbook.yaml`
- `configs/cassettes/demo_hairpin_fast.cassette.solve.yaml`
- `configs/cassettes/demo_hairpin_balanced.cassette.solve.yaml`
- `configs/cassettes/demo_hairpin_deep_mmr.cassette.solve.yaml`

After the solve finishes, use the emitted solve-level job to validate and render the duplex contact sheet in place:

```bash
# Confirm the solve-level duplex QA job is valid before rendering.
uv run baserender job validate outputs/cassette_solves/<solve_id>/baserender_jobs/top_hits_duplex.job.yaml
# Render the solve-level duplex QA sheet into the sibling renders/ directory.
uv run baserender job run outputs/cassette_solves/<solve_id>/baserender_jobs/top_hits_duplex.job.yaml
```

Then validate and render one per-hit hairpin figure in place:

```bash
# Confirm the per-hit hairpin QA job is valid before rendering.
uv run baserender job validate outputs/cassette_solves/<solve_id>/hits/hit_001_<solution_id>/baserender_jobs/ssdna_hairpin.job.yaml
# Render one per-hit hairpin QA figure into the sibling renders/ directory.
uv run baserender job run outputs/cassette_solves/<solve_id>/hits/hit_001_<solution_id>/baserender_jobs/ssdna_hairpin.job.yaml
```

Use `solve_report.json`, `solve_status.json`, or `table__hits.csv` to fill in `<solve_id>` and `<solution_id>` after the run.

### Inspect outputs

Everything stays inside the same cassette workspace root:

```text
src/dnadesign/cruncher/workspaces/cassette_lab_demo/
  configs/
    runbook.yaml
    cassettes/
      demo_hairpin_fast.cassette.solve.yaml
  outputs/
    cassette_solves/
      <solve_id>/
        solve_report.json
        table__hits.csv
        views/
          top_hits.linear_duplex.v1.jsonl
          top_hits.ssdna_hairpin.v1.jsonl
        baserender_jobs/
          top_hits_duplex.job.yaml
          top_hits_hairpin.job.yaml
        renders/
          top_hits_duplex_qa_sheet.pdf
          top_hits_hairpin_qa_sheet.pdf
        hits/
          hit_001_<solution_id>/
            explicit/
            views/
            baserender_jobs/
            renders/
```

That scope is intentional:

- Cruncher publishes solve reports, view contracts, and baserender jobs into the cassette workspace.
- BaseRender reads those local job files and writes PDFs back into sibling `renders/` directories.
- `workspaces list` can inventory the scaffold because `configs/runbook.yaml` makes it a `runbook-only` workspace.
- No separate baserender workspace or Cruncher `run_index.json` entry is involved.

### Related docs

- [`../guides/cassette_solve_workflow.md`](../guides/cassette_solve_workflow.md): solve profile details, guardrails, and selection semantics.
- [`../guides/cassette_workflow.md`](../guides/cassette_workflow.md): explicit authored-spec validation and design flow.
- [`../reference/cassette_artifacts.md`](../reference/cassette_artifacts.md): file-by-file artifact layout for explicit and solve runs.
- [`../reference/cli.md`](../reference/cli.md): command-level contracts for `cassette init-workspace`, `solve`, and `catalog init-neb`.
