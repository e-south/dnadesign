## Runbook Catalog

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-19

Use this page when you want a concise inventory of authoritative runbooks, workflows, and owner-local tool entrypoints without guessing which tool owns the next deep procedure. This page is a discovery surface only. It does not replace the owner-local runbook or workflow that each entry links to, and it does not replace the owner-local docs entrypoint for each tool either.

If you prefer terminal discovery, use `uv run ops catalog list` for the grouped inventory, `uv run ops catalog list --plane data-plane --query infer` to narrow the inventory by intent, `uv run ops catalog list --section tool-sources` when you only want owner-local entrypoints, `uv run ops catalog list --section tool-sources --query "promoter feature matrix"` when you want a broad owner-local search around the Evo2/promoter downstream path, `uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix` when you want typed related tool docs around one registered procedure, `uv run ops catalog list --related-to usr.data-plane.promoter-feature-matrix` for typed related procedures around one cross-tool path, `uv run ops catalog show <registry-id>` for one registered cross-tool procedure with owner-boundary, owner docs, typed relation detail, typed related tool docs, exact deep docs when declared, required progress inputs, and next shell commands, `uv run ops progress show <registry-id> ...` for one registered progress surface, `uv run ops progress scaffold <registry-id> ...` to emit a manifest skeleton with the right required inputs, `uv run ops progress scaffold --related-to <registry-id>` to turn one related-procedure view into an explicit campaign starting point, and `uv run ops progress campaign --manifest <manifest.yaml>` for an explicit multi-step summary.

### Discovery shortcuts

- `uv run ops catalog list`: full inventory of authoritative cross-tool procedures plus tool-local runbook sources.
- `uv run ops catalog list --section tool-sources`: only tool-local docs entrypoints when you already know you want a tool-owned demo, tutorial, or runbook family.
- `uv run ops catalog list --section tool-sources --query "promoter feature matrix"`: owner-local docs entrypoints for the USR, Infer, Cluster, and OPAL surfaces around the promoter/Evo2 path.
- `uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix`: typed related tool docs for the DenseGen -> construct -> infer -> cluster/OPAL path around one registered procedure.
- `uv run ops catalog list --related-to usr.data-plane.promoter-feature-matrix`: procedures related through typed owner-local registry metadata, not inferred from prose link placement.
- `uv run ops catalog show usr.data-plane.promoter-feature-matrix`: one procedure with owner-boundary, owner docs, typed related tool docs, exact deep docs when declared, entry/exit artifact, typed relation detail, required progress inputs, and next shell commands for progress interrogation or relation-based scaffolding.
- `uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>`: one registered progress surface with an explicit artifact contract.
- `uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix`: emit an explicit multi-step manifest skeleton with required field placeholders derived from the shared registry.
- `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix`: expand one registered procedure into a starting manifest with the named procedure first and its typed related procedures after it.
- `uv run ops progress campaign --manifest <manifest.yaml>`: read-only summary for an explicit multi-step campaign without inventing a second registry.

### Authoritative cross-tool procedures

This table is generated from owner-local `*.registry.yaml` metadata sidecars. Edit those files instead of hand-editing rows here.

| Registry id | Procedure | Type | Plane | Execution kind | Progress kind | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| `ops.control-plane.orchestration` | [Orchestration runbooks](../operations/orchestration-runbooks.md) | `runbook` | `control-plane` | `executable` | `ops-audit-json` | Deterministic control-plane runbook contract for DenseGen or Infer batch submit flows with optional Notify chaining. |
| `usr.data-plane.hpc-sync` | [USR HPC Sync Flow](../../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | `runbook` | `data-plane` | `iterative` | `usr-sync-audit` | Keep one USR dataset synchronized between HPC and local analysis with explicit diff, pull, and push verification. |
| `usr.data-plane.chained-densegen-infer-sync` | [Chained DenseGen and Infer Sync Runbook](../../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md) | `runbook` | `data-plane` | `iterative` | `usr-sync-audit` | Coordinate DenseGen-on-HPC and Infer-local writes against one USR dataset with explicit sync checkpoints. |
| `usr.data-plane.multi-source-source-of-truth` | [Multi-Source Source-of-Truth Assembly](../../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Merge multiple USR-backed sources, preserve explicit carry, and hand one construct-backed downstream dataset to Infer and Notify. |
| `usr.data-plane.construct-infer-source-of-truth` | [Construct -> USR -> Infer Source-of-Truth Runbook](../../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Realize construct outputs into one shared USR dataset and use that dataset as the durable Infer handoff. |
| `usr.data-plane.promoter-feature-matrix` | [Promoter Characterization Feature Matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Build one infer-annotated feature matrix from mixed promoter sources before branching into Cluster or OPAL. |
| `cluster.downstream.exploratory-clustering` | [Exploratory clustering workflow](../../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) | `workflow` | `downstream-tool` | `exploratory` | `cluster-run-index` | Explore one chosen feature definition through clustering, UMAP, and downstream summaries. |
| `opal.downstream.usr-infer-x-active-learning` | [USR Dataset With Infer-Derived X -> OPAL Active Learning](../../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) | `workflow` | `downstream-tool` | `round-loop` | `opal-campaign-state` | Start the label, train, and select loop once one explicit infer-derived X column or exported matrix already exists. |

### Tool-local runbook sources

This table is generated from owner-local `*.tool-source.yaml` metadata sidecars. Edit those files instead of hand-editing rows here.

| Tool | Docs entrypoint | What you will find |
| --- | --- | --- |
| `densegen` | [DenseGen documentation](../../src/dnadesign/densegen/docs/README.md) | Tool-owned tutorials, HPC runbooks, and event-producing demo flows. |
| `construct` | [construct docs](../../src/dnadesign/construct/docs/README.md) | Tool-owned workspace demos, template realization docs, and anchor-placement contracts. |
| `usr` | [USR docs](../../src/dnadesign/usr/docs/README.md) | Tool-owned dataset lifecycle docs, source-of-truth runbooks, sync routes, and promoter feature-matrix handoffs. |
| `infer` | [infer docs](../../src/dnadesign/infer/docs/README.md) | Tool-owned feature extraction runbooks, Evo2 docs, feature-schema contracts, and pressure-test flows. |
| `cluster` | [cluster docs](../../src/dnadesign/cluster/docs/README.md) | Tool-owned exploratory analysis workflow plus CLI, results, and artifact contracts. |
| `opal` | [OPAL Documentation](../../src/dnadesign/opal/docs/index.md) | Tool-owned active-learning workflows, campaign configuration references, and downstream infer-to-OPAL routes. |
| `notify` | [Notify Operations](../notify/README.md) | Tool-owned operator routes for watcher setup, delivery validation, recovery, and scheduler-adjacent notification flows. |
| `cruncher` | [Cruncher Documentation Index](../../src/dnadesign/cruncher/docs/README.md) | Tool-owned demos, studies, analysis guides, and optimization references. |
| `ops` | [ops docs](../../src/dnadesign/ops/docs/README.md) | Control-plane orchestration docs, packaged presets, and runbook lifecycle commands. |

### Progress surface glossary

`Progress kind` names the owner-local status surface and the corresponding read-only `ops progress show` adapter. Use `uv run ops progress scaffold <registry-id> ...` to emit the required manifest keys for one or more registered procedures, then use `uv run ops progress campaign --manifest <manifest.yaml>` when you want one explicit multi-step summary. This is still not an inferred global campaign engine.

| Progress kind | Meaning | Check next |
| --- | --- | --- |
| `ops-audit-json` | Workspace-scoped control-plane audit payload emitted by `ops runbook execute`. | Inspect `<workspace-root>/outputs/logs/ops/audit/*.json` plus the orchestration audit contract in [orchestration runbooks](../operations/orchestration-runbooks.md). |
| `usr-sync-audit` | USR sync parity and drift review surface for iterative cross-host updates. | Inspect the linked USR sync runbook and the [USR sync audit loop](../../src/dnadesign/usr/docs/operations/sync-audit-loop.md). |
| `usr-dataset-state` | Current USR dataset shape, overlays, and validation state after staged data-plane work. | Inspect the linked USR runbook plus `usr validate`, `usr head`, and related dataset-state checks named there. |
| `cluster-run-index` | Cluster workspace run records, embeddings, plots, and analysis outputs. | Inspect the linked cluster workflow and the cluster results root for the chosen workspace or direct run. |
| `opal-campaign-state` | OPAL round state, run ledgers, and latest selection outputs. | Inspect the linked OPAL workflow and its `opal status`, `opal runs list`, and `opal ctx audit` commands. |

### Explicit campaign manifest shape

Use `ops progress scaffold <registry-id> ...` when you want the smallest explicit manifest skeleton with the right placeholder fields, use `ops progress scaffold --related-to <registry-id>` when you want a relation-based starting point from one registered procedure, then use `ops progress campaign` when you want one concise progress summary across multiple registered procedures while still keeping every step pointed back to owner-local artifacts.

```yaml
campaign_id: demo_cross_tool_campaign
steps:
  - label: orchestration
    registry_id: ops.control-plane.orchestration
    audit_json: <workspace-root>/outputs/logs/ops/audit/latest.json
  - label: feature-matrix
    registry_id: usr.data-plane.promoter-feature-matrix
    usr_root: <usr-root>
    dataset: <dataset>
  - label: active-learning
    registry_id: opal.downstream.usr-infer-x-active-learning
    opal_config: <opal-workdir>/configs/campaign.yaml
```

- Generate the same skeleton from registry ids with `uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix opal.downstream.usr-infer-x-active-learning`.
- Expand one registered procedure into a relation-based starting point with `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix`.
- Relative artifact paths in the manifest resolve from the manifest directory, not from the shell's current working directory.
- `--related-to` expands the named procedure first, then typed related procedures in owner-local registry metadata relation order. Reorder the manifest explicitly when your campaign chronology differs.
- The manifest is explicit by design. Ops does not infer hidden steps or invent owner-local progress semantics.

### Boundary reminders

- Keep runbooks and workflows owner-local. This catalog links to them; it does not duplicate them.
- Catalog rows are generated from owner-local `*.registry.yaml` metadata sidecars and must stay aligned with the linked owner-local procedure metadata fields; drift is a docs-check failure.
- `ops` owns executable control-plane runbooks. It does not own durable USR-backed data-plane procedures.
- `docs/README.md` remains the top-level router by ownership plane. Use this catalog when you want a concise inventory view first.
- `Progress kind` names the owner-local status surface plus the corresponding read-only `ops progress show` adapter. `ops progress scaffold` emits placeholders only for the explicit registered steps you name, and `ops progress campaign` summarizes only the explicit steps named in a manifest.
