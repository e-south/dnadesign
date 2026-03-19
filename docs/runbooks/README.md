## Runbook Catalog

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-19

Use this page when you want a concise inventory of authoritative runbooks and workflows without guessing which tool owns the next deep procedure. This page is a discovery surface only. It does not replace the owner-local runbook or workflow that each entry links to.

If you prefer terminal discovery, use `uv run ops catalog list` for the grouped inventory, `uv run ops catalog list --plane data-plane --query infer` to narrow the inventory by intent, and `uv run ops catalog show <registry-id>` for one registered cross-tool procedure.

### Authoritative cross-tool procedures

| Registry id | Procedure | Type | Plane | Execution kind | Progress kind | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| `ops.control-plane.orchestration` | [Orchestration runbooks](../operations/orchestration-runbooks.md) | `runbook` | `control-plane` | `executable` | `ops-audit-json` | Deterministic control-plane runbook contract for DenseGen or Infer batch submit flows with optional Notify chaining. |
| `usr.data-plane.hpc-sync` | [USR HPC sync flow](../../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | `runbook` | `data-plane` | `iterative` | `usr-sync-audit` | Keep one USR dataset synchronized between HPC and local analysis with explicit diff, pull, and push verification. |
| `usr.data-plane.chained-densegen-infer-sync` | [Chained DenseGen and Infer sync runbook](../../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md) | `runbook` | `data-plane` | `iterative` | `usr-sync-audit` | Coordinate DenseGen-on-HPC and Infer-local writes against one USR dataset with explicit sync checkpoints. |
| `usr.data-plane.multi-source-source-of-truth` | [Multi-source source-of-truth assembly](../../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Merge multiple USR-backed sources, preserve explicit carry, and hand one construct-backed downstream dataset to Infer and Notify. |
| `usr.data-plane.construct-infer-source-of-truth` | [Construct -> USR -> Infer source-of-truth runbook](../../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Realize construct outputs into one shared USR dataset and use that dataset as the durable Infer handoff. |
| `usr.data-plane.promoter-feature-matrix` | [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Build one infer-annotated feature matrix from mixed promoter sources before branching into Cluster or OPAL. |
| `cluster.downstream.exploratory-clustering` | [Exploratory clustering workflow](../../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) | `workflow` | `downstream-tool` | `exploratory` | `cluster-run-index` | Explore one chosen feature definition through clustering, UMAP, and downstream summaries. |
| `opal.downstream.usr-infer-x-active-learning` | [USR dataset with infer-derived X -> OPAL active learning](../../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) | `workflow` | `downstream-tool` | `round-loop` | `opal-campaign-state` | Start the label, train, and select loop once one explicit infer-derived X column or exported matrix already exists. |

### Tool-local runbook sources

These links are the owner-local entrypoints for tool-owned demos, tutorials, and operational procedures. Use them when you want packaged examples or tool-specific runbooks that are not part of the smaller cross-tool catalog above.

| Tool | Docs entrypoint | What you will find |
| --- | --- | --- |
| `densegen` | [DenseGen docs](../../src/dnadesign/densegen/docs/README.md) | Tool-owned tutorials, HPC runbooks, and event-producing demo flows. |
| `construct` | [construct docs](../../src/dnadesign/construct/docs/README.md) | Tool-owned workspace demos, template realization docs, and workspace registry routes. |
| `infer` | [infer docs](../../src/dnadesign/infer/docs/README.md) | Tool-owned feature extraction runbooks, Evo2 docs, and pressure-test flows. |
| `cluster` | [cluster docs](../../src/dnadesign/cluster/docs/README.md) | Tool-owned exploratory analysis workflow plus CLI and artifact contracts. |
| `opal` | [opal docs index](../../src/dnadesign/opal/docs/index.md) | Tool-owned active-learning workflows and campaign configuration references. |
| `cruncher` | [Cruncher documentation index](../../src/dnadesign/cruncher/docs/README.md) | Tool-owned demos, studies, and optimization runbooks. |
| `ops` | [ops docs](../../src/dnadesign/ops/docs/README.md) | Control-plane orchestration docs, packaged presets, and runbook lifecycle commands. |

### Boundary reminders

- Keep runbooks and workflows owner-local. This catalog links to them; it does not duplicate them.
- Catalog rows mirror `Registry-id`, `Type`, `Plane`, `Execution-kind`, `Progress-kind`, and `Summary` declared in the linked owner-local procedure; drift is a docs-check failure.
- `ops` owns executable control-plane runbooks. It does not own durable USR-backed data-plane procedures.
- `docs/README.md` remains the top-level router by ownership plane. Use this catalog when you want a concise inventory view first.
- `Progress kind` names the owner-local status surface to inspect next. It is not yet a unified multi-tool campaign status API.
