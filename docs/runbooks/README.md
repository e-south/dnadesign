## Runbook Catalog

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-25

Use this page when you want a command first. Start with `uv run ops catalog list --simple`, then open the linked runbook or tool doc once you know the route.
Treat the CLI surfaces explicitly: `ops catalog` is discovery, `ops progress` is observation, and `ops runbook` is the control plane that plans or executes deterministic batch work.

Use the command table first. The generated tables later on are reference.

### Command lookup

| If you want to... | Use this command | What it does |
| --- | --- | --- |
| Browse a quick inventory | `uv run ops catalog list --simple` | Registered procedures and tool docs without the extra type and plane labels. |
| Browse the full inventory | `uv run ops catalog list` | Full list of cross-tool procedures plus tool docs entrypoints. |
| Narrow by rough intent | `uv run ops catalog list --plane data-plane --query infer` | Smaller procedure set when you already know the downstream path is data-plane and infer-adjacent. |
| Browse tool docs only | `uv run ops catalog list --section tool-sources` | Tool entrypoints when you want package docs first. |
| Inspect one registered procedure | `uv run ops catalog show <registry-id>` | Owner docs, related procedures, linked deeper docs, required status inputs, and next shell commands. |
| Explain one status check | `uv run ops progress explain <registry-id>` | Required flags, direct `progress show` command, and adapter-specific notes before you touch artifacts. |
| Check one status view | `uv run ops progress show <registry-id> ...` | One registered status surface with explicit artifact inputs. |
| Start a campaign manifest | `uv run ops progress scaffold <registry-id> ...` or `uv run ops progress scaffold --related-to <registry-id>` | YAML manifest skeleton for one route or one related route set. It prints to stdout unless you pass `--out`. |
| Check a campaign | `uv run ops progress campaign --manifest <manifest.yaml>` | Summary of the steps you list in the manifest. |
| Fill missing Infer lanes | `uv run ops runbook fill-infer --study-dir docs/studies/<study-id>` | Inspects checked-in Infer runbooks, skips complete sequence-view lanes, and plans only missing vector/scalar work. |

### Common examples

- `uv run ops catalog list --simple`: shorter inventory when you are new to the registry and do not need the taxonomy first.
- `uv run ops catalog show usr.data-plane.promoter-feature-matrix`: one procedure with its owner docs, related tool docs, linked deeper docs, required status inputs, and next commands.
- `uv run ops progress explain usr.data-plane.promoter-feature-matrix`: required flags, direct `progress show` command, and notes before you read that status view.
- `uv run ops progress show usr.data-plane.promoter-study-status`: one checked-in active-study summary for current phase, dataset presence, and next ready surface. Add `--repo-root <repo-root> --study-dir docs/studies/<study-id>` to pin a different study or invoke it from outside the repo checkout.
- `uv run ops progress show usr.data-plane.promoter-study-preflight`: deeper read-only preflight across DenseGen, Construct, Infer, Notify, and batch-plan surfaces for the checked-in active study.
- `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel`: study-level Infer completion plan for all declared sequence-view runbooks; add `--submit` only from the target HPC batch environment after Notify is configured.
- `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix`: expand one registered procedure into a starting manifest with the named procedure first and its related procedures after it.
- `uv run ops progress campaign --manifest <manifest.yaml>`: read-only summary for the steps listed in the manifest.
- For one live promoter study, keep the checked-in files under `docs/studies/<study-id>/`. Use [Study records index](../studies/README.md) for the required layout and selector rules.

Start with the summary and linked doc. The extra labels matter only when two routes look similar.

### Cross-tool procedures

This table is generated from `*.registry.yaml` sidecars. Edit those files instead of hand-editing rows here.

| Registry id | Procedure | Type | Plane | Execution kind | Status kind | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| `ops.control-plane.orchestration` | [Orchestration runbooks](../operations/orchestration-runbooks.md) | `runbook` | `control-plane` | `executable` | `ops-audit-json` | Deterministic control-plane runbook contract for DenseGen or Infer batch submit flows with optional Notify chaining. |
| `usr.data-plane.hpc-sync` | [USR HPC Sync Flow](../../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md) | `runbook` | `data-plane` | `iterative` | `usr-sync-audit` | Keep one USR dataset synchronized between HPC and local analysis with explicit diff, pull, and push verification. |
| `usr.data-plane.chained-densegen-infer-sync` | [Chained DenseGen and Infer Sync Runbook](../../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-runbook.md) | `runbook` | `data-plane` | `iterative` | `usr-sync-audit` | Coordinate DenseGen-on-HPC and Infer-local writes against one USR dataset with explicit sync checkpoints. |
| `usr.data-plane.multi-source-source-of-truth` | [Multi-Source Shared Dataset Assembly](../../src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Merge multiple USR-backed sources, preserve explicit carry, and hand one construct-backed shared dataset to Infer and Notify. |
| `usr.data-plane.construct-infer-source-of-truth` | [Construct -> USR -> Infer Shared Dataset Runbook](../../src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Realize construct outputs into one shared USR dataset and use that dataset as the durable Infer handoff. |
| `usr.data-plane.promoter-study-status` | [Promoter Study Status Contract](../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md) | `contract` | `data-plane` | `iterative` | `promoter-study-status` | Read one checked-in promoter-study record and report the current phase, datasets, and handoff surfaces without reconstructing the workflow by hand. |
| `usr.data-plane.promoter-study-preflight` | [Promoter Study Preflight](../../src/dnadesign/usr/docs/operations/promoter-study-preflight.md) | `contract` | `data-plane` | `iterative` | `promoter-study-preflight` | Run the active promoter-study preflight suite across DenseGen, Construct, Infer, Notify, and batch-plan contracts without mutating data or submitting jobs. |
| `cruncher.data-plane.cruncher-study-status` | [Cruncher Study Status Contract](../../src/dnadesign/cruncher/docs/operations/cruncher-study-status.md) | `contract` | `data-plane` | `iterative` | `cruncher-study-status` | Read one checked-in cruncher-study record and report the current phase, command groups, and context refs without reconstructing the study by hand. |
| `cruncher.data-plane.cruncher-study-preflight` | [Cruncher Study Preflight](../../src/dnadesign/cruncher/docs/operations/cruncher-study-preflight.md) | `contract` | `data-plane` | `iterative` | `cruncher-study-preflight` | Run the current cruncher-study preflight suite across declared workspace-layout and command-validation surfaces without mutating outputs. |
| `usr.data-plane.promoter-feature-matrix` | [Promoter Characterization Feature Matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) | `runbook` | `data-plane` | `staged` | `usr-dataset-state` | Build one infer-annotated feature matrix from mixed promoter sources before branching into Cluster or OPAL. |
| `cluster.downstream.exploratory-clustering` | [Exploratory clustering workflow](../../src/dnadesign/cluster/docs/workflows/exploratory-clustering.md) | `workflow` | `downstream-tool` | `exploratory` | `cluster-run-index` | Explore one chosen feature column or exported matrix through clustering, UMAP, and downstream summaries. |
| `opal.downstream.usr-infer-x-active-learning` | [USR Dataset With Infer-Derived X -> OPAL Active Learning](../../src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md) | `workflow` | `downstream-tool` | `round-loop` | `opal-campaign-state` | Start the label, train, and select loop once one explicit infer-derived X column or exported matrix already exists. |

### Tool docs

This table is generated from `*.tool-source.yaml` sidecars. Edit those files instead of hand-editing rows here.

| Tool | Docs entrypoint | What you will find |
| --- | --- | --- |
| `densegen` | [DenseGen documentation](../../src/dnadesign/densegen/docs/README.md) | Tool-owned tutorials, HPC runbooks, and event-producing demo flows. |
| `construct` | [Construct docs](../../src/dnadesign/construct/docs/README.md) | Tool-owned workspace demos, template realization docs, and anchor-placement contracts. |
| `usr` | [USR docs](../../src/dnadesign/usr/docs/README.md) | Tool-owned dataset lifecycle docs, dataset handoffs, sync routes, and promoter feature assembly. |
| `infer` | [infer docs](../../src/dnadesign/infer/docs/README.md) | Tool-owned feature extraction runbooks, Evo2 docs, feature-schema contracts, and pressure-test flows. |
| `cluster` | [Cluster docs](../../src/dnadesign/cluster/docs/README.md) | Tool-owned exploratory analysis workflow plus CLI, results, and artifact contracts. |
| `opal` | [OPAL Documentation](../../src/dnadesign/opal/docs/index.md) | Tool-owned active-learning workflows, campaign configuration references, and downstream infer-to-OPAL routes. |
| `latentdna` | [LatentDNA Docs](../../src/dnadesign/latentdna/docs/README.md) | Pre-assay triage workflows, workspace contracts, and read-only QC surfaces for promoter studies. |
| `notify` | [Notify Operations](../notify/README.md) | Tool-owned operator routes for watcher setup, delivery validation, recovery, and scheduler-adjacent notification flows. |
| `cruncher` | [Cruncher docs](../../src/dnadesign/cruncher/docs/README.md) | Tool-owned demos, studies, analysis guides, and optimization references. |
| `ops` | [Ops docs](../../src/dnadesign/ops/docs/README.md) | Ops commands, packaged presets, and runbook lifecycle docs. |

### Status views

You only need this section after `uv run ops progress explain <registry-id>` or `uv run ops catalog show <registry-id>` points you to a specific status view.

`Status kind` names the status adapter used by `ops progress show`. Use `uv run ops progress scaffold <registry-id> ...` to emit the required manifest keys, then `uv run ops progress campaign --manifest <manifest.yaml>` when you want one multi-step summary. `ops progress scaffold` prints YAML to stdout by default and only writes when you pass `--out`. This command only summarizes the manifest you provide.

| Status kind | Meaning | Check next |
| --- | --- | --- |
| `ops-audit-json` | Workspace-scoped control-plane audit payload emitted by `ops runbook execute`. | Inspect `<workspace-root>/outputs/logs/ops/audit/*.json` plus the orchestration audit contract in [orchestration runbooks](../operations/orchestration-runbooks.md). |
| `usr-sync-audit` | USR sync parity and drift review for iterative cross-host updates. | Inspect the linked USR sync runbook and the [USR sync audit loop](../../src/dnadesign/usr/docs/operations/sync-audit-loop.md). |
| `usr-dataset-state` | Current USR dataset shape, overlays, and validation state after staged data-plane work. | Inspect the linked USR runbook plus `usr validate`, `usr head`, and related dataset-state checks named there. |
| `promoter-study-status` | Checked-in promoter-study directory state, including source-dataset presence, current phase, and missing execution surfaces. | Inspect the linked study-status contract plus the checked-in `ops.study.yaml`, `datasets.yaml`, `status.md`, and `pipeline.yaml` when execution surfaces are present. |
| `promoter-study-preflight` | Checked-in promoter-study command-level preflight across DenseGen, Construct, Infer, Notify, and batch-plan surfaces. | Inspect the linked preflight contract plus the checked-in `ops.study.yaml`, `datasets.yaml`, `status.md`, and `pipeline.yaml` when execution surfaces are present. |
| `cruncher-study-status` | Checked-in cruncher-study directory state, including current phase, command groups, and native-agent bootstrap context. | Inspect the linked study-status contract plus the checked-in `ops.study.yaml`, `status.md`, `routes.md`, and `pipeline.yaml` for the selected study. |
| `cruncher-study-preflight` | Checked-in cruncher-study command-level preflight across declared workspace-layout and read-only validation-command surfaces. | Inspect the linked preflight contract plus the checked-in `ops.study.yaml`, `status.md`, `routes.md`, and `pipeline.yaml` for the selected study. |
| `cluster-run-index` | Cluster workspace run records, embeddings, plots, and analysis outputs. | Inspect the linked cluster workflow and the cluster results root for the chosen workspace or direct run. |
| `opal-campaign-state` | OPAL round state, run ledgers, and latest selection outputs. | Inspect the linked OPAL workflow and its `opal status`, `opal runs list`, and `opal ctx audit` commands. |

### Explicit campaign manifest shape

Use `ops progress scaffold <registry-id> ...` when you want a manifest template with the right placeholder fields. Use `ops progress scaffold --related-to <registry-id>` when you want a starting point that includes linked procedures. Then use `ops progress campaign` when you want one summary across multiple steps. `scaffold` prints to stdout unless you pass `--out`.

```yaml
version: 2
path_base: repo
campaign_id: demo_cross_tool_campaign
steps:
  - label: orchestration
    registry_id: ops.control-plane.orchestration
    inputs:
      audit_json: repo:<workspace-root>/outputs/logs/ops/audit/latest.json
  - label: feature-matrix
    registry_id: usr.data-plane.promoter-feature-matrix
    inputs:
      usr_root: repo:<usr-root>
      dataset: <dataset>
  - label: active-learning
    registry_id: opal.downstream.usr-infer-x-active-learning
    inputs:
      opal_config: manifest:./opal/configs/campaign.yaml
```

- Generate the same skeleton from registry ids with `uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix opal.downstream.usr-infer-x-active-learning`.
- Expand one registered procedure into a relation-based starting point with `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix`.
- For a real promoter effort, store the manifest under `docs/studies/<study-id>/campaign.yaml` and keep the paired `datasets.yaml` registry and `status.md` there too. Use [Study records index](../studies/README.md) for the full layout.
- Campaign manifests must declare `version: 2` and `path_base: repo`, `manifest`, or `cwd`.
- `repo:` references resolve from repository root. `manifest:` plus `./` or `../` resolve from the manifest directory.
- Bare relative paths resolve from `path_base`.
- Provider inputs belong under `inputs:`. Ops does not read loose top-level step keys.
- `--related-to` expands the named procedure first, then related procedures in catalog order. Reorder the manifest when your campaign chronology differs.
- The manifest is explicit by design. Ops does not infer hidden steps.
- Smallest working status example: run `uv run ops runbook execute ... --no-submit --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json`. On workstations without `qstat`, add `--allow-missing-qstat` so the queue probe is explicit but non-fatal. Then pass the same audit path to `uv run ops progress show ops.control-plane.orchestration --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json`.
