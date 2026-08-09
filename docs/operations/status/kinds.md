## OPS status kinds

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-09

Registered public OPS routes and their status-kind providers:

| Registry id | Status kind | Observed plane | Surface type | Summary scope | Cost | Required inputs | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ops.control-plane.orchestration` | `ops-audit-json` | `control` | `orchestration_audit` | `workspace` | `cheap` | `--audit-json` | `ops` |
| `usr.data-plane.hpc-sync` | `usr-sync-audit` | `data` | `sync_audit` | `workspace` | `cheap` | `--sync-audit-json` | `usr` |
| `usr.data-plane.chained-densegen-infer-sync` | `usr-sync-audit` | `data` | `sync_audit` | `workspace` | `cheap` | `--sync-audit-json` | `usr` |
| `usr.data-plane.multi-source-source-of-truth` | `usr-dataset-state` | `data` | `dataset_snapshot` | `workspace` | `cheap` | `--usr-root`, `--dataset` | `usr` |
| `usr.data-plane.construct-infer-source-of-truth` | `usr-dataset-state` | `data` | `dataset_snapshot` | `workspace` | `cheap` | `--usr-root`, `--dataset` | `usr` |
| `usr.data-plane.promoter-feature-matrix` | `usr-dataset-state` | `data` | `dataset_snapshot` | `workspace` | `cheap` | `--usr-root`, `--dataset` | `usr` |
| `cluster.downstream.exploratory-clustering` | `cluster-run-index` | `data` | `run_index` | `workspace` | `cheap` | `--cluster-results-root` | `cluster` |
| `opal.downstream.usr-infer-x-active-learning` | `opal-campaign-state` | `control` | `campaign_snapshot` | `workspace` | `cheap` | `--opal-config` | `opal` |

### Direct status-kind inventory

`uv run ops progress kinds --json` is the complete provider inventory for the
installed environment. It also lists owner-provided status kinds that are not
catalog-backed public routes yet, including `latentdna-workspace-snapshot` for
LatentDNA workspace snapshot inspection. External packages may register
additional routes; dnadesign does not advertise study routes it does not ship.

### Notes

- `registry id` is the public route name you pass to `ops catalog show`, `ops progress explain`, and `ops progress show`.
- `status kind` is the lower-level shared status implementation contract.
- `cost` describes the expected read cost of the status surface, not whether the underlying workflow is cheap.
- Start with `uv run ops progress explain <registry-id>` if you do not already know the required flags.
