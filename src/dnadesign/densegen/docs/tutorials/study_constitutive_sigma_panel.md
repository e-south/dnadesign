## Constitutive sigma70 panel tutorial

This tutorial covers the `study_constitutive_sigma_panel` workspace as a sigma70-only constitutive promoter panel study. Read it when you need to run a fixed-element combinatorics campaign end to end, understand how expansion drives quotas, and review plots in the generated notebook gallery. Panel motifs follow the constitutive promoter study in DOI `10.1038/s41467-017-02473-5`.

### What this tutorial demonstrates
This section summarizes the runtime behaviors this workspace is designed to exercise.

- Plan-template expansion from sigma70 `-35` and `-10` motif sets.
- Fixed-element combinatorics using `promoter_matrix` (`cross_product` and `explicit_pairs`).
- Global motif exclusion with fixed-element allowlist exceptions.
- Full run flow: init, validate, run, plot, notebook generate, notebook open.

### Prerequisites
This section verifies that your environment is ready before running a larger study.

```bash
# Install locked dependencies for this repository.
uv sync --locked

# Confirm the DenseGen CLI is available.
uv run dense --help

# Validate the packaged constitutive workspace config and probe solver availability.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
```

### Key config knobs
This section highlights the highest-signal keys that control expansion size, quotas, and study behavior. For exact field contracts, use **[config reference](../reference/config.md)** and inspect **[`study_constitutive_sigma_panel/config.yaml`](../../workspaces/study_constitutive_sigma_panel/config.yaml)**.

- `densegen.motif_sets.sigma70_upstream_35`: Defines six `-35` motifs (`a` through `f`).
- `densegen.motif_sets.sigma70_downstream_10`: Defines eight `-10` motifs (`A` through `H`).
- `densegen.generation.plan_templates[0].fixed_elements.promoter_matrix.pairing.mode: cross_product`: Expands to all `6 x 8 = 48` sigma70 promoter cores.
- `densegen.generation.plan_templates[0].total_quota: 480`: Distributes 10 sequences per expanded cross-product variant.
- `densegen.generation.plan_templates[1].pairing.mode: explicit_pairs`: Adds targeted top-up variants.
- `densegen.generation.plan_templates[1].total_quota: 20`: Brings overall quota to `500`.
- `densegen.generation.plan_template_max_expanded_plans: 64`: Caps expansion size; current config expands to 50 plans.
- `densegen.generation.plan_template_max_total_quota: 500`: Caps aggregate expanded quota and enforces study-size intent.
- `densegen.generation.sequence_constraints.forbid_kmers`: Blocks sigma70 panel motifs globally on both strands.
- `densegen.generation.sequence_constraints.allowlist`: Allows only intended fixed-element instances to bypass global motif blocking.
- `densegen.runtime.max_seconds_per_plan` and `densegen.runtime.max_failed_solutions`: Bound solve time and failure pressure at study scale.

### Expansion and quota behavior
This section explains exactly how the config expands and why the study quota is 500.

- `sigma70_panel` template: `cross_product` expansion over six `-35` and eight `-10` motifs gives 48 variants, each with quota 10 (`total_quota: 480`).
- `sigma70_topup` template: `explicit_pairs` adds two selected variants at quota 10 each (`total_quota: 20`).
- Expanded plan count: `48 + 2 = 50`.
- Expanded total quota: `480 + 20 = 500`.

### LacI and AraC exclusion stub
This section captures the planned extension point without wiring paths yet.

- Current config excludes sigma70 panel hexamers only.
- LacI and AraC exclusion is intentionally not wired in this template yet.
- When ready, add local motif artifacts and reference them using the same motif-artifact pattern used by other DenseGen workspaces.

### Walkthrough
This section runs the workspace in the intended order, then confirms artifacts used by plotting and notebook gallery views.

#### 1) Create a study workspace
This step creates an isolated workspace copy so you can run and edit safely.

```bash
# Resolve repo and workspace roots so paths stay explicit.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Initialize a local workspace from the packaged constitutive sigma70 template.
uv run dense workspace init --id constitutive_panel_trial --root "$WORKSPACE_ROOT" --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode both

# Enter the workspace and pin the config path for later commands.
cd "$WORKSPACE_ROOT/constitutive_panel_trial"
CONFIG="$PWD/config.yaml"
```

#### 2) Validate and inspect expansion before runtime
This step confirms that expanded plan count and total quota match expectations before generation work starts.

```bash
# Validate strict schema and probe solver.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect expanded plan rows and per-plan quotas from the CLI.
uv run dense inspect plan -c "$CONFIG"

# Print expansion totals directly from the resolved config object.
uv run python - <<'PY'
from dnadesign.densegen.src.config import load_config

cfg = load_config("config.yaml").root.densegen.generation
plans = list(cfg.plan or [])
print("expanded_plans:", len(plans))
print("total_quota:", sum(int(item.quota) for item in plans))
PY
```

#### 3) Build Stage-A pool and run DenseGen
This step generates the background pool, then runs solve-to-quota under fixed-element and sequence-constraint rules.

```bash
# Build Stage-A pool artifacts from scratch.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Run DenseGen without auto-plot so diagnostics can be inspected first.
uv run dense run --no-plot -c "$CONFIG"

# Inspect event stream and library summaries for plan pressure and rejections.
uv run dense inspect run --events --library -c "$CONFIG"
```

#### 4) Generate plots and validate gallery inputs
This step renders all configured plot families and confirms the manifest has plan-scoped entries for notebook gallery navigation.

```bash
# Generate stage_a_summary, placement_map, run_health, and tfbs_usage plots.
uv run dense plot -c "$CONFIG"

# Summarize plot manifest inventory that the notebook gallery will use.
uv run python - <<'PY'
import json
from collections import Counter
from pathlib import Path

manifest = json.loads(Path("outputs/plots/plot_manifest.json").read_text())
plots = list(manifest.get("plots") or [])
print("plot_entries:", len(plots))
print("plot_ids:", sorted({str(item.get("plot_id") or "") for item in plots}))
counts = Counter(str(item.get("plan_name") or "unscoped") for item in plots)
for plan_name in sorted(counts):
    print(f"{plan_name}: {counts[plan_name]}")
PY
```

#### 5) Generate and open the notebook
This step scaffolds the workspace notebook, validates it with marimo, and opens it for interactive review.

```bash
# Generate the workspace-scoped marimo notebook.
uv run dense notebook generate -c "$CONFIG"

# Validate notebook structure and runtime wiring.
uv run marimo check outputs/notebooks/densegen_run_overview.py

# Open the generated notebook in edit mode.
uv run dense notebook run -c "$CONFIG"
```

### Diversity and composition diagnostics
This section gives concrete checks for fixed-element coverage and composition balance after a run.

```bash
# Summarize accepted records by expanded plan name.
uv run python - <<'PY'
import pandas as pd

records = pd.read_parquet("outputs/tables/records.parquet", columns=["densegen__plan"])
print(records.groupby("densegen__plan").size().sort_values(ascending=False))
PY

# Summarize fixed-element variant usage from record metadata.
uv run python - <<'PY'
import pandas as pd

records = pd.read_parquet(
    "outputs/tables/records.parquet",
    columns=["densegen__plan", "densegen__fixed_elements"],
)
rows = []
for _, row in records.iterrows():
    fixed = row["densegen__fixed_elements"] or {}
    constraints = list(fixed.get("promoter_constraints") or [])
    if not constraints:
        continue
    pc = constraints[0]
    rows.append(
        {
            "plan": row["densegen__plan"],
            "upstream_variant_id": pc.get("upstream_variant_id"),
            "downstream_variant_id": pc.get("downstream_variant_id"),
        }
    )
summary = pd.DataFrame(rows)
print(summary.groupby(["upstream_variant_id", "downstream_variant_id"]).size().sort_values(ascending=False).head(20))
PY
```

### Expected outputs
This section lists the core artifacts that should exist when the flow completes.

- `outputs/meta/effective_config.json` (resolved expansion and quotas)
- `outputs/meta/events.jsonl` (runtime diagnostics)
- `outputs/tables/records.parquet`
- `outputs/tables/composition.parquet`
- `outputs/libraries/library_members.parquet`
- `outputs/plots/plot_manifest.json` (gallery source)
- `outputs/plots/stage_a/*.pdf`
- `outputs/plots/stage_b/<plan>/occupancy.pdf`
- `outputs/plots/stage_b/<plan>/tfbs_usage.pdf`
- `outputs/plots/run_health/*.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section lists the highest-frequency issues for this study shape and how to address them.

- Expansion cap error: increase `plan_template_max_expanded_plans` only after confirming motif-set sizes are intentional.
- Total quota cap error: reduce template quotas or raise `plan_template_max_total_quota` deliberately.
- Slow convergence in specific plans: increase `runtime.max_seconds_per_plan`, increase Stage-B `library_size`, or run a smaller smoke quota first.
- Unexpected motif rejections: inspect `outputs/meta/events.jsonl` and verify `allowlist` selectors match fixed-element components.
- Notebook gallery missing plots: rerun `uv run dense plot -c "$CONFIG"` and check `outputs/plots/plot_manifest.json`.
