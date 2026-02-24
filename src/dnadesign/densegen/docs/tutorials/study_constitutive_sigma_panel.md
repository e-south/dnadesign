## Constitutive sigma70 panel tutorial

This tutorial covers the `study_constitutive_sigma_panel` workspace as a constitutive sigma70 promoter panel study with strict LacI/AraC background exclusion. Read it when you need to run a fixed-element combinatorics campaign end to end, understand how expansion drives quotas, and review plots in the generated notebook gallery. Panel motifs follow the constitutive promoter study in DOI `10.1038/s41467-017-02473-5`.

### Fast path
For a command-only runbook mirror of this tutorial, use **[`study_constitutive_sigma_panel/runbook.md`](../../workspaces/study_constitutive_sigma_panel/runbook.md)**.

Run this single command from the repository root:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="constitutive_panel_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode both && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && pixi run fimo --version && pixi run dense validate-config --probe-solver -c "$CONFIG" && pixi run dense run --fresh --no-plot -c "$CONFIG" && pixi run dense inspect run --events --library -c "$CONFIG" && pixi run dense plot -c "$CONFIG" && pixi run dense notebook generate -c "$CONFIG"
```

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### What this tutorial demonstrates
This section summarizes the runtime behaviors this workspace is designed to exercise.

- Deterministic plan expansion from sigma70 `-35` and `-10` motif sets.
- Fixed-element combinatorics using `fixed_element_matrix` (`cross_product`).
- Background exclusion against Cruncher-exported LacI/AraC PWMs (`allow_zero_hit_only=true`).
- Global sigma70 motif exclusion with fixed-element allowlist exceptions.
- Full run flow: init, validate, run, plot, notebook generate, notebook open.

### Workspace intent and outcome design
This section explains why this workspace exists and what outcome shape it is designed to produce.

- Intent: build a constitutive sigma70 core panel where promoter core strength is the primary controlled variable.
- Use case: generate a broad, deterministic benchmark panel for downstream analysis of occupancy and composition under strict background filtering.
- Outcome design: produce balanced, plan-scoped cohorts so each core variant is directly comparable without quota bias.
- Operational meaning: matrix expansion moves combinatorics into schema-time compilation, so runtime only solves explicit concrete plans.

### Prerequisites
This section verifies that your environment is ready before running a larger study.
Run these commands from the repository root so `pixi run dense ...` resolves project tasks correctly.

```bash
# Install locked dependencies for this repository.
uv sync --locked

# Install pixi toolchain when FIMO is needed in Stage-A.
pixi install

# Confirm the DenseGen CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" dense --help

# Confirm the Cruncher CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" cruncher --help

# Confirm FIMO is available for strict background exclusion.
pixi run fimo --version

# Validate the packaged constitutive workspace config and probe solver availability.
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
```

### Key config knobs
This section highlights the highest-signal keys that control expansion size, quotas, and study behavior. For exact field contracts, use **[config reference](../reference/config.md)** and inspect **[`study_constitutive_sigma_panel/config.yaml`](../../workspaces/study_constitutive_sigma_panel/config.yaml)**.

- `densegen.motif_sets.sigma70_upstream_35`: Defines six `-35` motifs (`a` through `f`).
- `densegen.motif_sets.sigma70_downstream_10`: Defines eight `-10` motifs (`A` through `H`).
- `densegen.generation.plan[0].fixed_elements.fixed_element_matrix.pairing.mode: cross_product`: Expands to all `6 x 8 = 48` sigma70 promoter cores.
- `densegen.generation.plan[0].sequences: 48`: Distributes 1 sequence per expanded cross-product variant.
- `densegen.generation.plan[*].expanded_name_template: "{base}__sig35={up}__sig10={down}"`: Makes expanded plan identity explicit in outputs, plots, and notebook filters.
- `densegen.generation.expansion.max_plans: 64`: Caps expansion size; current config expands to 48 plans.
- `densegen.inputs[name=lacI_pwm|araC_pwm]`: Consumes committed Cruncher motif artifacts.
- `densegen.inputs[name=lacI_pwm|araC_pwm].sampling.trimming.window_length: 16`: Keeps MMR core length valid with `length.range: [16, 20]`.
- `densegen.inputs[name=background].sampling.filters.fimo_exclude`: Enforces strict LacI/AraC negative selection (`allow_zero_hit_only=true`).
- `densegen.inputs[name=background].sampling.mining.budget.candidates: 8000000`: Increases mining budget to preserve feasible yield under strict exclusion.
- `densegen.generation.sequence_constraints.forbid_kmers`: Blocks sigma70 panel motifs globally on both strands.
- `densegen.generation.sequence_constraints.allowlist`: Allows only intended fixed-element instances to bypass global motif blocking.
- `densegen.runtime.max_seconds_per_plan` and `densegen.runtime.max_failed_solutions`: Bound solve time and failure pressure at study scale.
- `plots.options.placement_map|tfbs_usage.scope: auto` with `max_plans: 12`: keeps Stage-B plots compact for expanded studies while allowing per-plan drilldown.

### Expansion and quota behavior
This section explains exactly how the config expands and why the study quota is 48.

- Matrix expansion treats `sequences` as the base-plan target and enforces an exact divisible split across expanded variants.
- `sigma70_panel` plan: `cross_product` expansion over six `-35` and eight `-10` motifs gives 48 variants, each with quota 1 (`sequences: 48`).
- Expanded plan count: `48`.
- Expanded total quota: `48`.

To scale this workspace while preserving balanced per-variant quotas:

- set `sequences` to a multiple of the expanded variant count (`48`)
- increase `generation.expansion.max_plans` only if you intentionally expand the variant domain
- use `dense run --resume --extend-quota <n>` when you want runtime growth without changing config

### Matrix policy in this workspace
This section records the hard behavior guarantees for expansion and plan compilation.

- Expansion is deterministic and static at config-load time.
- Expansion fails fast on invalid variant IDs, invalid pairing, duplicate expanded names, quota mismatches, or cap overflow.
- Matrix expansion uses strict uniform split and requires exact divisibility.
- Runtime does not adapt or mutate expansion; Stage-B executes resolved concrete plans only.

### LacI and AraC exclusion contract
This section defines the enforced Cruncher-to-DenseGen handoff for this workspace.

- `study_constitutive_sigma_panel` ships with committed LacI/AraC PWM artifacts under `inputs/motif_artifacts/`.
- Background sequences are filtered through `fimo_exclude` against those two PWM inputs with `allow_zero_hit_only=true`.
- The default artifact producer is `src/dnadesign/cruncher/workspaces/pairwise_laci_arac/`.

### Walkthrough
This section runs the workspace in the intended order, then confirms artifacts used by plotting and notebook gallery views.

#### 1) Create a study workspace
This step creates an isolated workspace copy so you can run and edit safely.

```bash
# Resolve repo and workspace roots so paths stay explicit.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Initialize a local workspace from the packaged constitutive sigma70 template.
uv run --project "$REPO_ROOT" dense workspace init --id constitutive_panel_trial --root "$WORKSPACE_ROOT" --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode both

# Pin workspace and config paths for later commands.
WORKSPACE="$WORKSPACE_ROOT/constitutive_panel_trial"
CONFIG="$WORKSPACE/config.yaml"
```

#### 2) Refresh LacI/AraC artifacts from Cruncher (optional)
This step refreshes workspace-local LacI/AraC PWM artifacts from the canonical Cruncher source.

```bash
# Refresh artifacts directly into this DenseGen workspace.
uv run --project "$REPO_ROOT" cruncher catalog export-densegen --set 1 --densegen-workspace "$WORKSPACE" -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml"

# Verify manifest and artifact files.
ls -1 "$WORKSPACE/inputs/motif_artifacts"
```

#### 3) Validate and inspect expansion before runtime
This step confirms strict config validity, expanded plan count, and total quota before generation work starts.

```bash
# Validate strict schema and probe solver.
pixi run dense validate-config --probe-solver -c "$CONFIG"

# Inspect expanded plan rows and per-plan quotas from the CLI.
pixi run dense inspect plan -c "$CONFIG"

# Print matrix-derived expansion totals directly from config.yaml.
uv run --project "$REPO_ROOT" python - <<PY
from pathlib import Path

import yaml

payload = yaml.safe_load(Path("$CONFIG").read_text())
densegen = payload.get("densegen", {})
motif_sets = densegen.get("motif_sets", {})
generation = densegen.get("generation", {})

upstream_set = motif_sets.get("sigma70_upstream_35") or {}
downstream_set = motif_sets.get("sigma70_downstream_10") or {}
upstream = len(upstream_set) if isinstance(upstream_set, dict) else len((upstream_set or {}).get("variants") or [])
downstream = len(downstream_set) if isinstance(downstream_set, dict) else len((downstream_set or {}).get("variants") or [])
plans = list(generation.get("plan") or [])
cross_product_plans = upstream * downstream
total_sequences = sum(int(item.get("sequences", 0) or 0) for item in plans)

print("expanded_plans_expected:", cross_product_plans)
print("total_sequences_expected:", total_sequences)
PY
```

#### 4) Build Stage-A pool and run DenseGen
This step generates the background pool, then runs solve-to-quota under fixed-element and sequence-constraint rules.

```bash
# Build Stage-A pool artifacts from scratch.
pixi run dense stage-a build-pool --fresh -c "$CONFIG"

# Run DenseGen without auto-plot so diagnostics can be inspected first.
pixi run dense run --no-plot -c "$CONFIG"

# Inspect event stream and library summaries for plan pressure and rejections.
pixi run dense inspect run --events --library -c "$CONFIG"
```

#### 5) Generate plots and validate gallery inputs
This step renders all configured plot families and confirms the manifest has plan-scoped entries for notebook gallery navigation.
For this workspace, Stage-A plots are expected to include only `background_logo` because plans sample only `background` (not PWM input pools).

```bash
# Generate stage_a_summary, placement_map, run_health, and tfbs_usage plots.
pixi run dense plot -c "$CONFIG"

# Summarize plot manifest inventory that the notebook gallery will use.
uv run --project "$REPO_ROOT" python - <<PY
import json
from collections import Counter
from pathlib import Path

manifest = json.loads(Path("$WORKSPACE/outputs/plots/plot_manifest.json").read_text())
plots = list(manifest.get("plots") or [])
print("plot_entries:", len(plots))
print("plot_ids:", sorted({str(item.get("plot_id") or "") for item in plots}))
counts = Counter(str(item.get("plan_name") or "unscoped") for item in plots)
for plan_name in sorted(counts):
    print(f"{plan_name}: {counts[plan_name]}")
PY
```

#### 6) Generate and open the notebook
This step scaffolds the workspace notebook, validates it with marimo, and opens it for interactive review.

```bash
# Generate the workspace-scoped marimo notebook.
pixi run dense notebook generate -c "$CONFIG"

# Validate notebook structure and runtime wiring.
uv run --project "$REPO_ROOT" marimo check "$WORKSPACE/outputs/notebooks/densegen_run_overview.py"

# Open the generated notebook in run mode (opens a browser tab by default).
# If no tab opens in your shell environment, open the printed Notebook URL manually.
pixi run dense notebook run -c "$CONFIG"
```

### Diversity and composition diagnostics
This section gives concrete checks for fixed-element coverage and composition balance after a run.

```bash
# Summarize accepted records by expanded plan name.
uv run --project "$REPO_ROOT" python - <<PY
import pandas as pd

records = pd.read_parquet("$WORKSPACE/outputs/tables/records.parquet", columns=["densegen__plan"])
print(records.groupby("densegen__plan").size().sort_values(ascending=False))
PY

# Summarize fixed-element variant usage from record metadata.
uv run --project "$REPO_ROOT" python - <<PY
import pandas as pd

records = pd.read_parquet(
    "$WORKSPACE/outputs/tables/records.parquet",
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
- `inputs/motif_artifacts/artifact_manifest.json`
- `inputs/motif_artifacts/lacI__pairwise_laci_arac_merged_meme_oops__lacI_WWWTGTGAGCGNDTMACAA.json`
- `inputs/motif_artifacts/araC__pairwise_laci_arac_merged_meme_oops__araC_TAKSRATWWWHHCSMTA.json`
- `outputs/tables/records.parquet`
- `outputs/tables/composition.parquet`
- `outputs/libraries/library_members.parquet`
- `outputs/plots/plot_manifest.json` (gallery source)
- `outputs/plots/stage_a/background_logo.pdf`
- `outputs/plots/stage_b/<plan>/occupancy.pdf`
- `outputs/plots/stage_b/<plan>/tfbs_usage.pdf`
- `outputs/plots/run_health/*.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section lists the highest-frequency issues for this study shape and how to address them.

- Expansion cap error: increase `generation.expansion.max_plans` only after confirming motif-set sizes are intentional.
- Slow convergence in specific plans: increase `runtime.max_seconds_per_plan`, increase Stage-B `library_size`, or run a smaller smoke quota first.
- Background pool shortfall after exclusion: increase `background_pool.sampling.mining.budget.candidates` or widen `background_pool.sampling.length.range`.
- Artifact refresh fails: run the Cruncher `pairwise_laci_arac` runbook through `discover motifs` before `catalog export-densegen`.
- Unexpected motif rejections: inspect `outputs/meta/events.jsonl` and verify `allowlist` selectors match fixed-element components.
- Notebook gallery missing plots: rerun `pixi run dense plot -c "$CONFIG"` and check `outputs/plots/plot_manifest.json`.
