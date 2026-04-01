## demo_monotypic_tetr Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/

**Regulators**
- [tetR]

**Purpose**
- Run a single-TF multiplicity demo that optimizes a 19 bp sequence for two offset-distinct TetR placements from the Westmann MITOMI ddG matrix. The raw `PO A T C G` table is ingested through an explicit `DDG_TABLE` parser module, converted into a probability PWM with a Boltzmann normalization step, exported as a minimal MEME artifact for downstream tool interoperability, and then carried through the standard `lock -> parse -> sample -> analyze -> logos` lifecycle. This workspace uses the same 8-chain `15k/90k` occurrence-aware sweep posture as the monotypic BaeR demo, but asks for two selected TetR copies instead of four. Export, studies, and portfolio aggregation remain gated for this demo because those downstream readers still require the representative-hit contract.

**Run This Single Command**

Run this single command to do everything below:

    uv run cruncher workspaces run --runbook configs/runbook.yaml

Quick smoke path (main lifecycle only):

    uv run cruncher workspaces run --runbook configs/runbook.yaml --step reset_workspace --step config_summary --step fetch_motifs_westmann --step lock_targets --step parse_run --step sample_run --step analyze_summary --step show_sample_outputs --step render_logos --step export_meme

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr
    CONFIG="$PWD/configs/config.yaml"
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    # Standard transient cleanup command for workspace hygiene.
    cruncher workspaces reset --root . --confirm
    # Optional config sanity check.
    cruncher config summary -c "$CONFIG"
    # Fetch the TetR motif from the local raw ddG table through the explicit parser-module contract.
    cruncher fetch motifs --source westmann_tetr_mitomi --tf tetR --update -c "$CONFIG"
    # Freeze resolved motif provenance for the TetR set.
    cruncher lock -c "$CONFIG"
    cruncher parse --force-overwrite -c "$CONFIG"
    cruncher sample --force-overwrite -c "$CONFIG"
    # Analyze the occurrence-aware TetR run and render the standard static plot suite, including multi-offset elites_showcase panels.
    cruncher analyze --summary -c "$CONFIG"
    # Occurrence-aware runs write elites_objective_scores.parquet and elites_occurrences.parquet instead of representative-hit tables.
    cruncher runs show outputs -c "$CONFIG"
    cruncher catalog logos --source westmann_tetr_mitomi --set 1 -c "$CONFIG"
    # Export the normalized TetR PWM as a minimal MEME file after the main outputs tree is stable.
    cruncher catalog export-meme --set 1 --source westmann_tetr_mitomi -c "$CONFIG"

### Optional output checks

    find outputs/optimize/tables -maxdepth 1 -type f | sort

    uv run python - <<'PY'
    import pandas as pd
    scores = pd.read_parquet("outputs/optimize/tables/elites_objective_scores.parquet")
    occ = pd.read_parquet("outputs/optimize/tables/elites_occurrences.parquet")
    print(scores[["elite_id", "objective_id", "scalar_score", "requested_copies", "selected_copies"]])
    print(occ[["elite_id", "objective_id", "occurrence_rank", "start", "end", "strand", "scaled_score"]])
    PY

    find outputs/artifacts/meme -maxdepth 1 -type f | sort
    find outputs/plots -maxdepth 1 -type f | sort
