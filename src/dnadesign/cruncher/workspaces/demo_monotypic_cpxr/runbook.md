## demo_monotypic_cpxr Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_monotypic_cpxr/

**Regulators**
- [cpxR]

**Purpose**
- Run a single-TF multiplicity demo that optimizes an 18 bp sequence for three offset-distinct CpxR placements, merging the checked-in local MEME motif evidence with RegulonDB site evidence before MEME OOPS discovery. The workspace uses the same 8-chain `15k/90k` occurrence-aware sweep posture as the packaged monotypic demos, persists one exemplar elite, and renders the standard static plot suite including the multi-offset elite showcase. Because multiplicity disables representative-hit fast paths, Export, studies, and portfolio aggregation remain gated.

**Run This Single Command**

Run this single command to do everything below:

    uv run cruncher workspaces run --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_monotypic_cpxr
    CONFIG="$PWD/configs/config.yaml"
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    # Standard transient cleanup command for workspace hygiene.
    cruncher workspaces reset --root . --confirm
    # Optional config sanity check.
    cruncher config summary -c "$CONFIG"
    # Merge local MEME-derived CpxR sites with RegulonDB evidence before discovery.
    cruncher fetch sites --source demo_local_meme --tf cpxR --update -c "$CONFIG"
    cruncher fetch sites --source regulondb --tf cpxR --update -c "$CONFIG"
    # Discover motifs from merged site evidence. This step merges all fetched site sets across sources and runs MEME OOPS into the workspace-specific source id.
    cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_cpxr_multiplicity_meme_oops -c "$CONFIG"
    # Freeze resolved motif/site provenance for this set.
    # If you change catalog.source_preference or discovery --source-id, re-run cruncher lock -c "$CONFIG" before parse.
    cruncher lock -c "$CONFIG"
    cruncher parse --force-overwrite -c "$CONFIG"
    cruncher sample --force-overwrite -c "$CONFIG"
    # Occurrence-aware analysis now renders the standard static plot suite, including multi-offset elites_showcase panels.
    cruncher analyze --summary -c "$CONFIG"
    # Occurrence-aware runs write elites_objective_scores.parquet and elites_occurrences.parquet instead of representative-hit tables.
    cruncher runs show outputs -c "$CONFIG"
    cruncher catalog logos --source demo_cpxr_multiplicity_meme_oops --set 1 -c "$CONFIG"
    # Validate and render the sample-backed YIU payload bundle under outputs/plots, then inspect that bundle.
    cruncher yiu validate --spec configs/yiu/cpxr_monotypic_hit.yiu.yaml
    cruncher yiu render --spec configs/yiu/cpxr_monotypic_hit.yiu.yaml --force-overwrite --emit-renders
    cruncher yiu show --bundle outputs/plots/yiu__cpxr_monotypic_hit

### Optional output checks

    find outputs/optimize/tables -maxdepth 1 -type f | sort

    uv run python - <<'PY'
    import pandas as pd
    scores = pd.read_parquet("outputs/optimize/tables/elites_objective_scores.parquet")
    occ = pd.read_parquet("outputs/optimize/tables/elites_occurrences.parquet")
    print(scores[["elite_id", "objective_id", "scalar_score", "requested_copies", "selected_copies"]])
    print(occ[["elite_id", "objective_id", "occurrence_rank", "start", "end", "strand", "scaled_score"]])
    PY

    find outputs/plots -maxdepth 1 -type f | sort
    find outputs/plots/yiu__cpxr_monotypic_hit -maxdepth 1 -type f | sort
