---
doc_id: bu-scc-job-templates
surface: ops-runbook
owner: dnadesign-maintainers
last_verified: 2026-07-09
---

## BU SCC job templates

These scripts are submit-ready templates for BU SCC SGE jobs:

- `densegen-cpu.qsub`: DenseGen CPU batch run
- `densegen-analysis.qsub`: post-run DenseGen analysis (plots)
- `evo2-gpu-infer.qsub`: Evo2 GPU infer batch run
- `eco1-proteinmpnn-generation-policy.qsub`: Eco1 ProteinMPNN v3 generation-policy array run
- `eco1-colabfold-foldcheck.qsub`: Eco1 fold-check ColabFold smoke/full run
- `permuter-evaluate.qsub`: Permuter workspace evaluate batch run, optionally preceded by `permuter run`
- `notify-watch.qsub`: Notify watcher for USR `.events.log`

### Quick start

Use project (`-P`) and runtime/config overrides at submit time for reusable
templates. The Eco1 ColabFold fold-check template is study-specific and pins
`#$ -P dunlop` so the SCC smoke/full runs fail fast under the intended project.

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--fresh --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
qsub -P <project> \
  -hold_jid <densegen_cpu_job_name_or_id> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/densegen-analysis.qsub
qsub -P <project> \
  -v INFER_CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/evo2-gpu-infer.qsub
qsub -t 1 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
qsub \
  -v DNADESIGN_REPO=<dnadesign_repo>,FOLDCHECK_SEQUENCE_LIMIT=6,COLABFOLD_BATCH=/projectnb/dunlop/esouth/tools/localcolabfold/.pixi/envs/default/bin/colabfold_batch,COLABFOLD_EXTRA_ARGS='--num-models 1' \
  docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub
qsub -P <project> \
  -v PERMUTER_WORKSPACE=<dnadesign_repo>/src/dnadesign/permuter/workspaces/<workspace>/config.yaml,PERMUTER_REF=<ref_name>,PERMUTER_RUN_FIRST=1,PERMUTER_EVALUATE_ARGS='--with smoke:placeholder:log_likelihood' \
  docs/bu-scc/jobs/permuter-evaluate.qsub
qsub -P <project> docs/bu-scc/jobs/notify-watch.qsub
```

### DenseGen CPU submissions

Fresh-mode template run:

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--fresh --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

`densegen-cpu.qsub` command defaults:
- validation: `uv run dense validate-config --probe-solver -c "$DENSEGEN_CONFIG"`
- run: `DENSEGEN_RUN_ARGS` must include exactly one of `--fresh` or `--resume`
- actor tags: `USR_ACTOR_TOOL=densegen`, `USR_ACTOR_RUN_ID=$JOB_ID.$SGE_TASK_ID`
- thread alignment: `OMP_NUM_THREADS=${NSLOTS:-1}`
- runtime trace: `<DENSEGEN_CONFIG directory>/outputs/logs/ops/runtime/dnadesign_densegen_cpu.$JOB_ID.trace.log`
- GUROBI bootstrap defaults:
  - `module load gurobi/10.0.1` when modules are available
  - `GUROBI_HOME=/share/pkg.7/gurobi/10.0.1/install`
  - `GRB_LICENSE_FILE=/usr/local/gurobi/gurobi.lic`
  - `TOKENSERVER=sccsvc.bu.edu`
  - `LD_LIBRARY_PATH=$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}`

Override command args at submit time when needed:
- `DENSEGEN_VALIDATE_ARGS` (example: `--probe-solver`)
- `DENSEGEN_RUN_ARGS` (example: `--resume --extend-quota 8 --no-plot`)

`densegen-analysis.qsub` command defaults:
- analysis chain: `uv run dense plot -c "$DENSEGEN_CONFIG" --only "$DENSEGEN_ANALYSIS_PLOTS"`
- default `DENSEGEN_ANALYSIS_PLOTS`: `source_cohort_concentration,stage_a_sampling_yield,stage_a_pool_diversity,plan_regulator_deployment_heatmap,placement_occupancy_map,retained_pool_coverage_by_regulator,attempt_outcome_timeline,solve_pressure_and_progress`
- fail-fast gate: `DENSEGEN_ANALYSIS_PLOTS` must be non-empty
- preflight gate: only the selected plots that depend on attempts/composition
  require those artifacts to exist as finalized parquet or part files
- read behavior: when part files exist, `dense plot` reads finalized tables
  and part files directly without mutating on-disk artifacts
- optional drilldown/video plots: append ids such as `source_plan_input_heatmap`, `tfbs_concentration_profile`, or `dense_array_showcase_video` only when they are explicitly needed
- optional video plot: include `dense_array_showcase_video` only when FFmpeg is available in the queue environment
- fail-fast gate: if `DENSEGEN_ANALYSIS_PLOTS` includes `dense_array_showcase_video` and `ffmpeg` is unavailable, the job exits with an explicit error

Resume + quota extension submission:

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--resume --extend-quota 8 --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

DenseGen + GUROBI with explicit 12-slot cap:

```bash
qsub -P <project> \
  -pe omp 12 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--fresh --no-plot' \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

Override bootstrap values when needed:

```bash
qsub -P <project> \
  -pe omp 12 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,DENSEGEN_RUN_ARGS='--fresh --no-plot',GUROBI_MODULE=gurobi/10.0.1,GUROBI_HOME=/share/pkg.7/gurobi/10.0.1/install,GRB_LICENSE_FILE=/usr/local/gurobi/gurobi.lic,TOKENSERVER=sccsvc.bu.edu \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

When using GUROBI, keep config aligned with scheduler slots:
- `densegen.solver.threads <= pe omp slots`
- set `densegen.solver.solver_attempt_timeout_seconds` for per-solve limits
- set `densegen.runtime.checkpoint_every` for flush/checkpoint cadence
- keep overall job runtime bounded via scheduler `-l h_rt=...`

For large campaigns with `runtime.round_robin: true`, avoid tiny turn caps:
- raise `densegen.runtime.max_accepted_per_library` so each round-robin turn emits a meaningful batch
- raise `output.usr.chunk_size` to reduce overlay-part fan-out
- use ops runbooks so preflight `usr-overlay-guard` blocks unsafe projected overlay-part growth and compacts existing overlay parts when configured

### Evo2 GPU submissions

```bash
qsub -P <project> \
  -v INFER_CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml,CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/bu-scc/jobs/evo2-gpu-infer.qsub
```

Use that direct submit as the default `evo2_7b` lane. For `evo2_20b`, keep the same template but submit through an ops runbook or pass explicit Hopper resources (`-l gpus=1 -l gpu_c=9.0`) so the scheduler request matches the model contract.

`evo2-gpu-infer.qsub` command defaults:
- fail-fast gate: `INFER_CONFIG` is required
- preflight: `uv run infer validate config --config "$INFER_CONFIG"`
- run: `uv run infer run --config "$INFER_CONFIG" ${INFER_RUN_ARGS:-}`
- actor tags: `USR_ACTOR_TOOL=infer`, `USR_ACTOR_RUN_ID=${OPS_JOB_NAME_SLUG:-$JOB_ID}` with `.SGE_TASK_ID` appended only for real array tasks
- sequence-view sidecar jobs should be submitted through Ops runbooks or `ops runbook fill-infer`; those plans add `infer validate sequence-view-completion` before submit and do not pass `INFER_RUN_ARGS=--overwrite`
- optional legacy reset contract: set `INFER_RUN_ARGS=--overwrite` only when a direct row-writeback Infer run intentionally needs to recompute requested write-back outputs in place

Before first submit on a host, run deterministic environment bootstrap:
- [BU SCC install GPU setup and verification runbook](../setup/install.md#gpu-setup-and-verification-runbook)
- [infer SCC Evo2 GPU environment runbook](../../../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md)

### Eco1 ProteinMPNN generation-policy submissions

The Eco1 generation-policy template is the v3 lane. It consumes request
roots under:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/
```

It maps SGE task ids to the three complete v3 policies:
`distal_scaffold_repack_v1`, `near_dna_rna_acid_free_v1`, and
`combined_near_acid_free_plus_distal_v1`. Each task runs one complete
ProteinMPNN policy. The template does not combine mutations across policy
outputs.

If request sidecars are missing, the template materializes
`generation_policy_manifest.yaml`, `generation_policy_positions.parquet`,
`generation_policy_alphabets.parquet`, and per-policy
`proteinmpnn_request/request_manifest.yaml` before running ProteinMPNN.

Use a one-policy smoke first:

```bash
mkdir -p /project/dunlop/esouth/proteinmpnn/eco1_rt_generation_policies/sge_logs
qsub -t 1 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
```

After the smoke writes `sample_table.parquet` and `candidate_table.parquet` for
the first policy, run the remaining policies:

```bash
qsub -t 2-3 \
  -v DNADESIGN_REPO=<dnadesign_repo>,PROTEINMPNN_ROOT=<dnadesign_repo>/.var/tools/proteinmpnn \
  docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub
```

Use `qsub -t 1-3` only when no policy has already been materialized or when
`ECO1_PROTEINMPNN_OVERWRITE=1` is intentional.

After all policy tasks finish, check that each policy wrote the expected sample
and candidate tables before starting fold checks:

```bash
cd <dnadesign_repo>
uv run python - <<'PY'
from pathlib import Path
import polars as pl

root = Path(
    "src/dnadesign/studies/units/eco1_rt_repack/workspaces/"
    "eco1_rt_conservative_v1/outputs/thread/generation_policies_v3"
)
for policy_id in (
    "distal_scaffold_repack_v1",
    "near_dna_rna_acid_free_v1",
    "combined_near_acid_free_plus_distal_v1",
):
    policy_root = root / policy_id
    sample = policy_root / "sample_table.parquet"
    candidate = policy_root / "candidate_table.parquet"
    sample_rows = pl.read_parquet(sample).height if sample.exists() else "missing"
    candidate_rows = pl.read_parquet(candidate).height if candidate.exists() else "missing"
    print(f"{policy_id}\tsample_rows={sample_rows}\tcandidate_rows={candidate_rows}")
PY
```

To continue the study on a local workstation, pull the generated v3 policy
bundle back into the same ignored output path. This is a generated-artifact
transfer, not a USR dataset sync:

```bash
rsync -avz \
  esouth@scc1.bu.edu:/project/dunlop/esouth/dnadesign/src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/ \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/

mkdir -p src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/sge_logs
rsync -avz \
  esouth@scc1.bu.edu:/project/dunlop/esouth/proteinmpnn/eco1_rt_generation_policies/sge_logs/ \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/sge_logs/
```

This ProteinMPNN generation lane stops after per-policy `sample_table.parquet`
and `candidate_table.parquet` materialization. It does not automatically run
ColabFold. After local pull-back, regenerate the v3 candidate-pool/fold-check
inputs from the policy-provenance tables, then run the local ColabFold path.

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  candidate-pool
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies \
  --repo-root . \
  foldcheck-request
```

For a bounded local smoke run, create a six-record FASTA from the v3 request
before running `colabfold_batch`:

```bash
uv run python -m dnadesign.thread.foldcheck.subset \
  --request-manifest src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_request/foldcheck_request_manifest.yaml \
  --sequence-limit 6 \
  --sequence-start 1 \
  --input-fasta src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_local_runs/smoke_6/input_sequences.fasta \
  --run-manifest src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_local_runs/smoke_6/colabfold_run_manifest.yaml \
  --output-dir src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_local_runs/smoke_6/colabfold_outputs \
  --schema-id eco1_rt.colabfold_local_run_manifest \
  --execution-status planned_local_colabfold_cli
colabfold_batch --num-models 1 \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_local_runs/smoke_6/input_sequences.fasta \
  src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/generation_policies_v3/foldcheck_local_runs/smoke_6/colabfold_outputs
```

### Eco1 ColabFold fold-check submissions

The Eco1 ColabFold template consumes the materialized fold-check request:

```text
src/dnadesign/studies/units/eco1_rt_repack/workspaces/eco1_rt_conservative_v1/outputs/thread/foldcheck_request/foldcheck_request_manifest.yaml
```

The runtime is the ColabFold `colabfold_batch` CLI. On BU SCC that command is
currently exposed by a LocalColabFold pixi environment at:

```text
/projectnb/dunlop/esouth/tools/localcolabfold/.pixi/envs/default/bin/colabfold_batch
```

LocalColabFold is the install path and environment wrapper. It should not be
described as a separate fold model, a hosted API, or a native DeepMind AlphaFold2
run.

Smoke run, WT plus the first five accepted ProteinMPNN candidates:

```bash
mkdir -p /project/dunlop/esouth/foldcheck/eco1_rt/sge_logs
qsub \
  -l h_rt=04:00:00 \
  -v DNADESIGN_REPO=<dnadesign_repo>,FOLDCHECK_SEQUENCE_LIMIT=6,COLABFOLD_BATCH=/projectnb/dunlop/esouth/tools/localcolabfold/.pixi/envs/default/bin/colabfold_batch,COLABFOLD_EXTRA_ARGS='--num-models 1' \
  docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub
```

Full current request, WT plus 96 accepted candidates:

```bash
mkdir -p /project/dunlop/esouth/foldcheck/eco1_rt/sge_logs
qsub \
  -l h_rt=24:00:00 \
  -v DNADESIGN_REPO=<dnadesign_repo>,FOLDCHECK_SEQUENCE_LIMIT=all,COLABFOLD_BATCH=/projectnb/dunlop/esouth/tools/localcolabfold/.pixi/envs/default/bin/colabfold_batch,COLABFOLD_EXTRA_ARGS='--num-models 1',FOLDCHECK_RUN_ROOT=/project/dunlop/esouth/foldcheck/eco1_rt/full_96_<run_id> \
  docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub
```

`eco1-colabfold-foldcheck.qsub` defaults:
- fail-fast gates: request manifest exists, request FASTA exists, output
  directory is empty unless `FOLDCHECK_ALLOW_EXISTING_OUTPUT=1`, and
  `colabfold_batch` is discoverable through `COLABFOLD_BATCH` or the activated
  environment. When `COLABFOLD_BATCH` points into a pixi environment, the
  wrapper prepends the sibling `lib/` directory to `LD_LIBRARY_PATH` so SCC does
  not fall back to the older system `libstdc++`.
- GPU floor: the template requests `gpus=1` and
  `gpu_compute_capability=6.0`. ColabFold on this LocalColabFold environment
  failed on a lower-capability GPU path with DNN initialization errors during
  the expanded design-class run.
- subset control: `FOLDCHECK_SEQUENCE_LIMIT=6` for smoke; use `all` for the
  full 97-record baseline request. For larger expanded requests, pass
  `FOLDCHECK_SHARD_SIZE=<count>` with `qsub -t <start>-<end> -tc <limit>`.
  The wrapper derives `FOLDCHECK_SEQUENCE_START` from `SGE_TASK_ID`, allows a
  shorter final shard, and writes each shard under `FOLDCHECK_RUN_ROOT/shard_N`.
  For one-off manual shards, pass `FOLDCHECK_SEQUENCE_START=<one-based-start>`
  with a bounded `FOLDCHECK_SEQUENCE_LIMIT=<count>`. The template delegates
  FASTA subsetting and run manifest writing to
  `python -m dnadesign.thread.foldcheck.subset`, which validates FASTA ids and
  sequence hashes against the request manifest.
- runtime args: pass `COLABFOLD_EXTRA_ARGS='--num-models 1'` for fast
  preflight smoke runs and the first full coverage screen; reserve heavier
  multi-model checks for selected candidates after this first pass
- durable output root: `FOLDCHECK_RUN_ROOT` defaults to
  `/project/dunlop/esouth/foldcheck/eco1_rt/$JOB_NAME.$JOB_ID`
- log path: array tasks write task-specific stdout/stderr files under
  `/project/dunlop/esouth/foldcheck/eco1_rt/sge_logs/`
- compact run manifest: `colabfold_run_manifest.yaml` records the source
  request hash, selected sequence ids, input FASTA, and output directory

This template deliberately does not parse ColabFold outputs into
`foldcheck_report.parquet`. Normalize completed output directories through the
study wrapper around `dnadesign.thread.adapters.colabfold`:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report \
  --repo-root . \
  --colabfold-output-root /project/dunlop/esouth/foldcheck/eco1_rt/<run_id> \
  --runtime-version <colabfold_version>
```

Raw model directories stay on SCC by default. The compact report can be synced
back after validation.

### Permuter evaluate submissions

```bash
qsub -P <project> \
  -v PERMUTER_WORKSPACE=<dnadesign_repo>/src/dnadesign/permuter/workspaces/<workspace>/config.yaml,PERMUTER_REF=<ref_name>,PERMUTER_RUN_FIRST=1,PERMUTER_EVALUATE_ARGS='--with llr:evo2_llr:log_likelihood_ratio' \
  docs/bu-scc/jobs/permuter-evaluate.qsub
```

`permuter-evaluate.qsub` command defaults:
- fail-fast gates: `PERMUTER_WORKSPACE` and `PERMUTER_REF` are required
- preflight: `uv run permuter workspace validate --workspace "$PERMUTER_WORKSPACE"`
- optional generation: set `PERMUTER_RUN_FIRST=1` to run `permuter run --json` before evaluation
- evaluation: `uv run permuter evaluate --json`; pass explicit metrics with `PERMUTER_EVALUATE_ARGS` or define `evaluate.metrics[]` in the workspace config
- actor tags: `USR_ACTOR_TOOL=permuter`, `USR_ACTOR_RUN_ID=${JOB_ID}` with `.SGE_TASK_ID` appended only for real array tasks
- runtime trace: `<PERMUTER_WORKSPACE directory>/outputs/logs/ops/runtime/dnadesign_permuter_evaluate.$JOB_ID.trace.log` when `PERMUTER_WORKSPACE` is a path; pass `PERMUTER_TRACE_ROOT` for registered scope ids
- output override: pass `PERMUTER_OUT=<output root>` to mirror local `permuter run/evaluate --out`

This wrapper is intentionally a Permuter surface, not an Infer workaround. Evo2 evaluators still call through Permuter's evaluator registry; feature-bundle or USR-sidecar Infer jobs should use the Infer runbook/template directly until the Permuter sidecar-native bridge lands.

### Notify watcher submissions

Preferred mode (profile-driven, secure by default):

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/notify/densegen"
WEBHOOK_FILE="$HOME/.config/dnadesign/notify_webhook.secret"

mkdir -p "$(dirname "$WEBHOOK_FILE")"
touch "$WEBHOOK_FILE"
chmod 600 "$WEBHOOK_FILE"
uv run notify setup webhook \
  --secret-source file \
  --secret-ref "file://$WEBHOOK_FILE"

uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source file \
  --secret-ref "file://$WEBHOOK_FILE" \
  --no-store-webhook \
  --policy densegen

# SCC TLS trust chain for HTTPS webhook delivery.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json",WEBHOOK_FILE="$WEBHOOK_FILE" \
  docs/bu-scc/jobs/notify-watch.qsub
```

Explicit env mode (no profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK,WEBHOOK_FILE="$WEBHOOK_FILE",NOTIFY_TLS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
  docs/bu-scc/jobs/notify-watch.qsub
```

`notify-watch.qsub` mode selection:
- if `NOTIFY_PROFILE` is set, it runs `notify usr-events watch --profile ... --follow`
- otherwise it requires `EVENTS_PATH` or auto-resolves from `NOTIFY_TOOL` + `NOTIFY_CONFIG`
- it accepts future `.events.log` paths and uses `--wait-for-events` for run-before-events startup
- profile mode requires a readable `WEBHOOK_FILE` (watcher loads secret from file each run)
- env mode requires a readable `WEBHOOK_FILE` (watcher loads secret from file each run)
- set `NOTIFY_TLS_CA_BUNDLE` (or `SSL_CERT_FILE`) for HTTPS webhook delivery
- watcher follow mode uses `--on-truncate restart` by default; override with `NOTIFY_ON_TRUNCATE=error|restart`
- watcher polling cadence is configurable via `NOTIFY_POLL_INTERVAL_SECONDS` (default `1.0`)
- env mode requires a policy (`NOTIFY_POLICY`) unless resolver mode (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) sets one
- env mode requires a namespace (`NOTIFY_NAMESPACE`) unless resolver mode (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) sets one
- you can override policy defaults with explicit `NOTIFY_ACTIONS` and `NOTIFY_TOOLS`
- env-mode default state paths are namespaced: `outputs/notify/<namespace>/cursor` and `outputs/notify/<namespace>/spool`

### Local qsub-like smoke harness

Run a real watcher flow locally without SCC scheduler access:

```bash
# Env mode: explicit events/policy/namespace wiring.
docs/bu-scc/jobs/notify-watch-local-smoke.sh \
  --mode env \
  --repo-root "$(pwd)" \
  --workdir /tmp/notify-smoke-env

# Profile mode: setup profile + watcher profile flow.
docs/bu-scc/jobs/notify-watch-local-smoke.sh \
  --mode profile \
  --repo-root "$(pwd)" \
  --workdir /tmp/notify-smoke-profile
```

What it does:
- writes a terminal USR event into a local test events file
- starts a local HTTP capture endpoint (no external Slack posting)
- runs `docs/bu-scc/jobs/notify-watch.qsub` with real `uv run notify ...` commands
- verifies at least one success payload was delivered
- keeps all runtime artifacts under the specified `--workdir`

### Logs

Scheduler stdout and runtime traces:

- most templates write scheduler stdout to `outputs/logs/$JOB_NAME.$JOB_ID.out`
- `permuter-evaluate.qsub` sets scheduler stdout to `/dev/null` and tees its
  own runtime trace after it resolves the workspace
- DenseGen runtime traces: `<DENSEGEN_CONFIG directory>/outputs/logs/ops/runtime/dnadesign_densegen_cpu.$JOB_ID.trace.log`
- Permuter runtime traces: `<PERMUTER_WORKSPACE directory>/outputs/logs/ops/runtime/dnadesign_permuter_evaluate.$JOB_ID.trace.log`

Tail logs:

```bash
tail -f outputs/logs/<job_name>.<job_id>.out
```

### Arrays

For arrays, use `qsub -t ...` and consume `SGE_TASK_ID` inside scripts.

Reference: [BU SCC Batch + Notify runbook: Job arrays](../runbooks/batch-notify.md#5-job-arrays-parameter-sweeps)

### Edit vs submit-time overrides

- Prefer `qsub -P ... -v ... -l ... -pe ...` for run-specific values.
- Keep template scripts stable and versioned.
- Avoid one-off manual edits in production submissions.

### References

- Runbook: [BU SCC Batch + Notify runbook](../runbooks/batch-notify.md)
- BU scheduler docs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>
