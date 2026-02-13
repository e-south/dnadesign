# BU SCC Resource Profiles (`dnadesign`)

Resource defaults derived from:
- `docs/hpc/bu_scc_quickstart.md`
- `docs/hpc/bu_scc_batch_notify.md`
- `docs/hpc/jobs/*.qsub`

Tune from these baselines after measuring runtime, memory, and queue wait.
When feasible, keep walltime requests at or below 12 hours for better shared-cluster scheduling access.

## Profiles

## DenseGen interactive debug

- mode: `qrsh` interactive CPU
- request:
  - `-l h_rt=01:00:00`
  - `-pe omp 8`
  - `-l mem_per_core=8G`
- shared-node hint:
  - BU docs recommend `-pe omp` values `1-4`, `8`, `16`, `28`, `36`
- use for:
  - config validation
  - short smoke runs
  - debugging runtime behavior

## DenseGen batch production (CPU)

- mode: `qsub` CPU batch
- request baseline:
  - `-l h_rt=08:00:00`
  - `-pe omp 16`
  - `-l mem_per_core=8G`
- invariants:
  - `densegen.solver.threads <= 16` (or requested slot count)
  - set solver time limits and per-plan runtime limits in config

## Notify watcher

- mode: `qsub` CPU batch
- request baseline:
  - `-l h_rt=24:00:00`
  - `-pe omp 1`
  - `-l mem_per_core=2G`
- use for:
  - long-running webhook monitoring
  - low-resource event tailing

## Evo2 GPU inference

- mode: `qsub` GPU batch
- request baseline:
  - `-l h_rt=04:00:00`
  - `-pe omp 4`
  - `-l mem_per_core=8G`
  - `-l gpus=1`
  - `-l gpu_c=8.9`
- notes:
  - load compatible CUDA and GCC modules
  - keep this profile for GPU workloads only

## Large transfer / model prefetch

- mode: transfer-node queue
- request baseline:
  - `-l download`
  - `-l h_rt=24:00:00`
  - `-pe omp 1`
- use for:
  - large data/model movement
  - prefetching artifacts
- avoid:
  - heavy compute in transfer queue
