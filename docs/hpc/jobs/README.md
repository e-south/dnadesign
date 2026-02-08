# BU SCC job templates

These scripts are submit-ready templates for BU SCC SGE jobs:

- `bu_scc_densegen_cpu.qsub`: DenseGen CPU batch run
- `bu_scc_evo2_gpu_infer.qsub`: Evo2 GPU smoke/inference job shell
- `bu_scc_notify_watch.qsub`: Notify watcher for USR `.events.log`

## Submit pattern

Pass project and per-job overrides at submit time:

```bash
qsub -P <project> docs/hpc/jobs/bu_scc_densegen_cpu.qsub
qsub -P <project> docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
qsub -P <project> docs/hpc/jobs/bu_scc_notify_watch.qsub
```

## Common overrides

DenseGen CPU:

```bash
qsub -P <project> \
  -v DENSEGEN_CONFIG=/abs/path/to/config.yaml \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

Evo2 GPU:

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

Notify watcher:

```bash
qsub -P <project> \
  -v EVENTS_PATH=/abs/path/to/.events.log,CURSOR_PATH=/projectnb/<project>/$USER/notify/demo.cursor,SPOOL_DIR=/projectnb/<project>/$USER/notify/spool,WEBHOOK_ENV=DENSEGEN_WEBHOOK \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

## Logs

Each script writes logs to:

- `outputs/logs/$JOB_NAME.$JOB_ID.out`

Tail logs:

```bash
tail -f outputs/logs/<job_name>.<job_id>.out
```

## Arrays

For arrays, use `qsub -t ...` and consume `SGE_TASK_ID` inside scripts.

Reference: [BU SCC Batch + Notify runbook: Job arrays](../bu_scc_batch_notify.md#3-job-arrays-parameter-sweeps)

## Edit vs pass at submit time

- Prefer `qsub -P ... -v ...` for run-specific values.
- Keep scripts stable and versioned; avoid one-off manual edits in production.
