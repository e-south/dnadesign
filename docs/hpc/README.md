# HPC docs

## Start here

- [BU SCC Quickstart](bu_scc_quickstart.md)
- [BU SCC Install bootstrap](bu_scc_install.md)
- [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)
- [Job templates](jobs/README.md)
- [BU SCC ops cheat sheet (interactive + batch commands)](bu_scc_ops_cheatsheet.md)
- [BU SCC ops skill for agents](skills/bu_scc_ops/SKILL.md)

## Codex skill activation

Install BU SCC Ops as a first-class local Codex skill:

```bash
mkdir -p ~/.codex/skills
ln -sfn /project/dunlop/esouth/dnadesign/docs/hpc/skills/bu_scc_ops ~/.codex/skills/bu_scc_ops
```

## Decision guide

- I need to install and verify CUDA/solver stack:
  [BU SCC Install bootstrap](bu_scc_install.md)
- I need to submit a batch job or array:
  [BU SCC Batch + Notify runbook](bu_scc_batch_notify.md)
- I need webhook notifications for USR events:
  [Notify USR events operator manual](../notify/usr_events.md)
- I need large model/dataset transfers:
  [BU SCC Batch + Notify runbook: Large downloads and transfer](bu_scc_batch_notify.md#5-large-downloads-and-datasetmodel-transfer)
