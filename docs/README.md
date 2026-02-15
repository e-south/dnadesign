# Documentation Index

This is the canonical docs entrypoint. Keep curated navigation here.

## Recommended entry flow

1. Local setup and environment sync: `installation.md`
2. Dependency policy: `dependencies.md`
3. Notebook workflow: `notebooks.md`
4. BU SCC operations and job submission: `bu-scc/README.md`
5. Notify CLI usage and watcher operations: `notify/README.md`
6. SGE operations skill package: `bu-scc/sge-hpc-ops/SKILL.md`

## Ownership map

- `bu-scc/`: BU SCC platform policy, scheduler vocabulary, and submit templates.
- `bu-scc/sge-hpc-ops/`: SGE-family operations contract; core guidance is scheduler-generic.
- `notify/`: Notify operator procedures and watcher semantics.

## Edit map

- Change BU queue/resource examples or `qsub` templates: edit `bu-scc/` and `bu-scc/jobs/`.
- Change skill trigger/probe behavior: edit `bu-scc/sge-hpc-ops/SKILL.md`.
- Change dnadesign workload-specific skill examples: edit `bu-scc/sge-hpc-ops/references/workload-dnadesign.md`.
- Change Notify CLI onboarding or watcher runbooks: edit `notify/README.md` and `notify/usr-events.md`.
