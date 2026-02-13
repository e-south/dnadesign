# Documentation Index

This page is the canonical docs navigation.
Keep curated docs links here and avoid duplicating curated navigation elsewhere.

## Start by task

- Local workstation setup and environment sync: `installation.md`
- Dependency maintenance policy: `dependencies.md`
- Notebook workflow: `notebooks.md`
- BU SCC operations and job submission: `bu-scc/README.md`
- Notify CLI usage and watcher operations: `notify/README.md`
- SGE operations skill package: `bu-scc/sge-hpc-ops/SKILL.md`

## Semantic ownership (single source rules)

- `bu-scc/`: BU SCC platform policy, scheduler vocabulary, and submit templates.
- `bu-scc/sge-hpc-ops/`: SGE-family operations contract; core guidance is scheduler-generic.
- `notify/`: Notify operator procedures and watcher semantics.

## Edit here for changes

- Change BU queue/resource examples or `qsub` templates: edit `bu-scc/` and `bu-scc/jobs/`.
- Change skill trigger/probe behavior: edit `bu-scc/sge-hpc-ops/SKILL.md`.
- Change dnadesign workload-specific skill examples: edit `bu-scc/sge-hpc-ops/references/workload-dnadesign.md`.
- Change Notify CLI onboarding or watcher runbooks: edit `notify/README.md` and `notify/usr-events.md`.
