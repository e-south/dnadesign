# DenseGen docs

This docs hub helps you choose the right path quickly.

If you only need one answer:
- use demos for step-by-step runs
- use guides for concepts
- use reference pages for exact contracts

## Contents
- [At a glance](#at-a-glance)
- [Read order](#read-order)
- [Runtime subprocess flow](#runtime-subprocess-flow)
- [Event boundary](#event-boundary)
- [Demos](#demos)
- [Guides](#guides)
- [Reference](#reference)
- [Workflows](#workflows)
- [Development](#development)

## At a glance

**Intent:** generate constrained synthetic DNA sequences with auditable outputs.

Use DenseGen when you need:
- controlled library generation with explicit constraints
- repeatable run state and resume behavior
- diagnostics and report artifacts

Current config schema version: `2.9`.

Do not use DenseGen for:
- canonical dataset ownership (use USR)
- alert/webhook delivery (use Notify)

## Read order

1. Minimal run with low moving parts:
   - [Binding-sites baseline demo](demo/demo_binding_sites.md)
2. Canonical PWM path:
   - [Three-TF PWM demo](demo/demo_pwm_artifacts.md)
   - [Cruncher PWM pipeline](workflows/cruncher_pwm_pipeline.md)
3. Full stack integration:
   - [DenseGen -> USR -> Notify demo](demo/demo_usr_notify.md)
   - [DenseGen -> USR -> Notify on HPC](workflows/usr_notify_hpc.md)
   - [DenseGen -> USR -> Notify on BU SCC](workflows/bu_scc_end_to_end.md)
   - [docs/hpc/bu_scc_install.md](../../../../docs/hpc/bu_scc_install.md)
   - [docs/hpc/bu_scc_batch_notify.md](../../../../docs/hpc/bu_scc_batch_notify.md)

## Runtime subprocess flow

DenseGen has four runtime subprocesses with clear boundaries:

1. **Stage-A pool build**: sample/ingest candidate sites per input, score/filter, and write pool artifacts.
2. **Stage-B library build**: create plan-scoped solver libraries from Stage-A pools.
3. **Solve to quota**: generate accepted arrays under constraints and runtime limits.
4. **Post-run rendering**: generate plots and reports from written tables/manifests.

Command mapping:
- `dense stage-a build-pool`
- `dense stage-b build-libraries`
- `dense run`
- `dense plot` and `dense report`

`dense run` orchestrates this pipeline end-to-end and auto-builds missing Stage-A/Stage-B artifacts.

## Event boundary

DenseGen and Notify intentionally use different event streams:

- DenseGen runtime telemetry: `outputs/meta/events.jsonl`
- Notify input stream: USR `<usr_root>/<dataset>/.events.log`

Canonical contract details:
- [reference/outputs.md#event-streams-and-consumers-densegen-vs-usr](reference/outputs.md#event-streams-and-consumers-densegen-vs-usr)

---

## Demos
- [Demo flows index](demo/README.md)
- [Binding-sites baseline demo](demo/demo_binding_sites.md)
- [Three-TF PWM demo](demo/demo_pwm_artifacts.md)
- [DenseGen -> USR -> Notify demo](demo/demo_usr_notify.md)

## Guides
- [Workspace](guide/workspace.md)
- [Inputs](guide/inputs.md)
- [Sampling](guide/sampling.md)
- [Generation](guide/generation.md)
- [Postprocess](guide/postprocess.md)
- [Outputs + metadata](guide/outputs-metadata.md)

## Reference
- [CLI](reference/cli.md)
- [Config](reference/config.md)
- [Outputs](reference/outputs.md)
- [Motif artifacts contract](reference/motif_artifacts.md)

## Workflows
- [Cruncher PWM pipeline](workflows/cruncher_pwm_pipeline.md)
- [DenseGen -> USR -> Notify on HPC](workflows/usr_notify_hpc.md)
- [DenseGen -> USR -> Notify on BU SCC](workflows/bu_scc_end_to_end.md)

## Development
- [Dev README](dev/README.md)
- [Architecture](dev/architecture.md)
- [Dev journal](dev/journal.md)
