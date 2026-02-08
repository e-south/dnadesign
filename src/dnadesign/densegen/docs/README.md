# DenseGen docs

## At a glance

**Intent:** Generate constrained synthetic DNA sequences via staged sampling and optimization.

**When to use:**
- Build controlled libraries for perturbation screens.
- Enforce regulatory composition constraints (required regulators, min counts, fixed promoter elements).
- Generate quota-bounded sequence datasets with audit artifacts.
- Produce run diagnostics (attempts, manifests, plots, reports).

**When not to use:**
- Do not use for canonical storage (use USR).
- Do not use for operational alerts directly (use Notify via USR events).
- Do not use DenseGen runtime events as Notify input.

**Boundary / contracts:**
- DenseGen runtime diagnostics live at `outputs/meta/events.jsonl`.
- USR mutation events live at `<usr_root>/<dataset>/.events.log`.
- Notify consumes USR `.events.log` only.
- Integration boundary details are canonical in `reference/outputs.md#event-streams-and-consumers-densegen-vs-usr`.

**Start here:**
- [Binding-sites baseline demo](demo/demo_binding_sites.md)
- [Three-TF PWM demo](demo/demo_pwm_artifacts.md)
- [DenseGen -> USR -> Notify demo](demo/demo_usr_notify.md)
- [CLI reference (config resolution)](reference/cli.md#config-resolution)

## Start here

Pick the path that matches what you are trying to do:

1) Understand DenseGen end-to-end with the smallest moving parts.
   - [Binding-sites baseline demo](demo/demo_binding_sites.md)

2) Run the canonical PWM workflow (Cruncher -> motif artifacts -> Stage-A PWM mining).
   - [Three-TF PWM demo](demo/demo_pwm_artifacts.md)
   - [Cruncher PWM pipeline](workflows/cruncher_pwm_pipeline.md)

3) Run the full stack locally (DenseGen -> USR -> Notify).
   - [DenseGen -> USR -> Notify demo](demo/demo_usr_notify.md)
   - (For clusters) [DenseGen -> USR -> Notify on HPC](workflows/usr_notify_hpc.md)
   - (BU SCC) [DenseGen -> USR -> Notify on BU SCC](workflows/bu_scc_end_to_end.md)
   - (Platform BU SCC docs) `docs/hpc/bu_scc_install.md` and `docs/hpc/bu_scc_batch_notify.md`

## Event boundary

DenseGen writes runtime diagnostics to `outputs/meta/events.jsonl`.
Notify does not read this stream.

Notify reads USR mutation events at `<usr_root>/<dataset>/.events.log`.
See `reference/outputs.md#event-streams-and-consumers-densegen-vs-usr`.

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
- [Config](reference/config.md)
- [CLI](reference/cli.md)
- [Outputs](reference/outputs.md)
- [Motif artifacts contract](reference/motif_artifacts.md)

## Workflows
- [Cruncher PWM pipeline](workflows/cruncher_pwm_pipeline.md)
- [DenseGen -> USR -> Notify on HPC](workflows/usr_notify_hpc.md)
- [DenseGen -> USR -> Notify on BU SCC](workflows/bu_scc_end_to_end.md)
- Platform BU SCC docs: `docs/hpc/bu_scc_install.md`, `docs/hpc/bu_scc_batch_notify.md`

## Development
- [Dev README](dev/README.md)
- [Architecture](dev/architecture.md)
- [Dev journal](dev/journal.md)
