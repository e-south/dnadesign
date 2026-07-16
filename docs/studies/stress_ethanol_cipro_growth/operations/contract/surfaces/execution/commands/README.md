## Stress Command Surfaces

Command fragments group read-only checks by owner lane.

- `densegen.yaml`: DenseGen config probes.
- `construct.yaml`: Construct workspace and project validation.
- `infer-validation.yaml`, `infer-dry-run.yaml`, `infer-completion.yaml`: Infer
  config, dry-run, and completion inventory checks.
- `notify/`: Notify profile-doctor and event-resolution checks split by
  subcommand family.
- `latentdna.yaml`: LatentDNA status/snapshot command surface.
- `opal/candidate-table.yaml`: candidate-table contract validation.
- `opal/round0-review.yaml`: read-only campaign, batch, and output checks for round-0 review.
- `opal/campaign-inspection.yaml`: reusable campaign and selection inspection surfaces.
- `opal/synthesis-handoffs.yaml`: explicit artifact-writing synthesis handoff surfaces.
- `opal/densegen-axis-probe.yaml`: study-owned DenseGen axis-probe inspection and publication surfaces.
