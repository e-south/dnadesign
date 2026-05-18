## Stress Command Surfaces

Command fragments group read-only checks by owner lane.

- `densegen.yaml`: DenseGen config probes.
- `construct.yaml`: Construct workspace and project validation.
- `infer-validation.yaml`, `infer-dry-run.yaml`, `infer-completion.yaml`: Infer
  config, dry-run, and completion inventory checks.
- `notify/`: Notify profile-doctor and event-resolution checks split by
  subcommand family.
- `latentdna.yaml`: LatentDNA status/snapshot command surface.
