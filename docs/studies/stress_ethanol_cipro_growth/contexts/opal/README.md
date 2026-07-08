## Stress OPAL Context

Use this lane for durable shared candidate-table and observed-label semantics.
Use `../../routes/decision/opal/README.md` for first-hop OPAL routing and
`../../routes/decision/opal/campaign-commands.md` for command examples.

- [Candidate table](candidate-table.md): shared USR candidate-table and label
  source contract.
- [Measured reader vec8 batch0 staging](measured-reader-vec8-batch0.md):
  campaign-local round-0 label inputs built from measured reader SFXI `vec8`
  records, with a separate gate for shared observed-label sidecar writes.
- [DenseGen TFBS learnability probe v1](densegen-tfbs-learnability-probe-v1.md):
  retained v1 contract for scalar TF family content and slot-position
  synthetic-control campaigns. Current realized profile boundaries live in the
  source package README and profile registry.
- [DenseGen motif QA K12/S3 v1](densegen-motif-qa-k12-s3-v1.md): historical
  K12, three-seed, trajectory-based motif-composition QA benchmark and
  execution precedent.
- [DenseGen axis probe v0](densegen-axis-probe-v0.md): scratch-only
  historical K6 synthetic-oracle probe for OPAL/LatentDNA readiness.
