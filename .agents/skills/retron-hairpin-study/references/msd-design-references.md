# MSD Genetic Compiler

Use this reference when a user provides a Retron MSD shorthand ID or enough
parts to compile a design reference.

## Input Completeness

Minimum complete reference:

- payload or target id
- cap id
- left base
- right base
- repeat count when sequence assembly is requested
- optional profile code, which must match the computed `S3/S2/S1/S0` profile

If the user provides a full lab-facing label, parse those fields from the label.
If any primitive is missing, route instead of guessing.

## Boundary

- Study-owned package: `dnadesign.studies.retron_hairpin_design`.
- Registry: `docs/studies/retron_hairpin_design/msd_design_registry.yaml`.
- Selected labels: `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`.
- Do not expose a top-level `retron-msd` script.
- Do not create Construct or Folding workspaces per ID.
- Snapback and scar-nick solve missing primitive parts; the compiler validates
  and assembles selected parts into contracts.
- Reader consumes `msd_design_catalog_v1`, not live dnadesign internals.

## Commands

Lint one label:

```bash
uv run python -m dnadesign.studies.retron_hairpin_design.cli lint \
  --id "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM" \
  --format json
```

Compile the selected hit list:

```bash
uv run python -m dnadesign.studies.retron_hairpin_design.cli compile \
  --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt \
  --out-dir /tmp/dnadesign_retron_msd_design_references \
  --format json
```

## Fail-Fast Semantics

- Reject malformed labels.
- Recompute scar-nick profile from left/right bases.
- Reject profile drift.
- Reject non-ligatable `S0`.
- Reject unknown payload/cap registry entries.
- Keep artifact paths and sequence digests nullable until concrete generated
  artifacts are attached.

## Output Posture

Ad hoc outputs belong in explicit transient directories such as
`/tmp/dnadesign_retron_msd_*`. Reader-linked outputs should later be copied
into the owning Reader experiment `inputs/designs/` directory. Do not commit
generated catalogs or visual/sequence artifacts unless the user explicitly asks.

## Service Handoff

- Construct: assemble a selected design into sequence artifacts.
- Folding: run secondary-structure QA from explicit files or producer bundle.
- BaseRender: render linear/component visual contracts.
- ViennaRNA plotting: fold-layout visualization through Folding, not
  BaseRender.
