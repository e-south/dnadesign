## MSD Design References Route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this route when the task starts from a lab-facing construct label and needs
a frozen design reference, one MSD sequence unit, Reader joins, or persistent
design-set provenance.

### Route Contract

- Type: `study-contract`
- Plane: `data-plane`
- Surface role: `record-plane design-reference-normalization`
- Owner-boundary: `studies/retron_hairpin_design`
- Current state: `compiler-ready`
- Registry: `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`
- Authoritative cohort:
  `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml`
- Convenience label input:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`
- Public module: `src/dnadesign/studies/studies/retron_hairpin_design/interfaces/cli/app.py`
- Typed compiler spec: `retron_msd_compiler_spec_v1`

### Commands

Lint one label:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app lint \
  --id "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"
```

Lint a typed spec:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app lint \
  --spec path/to/retron_msd_compiler_spec.yaml \
  --format json
```

Compile the current workbench-backed label input:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app compile \
  --input docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt \
  --out-dir /tmp/dnadesign_retron_msd_design_references \
  --format json
```

Materialize single-unit sequence artifacts after concrete sequences are
available:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app materialize \
  --input docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt \
  --out-dir /tmp/dnadesign_retron_msd_sequences \
  --payload-sequence TetR=<payload-sequence> \
  --cap-sequence C26=<cap-sequence> \
  --cap-sequence C172=<cap-sequence> \
  --render-format png \
  --format json
```

### Boundaries

This compiler is intentionally study-owned and is not registered as a top-level
`uv run retron-msd` tool. It parses lab-facing labels or typed
`retron_msd_compiler_spec_v1` design parts into the same trusted structure:
`construct_id`, payload/target, cap id, left/right scar-nick bases, and
optional profile code. It recomputes the `S3/S2/S1/S0` profile and fails fast
if the provided code drifts or `S0` is not ligatable.

Compiler specs may point at solved Snapback foldback primitives or scar-nick
stem-base primitives only through public `dnadesign.cruncher.snapback` and
`dnadesign.cruncher.scar_nick` APIs. `selector.mode=rank` is the current
explicit combination surface; rank lists, ranges, and all-hit selectors must
fail until a deliberate expansion contract exists.

The emitted `msd_design_catalog_v1` is the Reader-facing bridge. Reader should
not parse Construct, Folding, BaseRender, or Cruncher internals. Ad hoc compiles
write to explicit transient directories; Reader-linked runs snapshot the same
shallow bundle into the owning Reader experiment `inputs/designs/` directory.

### Output Layout

Reference-only compile bundles contain `README.md`, `manifest.json`,
`msd_design_catalog_v1.json`, `reference_index.tsv`, and flat
`references/*.msd_design_reference_v1.json` files.

Materialized sequence bundles keep the top level to `README.md`, `manifest/`,
and `variants/`. Bundle manifests live under `manifest/bundle/`, catalogs and
frozen references under `manifest/catalog/`, indexes under `manifest/indexes/`,
and generated composition configs under `manifest/configs/composition/`.

Each `variants/<msd_design_id>/` directory groups forward/reverse-complement
GenBank and FASTA under `sequences/`; `secondary_structure.native.png`,
`composition_overview.svg`, and `composition_overview.png` under `plots/`;
curated metadata under semantic `manifest/` groups; and raw producer output
under `runtime/construct/`.

### Workbench Provenance

Use `workbench/design_sets/` for persistent cohort meaning and
`workbench/provenance/compiler_runs/` for the statement that the study compiler was run on
that cohort. Do not treat generated transient compiler output as the
authoritative experimental notebook.
