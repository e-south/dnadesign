---
doc_id: study-retron-hairpin-design-route-compiler-msd-design-references
surface: study-route-detail
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: data-plane
owner_boundary: studies/retron_hairpin_design
surface_role: compiler-materialization-entrypoint
current_state: compiler-ready
entry_artifact: retron_msd_label_or_compiler_spec
exit_artifact: msd_design_catalog_v1_or_msd_single_unit_sequence_bundle_v1
---

## MSD Design References Route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this route when the task starts from a lab-facing construct label and needs
a frozen design reference, one MSD sequence unit, Reader joins, or persistent
design-set provenance.

### Route Contract

- Type: `route`
- Plane: `data-plane`
- Surface role: `compiler-materialization-entrypoint`
- Owner-boundary: `studies/retron_hairpin_design`
- Current state: `compiler-ready`
- Entry artifact: lab-facing MSD construct label or `retron_msd_compiler_spec_v1`
- Exit artifact: `msd_design_catalog_v1` or `msd_single_unit_sequence_bundle_v1`
- Registry: `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`
- Authoritative cohort:
  `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml`
- Convenience label input:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`
- Cap source lookup:
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`
- Full cohort materialization spec:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml`
- Non-default S0-control materialization spec:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_non_ligatable_s0_control_spec.yaml`
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

Materialize the full checked-in cohort from explicit 5'->3' cap/foldback
segments:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app materialize \
  --spec docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml \
  --out-dir /tmp/dnadesign_retron_msd_sequences \
  --render-format png \
  --format json
```

Materialize the explicit non-default S0-control request:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app materialize \
  --spec docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_non_ligatable_s0_control_spec.yaml \
  --out-dir docs/studies/retron_hairpin_design/workbench/outputs/retron-msd-177-194-non-ligatable-s0-control-20260518 \
  --render-format png \
  --format json
```

`C###` cap IDs are not implicit de033 lookups; materialization specs must
provide each cap as a literal 5'->3' sequence or an explicit public primitive
source. If topology is absent, the compiler emits the whole supplied
cap/foldback segment and omits subsection labels.

Non-ligatable S0 controls are not the default. Use
`--allow-non-ligatable-s0` or `allow_non_ligatable_s0: true` only when the
request intentionally preserves an `S0!=M` control. Profile drift still fails,
and emitted references mark `scar_nick.s0_match_required=false`.

### Boundaries

This compiler is intentionally study-owned and is not registered as a top-level
`uv run retron-msd` tool. It parses lab-facing labels or typed
`retron_msd_compiler_spec_v1` design parts into the same trusted structure:
`construct_id`, payload/target, cap id, left/right scar-nick bases, and
optional profile code. It recomputes the `S3/S2/S1/S0` profile and fails fast
if the provided code drifts or `S0` is not ligatable without explicit opt-in.

Compiler specs may point at solved Snapback foldback primitives or scar-nick
stem-base primitives only through public `dnadesign.cruncher.snapback` and
`dnadesign.cruncher.scar_nick` APIs. `selector.mode=rank` is the current
explicit combination surface; rank lists, ranges, and all-hit selectors must
fail until a deliberate expansion contract exists.

`C###` cap IDs are symbolic. Known cap sequences live in
`compiler/catalog/msd_cap_sources.yaml`: `C26=AGGC`,
`C43=tCCTCAGcccGCTGAGGa`, and the selected `C172-C176` de033 source labels.
The compiler does not infer de033 sequence or topology from an id pattern.
Sequence-producing specs must provide a literal 5'->3' cap sequence or an
explicit public primitive source. Topology is only needed for subsection
annotations or topology-specific claims.

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
