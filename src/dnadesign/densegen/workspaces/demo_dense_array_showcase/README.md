## demo_dense_array_showcase workspace

Run from this directory:

```bash
# Start a clean generation pass (default mode if omitted).
./runbook.sh --mode fresh
# Continue generation without wiping prior outputs.
./runbook.sh --mode resume
# Rebuild plots/notebook/video from existing outputs only.
./runbook.sh --mode analysis
```

This local demo uses toy TFBS inputs, CBC, fixed-length 100 bp sequences, and a parquet-only output sink. It is meant to showcase dense array packing under three constraint regimes: no fixed anchors, one fixed anchor pair, and two fixed anchor pairs.

- Runbook: [runbook.md](runbook.md)
- Config: [config.yaml](config.yaml)
- All workspaces: [../README.md](../README.md)

### Publication playback

This workspace owns the two public, study-neutral playback tiers:

1. [`playback.yaml`](playback.yaml) teaches generic overlap packing using
   unpadded examples and one neutral binding-site treatment.
2. [`playback-constraints.yaml`](playback-constraints.yaml) teaches fixed
   anchors, their required span, and the RNAP-coupled promoter abstraction
   without introducing study-specific TF identities.

Both recipes consume persisted DenseGen records and publish ignored,
regenerable bundles under `outputs/publication/playback/`. Run either recipe
from the `dnadesign` repository root:

```bash
# Publish one configured endpoint.
uv run python -m dnadesign.densegen.src.integrations.dense_arrays playback-config.yaml
```

Replace `playback-config.yaml` with the recipe path. Pass `--replace` only when
intentionally replacing an existing generated bundle.

DenseGen owns the interpretation of its persisted coordinates. Its record
adapter checks `offset`, `offset_raw` plus padding, and `offset_raw` against
the realized sequence. When a raw-coordinate alternative is required,
`playback_plan_from_densegen_record` and the publisher explicitly pass a
`coordinate_recovered` notice to Dense Arrays; the generic playback engine
does not infer that process claim from metadata names.

Endpoint `labels.overrides` and `presentation.colors_by_label` use exact
persisted placement labels. The publisher resolves both maps to placement IDs.
Color overrides apply to the graph and BaseRender duplex. Label overrides
change fixed-element annotation text when `duplex.fixed_element_annotations`
is `variant`; TFBS features keep their nucleotide sequences as text. Use
explicit legend entries to name TFBS groups.

Colors use opaque `#RRGGBB` values. For example, these fields can be added to a
supported endpoint recipe:

```yaml
presentation:
  color_profile: uniform # Use a neutral default for placements without overrides.
  colors_by_label:
    TF_A: "#9C572B" # Apply this color to every placement with the exact label.
  show_legend: true # Display the explicit entries below.
  legend_entries:
    - key: example_binding # Give this legend entry a unique presentation key.
      label: Example binding site # Author the display text explicitly.
      color: "#9C572B" # Match the corresponding placement color.
```

`legend_entries` are explicit `key`, `label`, and `color` records; the publisher
does not infer biological groups or legend text from label spellings.
Enabling `show_legend` requires at least one explicit entry. Entries may remain
configured with `show_legend: false` to hide the legend temporarily.
BaseRender owns the duplex distance brackets and declares that capability to
the raster renderer, which avoids drawing the same bracket twice. Publication
outputs remain the configured poster, MP4, and JSON bundle; the unused SVG
frame attachment path has been removed.

Playback starts with the full graph, duplex, placement tracks, and configured
annotations in light gray. Placements gain color as the animation advances;
uncovered bases remain gray. The layout and nucleotide size stay fixed within
each scene. The BaseRender frame callback accepts `None` for this pre-placement
view and an integer for the corresponding completed placement.

Motif letters and both sequence strands share centered nucleotide cells.
Fixed-element labels use the same visible text scale as the legend. Placement
reconstruction and ordering details remain in media metadata; failed
constraints remain visible on the figure. Study-specific selection, labels,
and interpretation remain in the owning research study.

### Endpoint schema migration

`audience` accepts `public` or `study_publication` and is preserved in the
bundle manifest. Both values use the same input validation and local output
installation. Study-owned records and media remain in their study workspace.

The publisher accepts `densegen.solution_path_playback_endpoint.v2`. To update
an existing v1 recipe:

1. Set `schema: densegen.solution_path_playback_endpoint.v2`.
2. If `show_legend` is enabled, author `presentation.legend_entries` with
   explicit keys, labels, and colors as above, or set `show_legend: false`.
3. Replace `graph_detail: inset` with `graph_detail: reduced` to retain the
   traversal-only graph. Supported values are `full`, `reduced`, and `none`.
4. Replace `color_profile: secg` with `uniform` or `categorical`, and supply
   any required placement colors through `colors_by_label`.
5. Re-run the configured endpoint to validate and publish its outputs.

An unsupported schema is rejected before the source table is read or outputs
are created. The realized-array adapter and publication bundle have separate
schema versions; this migration changes the endpoint configuration only.
