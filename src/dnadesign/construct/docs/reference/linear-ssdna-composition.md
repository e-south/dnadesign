<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](../README.md)

# Compose a linear ssDNA sequence

Use `construct compose` to join selected DNA segments in a declared order and
save the sequence, annotations and provenance together. It writes a local
artifact bundle without requiring a workspace or SeqLedger dataset.

For placement within a larger template, use a
[Construct workspace](../../workspaces/README.md) instead.

## Try a small example

With the [Construct environment active](../getting-started.md#install-and-activate),
save this configuration as `composition.yaml` in a new directory:

```yaml
contract: linear_ssdna_composition_v1
schema_version: 1
composition_id: synthetic_example
units:
  - unit_id: unit
    repeat_count: 1
    segments:
      - segment_id: left
        sequence: AAAA
      - segment_id: payload
        sequence: ACGT
      - segment_id: right
        sequence: TTTT
output:
  artifact_bundle: artifacts/synthetic_example
```

Validate the segments, then write their concatenated sequence:

```bash
# Check the composition without writing outputs.
construct compose validate --config composition.yaml
# Write FASTA, GenBank, annotations and the bundle manifest.
construct compose run --config composition.yaml
```

The result is the 12-base sequence `AAAAACGTTTTT` in
`artifacts/synthetic_example/`, relative to the configuration file. Inspect
`sequence.fa`, `sequence.gb` or `assembled_sequence.json`. Use
`--format json` on either command to read its summary in a script.

## Add a structure review

For a bundle that includes the required structure and component-span
artifacts, run:

```bash
# Replace the path with a bundle containing Folding and BaseRender outputs.
construct compose review --bundle path/to/bundle
```

The minimal example above does not request those artifacts. See
[Folding](../../../folding/docs/README.md) for structure prediction and
[BaseRender](../../../baserender/README.md) for component visualization.

`review` also accepts `--nucleotide-font-size-px <float>` when the combined
structure/component-span review needs a fixed nucleotide scale. The review
publisher emits both `visual/reviews/composition_overview.svg` and a
high-resolution `visual/reviews/composition_overview.png`.

## How the sequence is assembled

1. Parse the YAML as `linear_ssdna_composition_v1`.
2. Assemble ordered sequence segments into one canonical component unit.
3. Expand repeats only for the final sequence export.
4. Emit canonical visual and folding evidence for the representative component
   unit.
5. Write a manifest that records sequence exports, visual contracts, Folding
   requests or results, and optional review artifacts.

Segments assemble sequence. Annotations interpret spans. A composition config
should never depend on annotation labels to create nucleotides.

## Provenance and display

Segment sources use neutral provenance shapes:

- `literal` for sequence supplied in the config;
- `derived` for a sequence transformed from another declared segment;
- `usr` for a SeqLedger dataset record;
- `record` for an external record named by `authority` and `record_id`; and
- `artifact` for a typed artifact named by contract, URI, selector, and
  resolution evidence.

Record the source identity in `authority`, `contract` or the artifact URI,
using the fields for the selected source kind.

Display policy is also declarative. `facts` carries ordered label/value rows
for a review caption. `overview_hidden_components` and
`overview_hidden_annotations` remove caller-selected semantic IDs from the
overview when more precise labels would overlap. Construct does not recognize
study-specific IDs when applying those lists.

## Outputs and tool integration

`construct compose run` writes sequence exports and a manifest into the
configured directory. Folding can read this bundle and add
`secondary_structure_prediction_v2` results and ViennaRNA plots. BaseRender
uses the emitted visual contract or generated job configuration to render the
sequence components.

For integrations, the `linear_ssdna_composition_bundle_manifest_v1` manifest
implements the `producer_folding_bundle_v1` handoff. Running composition again
at the same path rewrites generated outputs; choose a new `artifact_bundle`
path to preserve an earlier result.

## Validation and failures

- invalid YAML or contract shape fails before writing outputs;
- invalid sequence transforms or span bounds fail during validation;
- missing required Folding backends fail the configured folding step;
- advisory Folding requests must still emit explicit warning or error state.

## Related references

- [Construct CLI reference](cli.md)
- [Construct outputs reference](outputs.md)
- [Folding docs](../../../folding/docs/README.md)
- [Composition boundary decision](../../../../../docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md)
