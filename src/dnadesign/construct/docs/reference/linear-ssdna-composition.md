## Linear ssDNA Composition

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-21
**Surface role:** tool-local reference; generic authority for
`linear_ssdna_composition_v1` and producer-owned Folding/BaseRender handoffs

Use `construct compose` when the caller already has the parts and wants
Construct to assemble a declared linear ssDNA product into a local artifact
bundle. This route is not a solver and does not create a Construct workspace.
Retron study docs may dogfood this route, but they do not replace this generic
authority.

### When To Use

Use this route for explicit segment composition:

- ordered physical segments are already selected;
- annotations interpret spans but do not add sequence;
- repeats are declared in the composition contract;
- outputs should remain under a caller-chosen bundle root;
- optional Folding and BaseRender artifacts should be written back into the
  same producer-owned bundle.

Use regular Construct workspaces when the task is template/context realization
from USR-backed anchors, templates, and downstream `construct__*` lineage.

### Commands

```bash
uv run construct compose validate --config <composition.yaml>
uv run construct compose run --config <composition.yaml>
uv run construct compose review --bundle <artifact-bundle>
```

All commands support `--format json`.

`review` also accepts `--nucleotide-font-size-px <float>` when the combined
structure/component-span review needs a pinned nucleotide scale. The review
publisher emits both `visual/reviews/composition_overview.svg` and a
high-resolution `visual/reviews/composition_overview.png`.

### Contract Flow

1. Parse the YAML as `linear_ssdna_composition_v1`.
2. Assemble ordered sequence segments into one canonical component unit.
3. Expand repeats only for the final sequence export.
4. Emit canonical visual and folding evidence for the representative component
   unit, not the repeat-expanded product.
5. Write a manifest that records sequence exports, visual contracts, Folding
   requests or results, and optional review artifacts.

Segments assemble sequence. Annotations interpret spans. A composition config
should never depend on annotation labels to create nucleotides.

### Provenance and display

Segment sources use neutral provenance shapes:

- `literal` for sequence supplied in the config;
- `derived` for a sequence transformed from another declared segment;
- `usr` for a USR dataset record;
- `record` for an external record named by `authority` and `record_id`; and
- `artifact` for a typed artifact named by contract, URI, selector, and
  resolution evidence.

The shared contract does not name a study, provider, or upstream tool. Put that
identity in `authority`, `contract`, or the artifact URI. Old provider-specific
source kinds are invalid rather than silently translated.

Display policy is also declarative. `facts` carries ordered label/value rows
for a review caption. `overview_hidden_components` and
`overview_hidden_annotations` remove caller-selected semantic IDs from the
overview when more precise labels would overlap. Construct does not recognize
study-specific IDs when applying those lists.

### Output Ownership

`construct compose run` writes a producer-owned artifact bundle. Folding is a
stateless service that may read the bundle manifest and write
`secondary_structure_prediction_v2` plus ViennaRNA plot artifacts back into the
same bundle. BaseRender remains the linear/component evidence renderer and
consumes the emitted visual contract or generated job handoff.

Folding bundle consumers should treat Construct's
`linear_ssdna_composition_bundle_manifest_v1` manifest as a
`producer_folding_bundle_v1`-compatible handoff, not as a Construct-private
workspace format.

Do not create persistent Folding workspaces. Do not create one Construct
workspace per ad hoc composition request. If a study needs a registry or
compiler above this layer, keep that code under the study-owned route and pass
only validated composition specs into Construct.

### Failure Posture

- invalid YAML or contract shape fails before writing outputs;
- invalid sequence transforms or span bounds fail during validation;
- output path collisions are governed by the composition config;
- missing required Folding backends fail the configured folding step;
- advisory Folding requests must still emit explicit warning or error state.

### Related Docs

- [Construct CLI reference](cli.md)
- [Construct outputs reference](outputs.md)
- [Folding docs](../../../folding/docs/README.md)
- [Composition boundary decision](../../../../../docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md)
