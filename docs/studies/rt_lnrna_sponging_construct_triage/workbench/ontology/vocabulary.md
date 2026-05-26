## RT-lnRNA Sponging Construct Vocabulary

- Last verified: 2026-05-22
- Owner: dnadesign-maintainers

### Core Terms

| Term | Meaning |
| --- | --- |
| `SyntheticRtLnrnaSpongingConstruct` | One paired lnRNA/msr-msd-payload cassette plus one RT CDS cassette in the fixed dual-expression context. |
| `RtCds` | DNA coding sequence for the reverse transcriptase in the construct subject row. |
| `LnrnaSequence` | DNA representation of the lnRNA cassette used in the construct context. |
| `MsdDesignSpec` | Study/compiler design metadata for the MSD, hairpin, payload, and feasibility details. |
| `YIU` | Study acronym for the YIU-compatible cloning method; in this study it composes Snapback cap and scar-nick stem-base primitives before MSD compilation. |
| `ConstructSlot` | Named placement slot owned by Construct, for example `lnrna` or `rt_cds`, with template span, source field, emitted span, and orientation-aware bounds. |
| `ConstructProjectionManifest` | Study-owned fixture that binds candidate RT/lnRNA source authority to Construct slots and expected context views. |
| `DualExpressionConstruct` | Concrete sequence instance emitted by Construct. |
| `ConstructContextView` | Declared construct-context sequence view with orientation, coordinates, spans, and pooling intent. |
| `InferFeatureAlias` | Pointer to model-derived feature vectors and aliases. |
| `AbundancePriorOverlay` | Literature/source abundance prior, not a sponging assay label. |
| `SpongingAssayObservation` | Future Reader-owned lab TF-sponging label rows with source-scoped SPOP numerics. |

### Naming Rules

- Use `ConstructSlot` for Construct placement semantics. The v1 slot ids are
  `lnrna` and `rt_cds`.
- Use `ConstructProjectionManifest` for the study-owned mapping from candidate
  source authority into Construct slots and expected views.
- Use Construct realization for the runtime act of emitting sequences, and
  construct context view materialization for writing USR sequence-view rows.
- Use `lab_anchor`, `working_anchor`, and `failed_anchor` only as source-history
  or candidate-control labels. They are not Construct part roles.
- Treat retron26 and retron43 as representative GenBank catalog rows, not as a
  separate GenBank overlay partition.
- Keep Khan abundance, Crawford abundance, and Reader SPOP numerics
  source-scoped. Ordinal bins and categorical hues are metadata views, not
  shared-scale numeric labels.
- Keep `anchor_mean`, `anchor_start_0`, `anchor_end_0`, and
  `construct_output_anchor_part` only where the USR/Infer/Construct APIs require
  a single pooled span. In study prose, describe the biological object as a
  named slot or candidate role, not as a second Construct anchor.

### Excluded Core Fields

The candidate core schema must not require these fields:

- `perturbation_class`
- `cloning_constraint_set`
- `representation_result`

Use `source_basis`, optional `variant_derivation`, `MsdDesignSpec`,
`ConstructContextView`, and `InferFeatureAlias` instead.
