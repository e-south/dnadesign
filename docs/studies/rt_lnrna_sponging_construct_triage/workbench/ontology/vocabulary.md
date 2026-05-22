## RT-lnRNA Sponging Construct Vocabulary

- Last verified: 2026-05-22
- Owner: dnadesign-maintainers

### Core Terms

| Term | Meaning |
| --- | --- |
| `SyntheticRtLnrnaSpongingConstruct` | One paired lnRNA/msr-msd-payload cassette plus one RT CDS cassette in the fixed dual-expression context. |
| `RtCds` | DNA coding sequence for the reverse transcriptase in the candidate row. |
| `LnrnaSequence` | DNA representation of the lnRNA cassette used in the construct context. |
| `MsdDesignSpec` | Study/compiler design metadata for the MSD, hairpin, payload, and feasibility details. |
| `ConstructSlot` | Named placement slot owned by Construct, for example `lnrna` or `rt_cds`, with template span, source field, emitted span, and orientation-aware bounds. |
| `ConstructProjectionManifest` | Study-owned fixture that binds candidate RT/lnRNA source authority to Construct slots and expected context views. |
| `DualExpressionConstruct` | Concrete sequence instance emitted by Construct. |
| `ConstructContextView` | Declared construct-context sequence view with orientation, coordinates, spans, and pooling intent. |
| `InferFeatureAlias` | Pointer to model-derived feature vectors and aliases. |
| `AbundancePriorOverlay` | Literature/source abundance prior, not a sponging assay label. |
| `SpongingAssayObservation` | Future lab TF-sponging label rows. |

### Excluded Core Fields

The candidate core schema must not require these fields:

- `perturbation_class`
- `cloning_constraint_set`
- `representation_result`

Use `source_basis`, optional `variant_derivation`, `MsdDesignSpec`,
`ConstructContextView`, and `InferFeatureAlias` instead.
