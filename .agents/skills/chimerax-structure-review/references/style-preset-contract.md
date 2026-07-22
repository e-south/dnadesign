# Style Preset Contract

Style presets describe visual grammar only. They must not encode biological claims.

## Ontology

- `model`: an opened structure file.
- `reference_model`: model used as the visual alignment reference.
- `query_model`: model being compared against the reference.
- `selection`: explicit ChimeraX atomspec, chain, or residue range.
- `scene`: visible representations, colors, labels, and lighting.
- `pose`: camera/view state plus model transforms for a still.
- `render`: saved image and optional session.
- `style_preset`: reusable display settings.
- `molecule_role`: named selector for protein, DNA, or RNA within one complex.

## Generic Presets

| Preset | Intent | Command keys |
| --- | --- | --- |
| `white_cartoon_reference` | Clean single-model cartoon view. | `set_background`, `camera_mode`, `lighting`, `silhouettes`, `color_selection`, `cartoon_style` |
| `faint_surface_shield` | Add a protein surface at 65 percent alpha around a selected model. | `surface_selection`, `surface_transparency`, `color_selection` |
| `chain_role_colors` | Color declared chains or roles distinctly. | `color_selection`, `show_selection` |
| `transparent_comparison_model` | Overlay a comparison model without hiding the reference. | `cartoon_transparency`, `color_selection` |
| `title_only` | Add a title without changing structure style. | `title_label` |
| `role_aware_protein_nucleic_complex` | Protein surface at 65 percent alpha with chain-colored DNA and RNA ladder views on native cartoons. | `rename_model`, `name_role`, `nucleotide_ladder`, `cartoon_backbone_suppression`, `cartoon_style`, `cartoon_tether`, `surface_selection`, `surface_transparency`, `color_selection`, `fit_view` |

## Naming Rules

Allowed:
- visual terms: `white`, `cartoon`, `surface`, `faint`, `reference`, `comparison`, `chain`
- role terms: `reference_model`, `query_model`, `selection`

Avoid:
- study-specific labels
- inferred function labels
- assay or phenotype terms

Study-specific projects may map their own biological labels to generic selections outside this skill.
