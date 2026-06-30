# Sibling Pattern Example

This page records useful visual-interaction patterns from the sibling `ec86kit` project. It is example material only. Do not import `ec86kit` from this skill.

## Reusable Patterns

1. White-background scene defaults.
   - `ec86kit` uses `bg_color=white` in apply-map configuration.
   - Generic skill mapping: `set_background`.

2. Surface coloring as an optional visual layer.
   - `ec86kit` can create a molecular surface and color it by a per-residue attribute.
   - Generic skill mapping: `surface_selection`, `surface_transparency`, `color_selection`.

3. Silhouettes for readability.
   - `ec86kit` enables silhouettes for cleaner white-background figures.
   - Generic skill mapping: `silhouettes`.

4. Chain-role aliases.
   - `ec86kit` creates aliases for protein and nucleic-acid chain roles in a specific complex.
   - Generic skill mapping: declared `selection` names. The skill should not infer biology from chain IDs.

5. Safe command wrapper.
   - `ec86kit` logs ChimeraX commands before executing them inside ChimeraX.
   - Generic skill mapping: REST command log plus response status.

## Evidence

- Sibling `ec86kit` apply-map script, when that sibling checkout is available.
- Sibling `ec86kit` ChimeraX command utility, when that sibling checkout is
  available.
- Sibling `ec86kit` pairing script, when that sibling checkout is available.

Do not record machine-local absolute paths in this skill. These examples are
portable pattern names, not required files.

## Boundary

The reusable ontology is about models, selections, styles, scenes, poses, renders, and manifests. Any biological meaning of a selection belongs to the calling study or project.
