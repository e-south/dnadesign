---
name: molecular-structure-visualization
description: Route protein, DNA, and RNA scenes across py3Dmol and ChimeraX. Use for renderer choice, source integrity, or cross-renderer consistency. Do not use for structure prediction or biological interpretation.
metadata:
  version: 0.1.4
  category: workflow-automation
  tags: [molecular-visualization, py3dmol, chimerax, protein, dna, rna]
---

# Molecular Structure Visualization

## Purpose

Choose the correct molecular renderer, preserve protein/DNA/RNA semantics, and
verify that browser and desktop scenes are chemically connected and visually
legible.

## Scope

In scope:
- renderer selection between py3Dmol and ChimeraX
- protein, DNA, and RNA role declarations
- coordinate-source and atom-completeness checks
- py3Dmol scene contracts through `dnadesign.thread.structure_views`
- cross-renderer color and representation mapping
- browser WebGL and molecular-scene acceptance checks

Out of scope:
- ChimeraX GUI lifecycle, REST control, pose capture, electrostatics, or movies;
  route those tasks to `chimerax-structure-review`
- structure prediction, fold scoring, or molecular dynamics
- study-specific biological claims

## Required Inputs

- Local structure path and declared structure format.
- Explicit protein, DNA, and RNA chain roles.
- Intended surface: browser interaction, desktop pose review, still, or movie.
- Required representation and optional protein-surface behavior.

## Success Criteria

- The structure format matches the parser declaration.
- Coordinate conversion preserves required backbone, sugar, glycosidic, and
  base-ring atoms.
- DNA and RNA remain visually distinct.
- ChimeraX uses native nucleic cartoons with `ladder` nucleotide display by
  default.
- py3Dmol uses a coordinate-derived, flat C4-prime ribbon mesh plus one
  attached base spoke per nucleotide; it does not expose full atom rings or a
  round backbone tube by default.
- Protein surfaces are applied only to the protein role, start off in interactive views, and use 65 percent alpha when shown.
- Browser acceptance uses a real WebGL-capable browser and tests controls, not
  only static HTML generation.

## Workflow

1. Read `references/renderer-router.md` and choose one primary renderer.
2. Validate molecule roles and source atoms with
   `references/cross-renderer-contract.md`.
3. For py3Dmol, use `references/py3dmol-rendering-contract.md` and the public
   `dnadesign.thread.structure_views` API.
4. For ChimeraX GUI control or capture, route to `chimerax-structure-review`.
5. Verify the final surface in its real runtime: GPU-backed browser for
   py3Dmol or graphical ChimeraX for desktop scenes.
   For a saved py3Dmol artifact, run `scripts/verify-py3dmol-webgl.py` and
   retain its screenshot and JSON output.
6. When both renderer artifacts exist, run
   `scripts/verify-molecular-scene-contract.py` against the browser manifest
   and ChimeraX script. For a notebook or report bundle, pass its top-level
   review manifest so every linked structure artifact is audited.
7. Run `scripts/audit-molecular-structure-visualization-skill.sh` after edits.

## Guardrails

- Do not force identical low-level representations across different renderers.
- Do not substitute a round `addCurve` tube for the flat nucleic ribbon.
- Do not infer missing covalent connectivity from visual proximity.
- Do not apply a surface to DNA or RNA under the default scene contract.
- Do not bypass the public `dnadesign.thread.structure_views` API from study
  notebooks.

## Required Deliverables

- selected renderer and reason
- source format, molecule-role map, and atom-completeness result
- representation and color contract
- machine-readable cross-renderer verification when both artifacts exist
- runtime validation evidence
- explicit limits where renderer capabilities differ

## Output Contract

Return the selected renderer, molecule-role and source-integrity checks,
renderer-specific representation, runtime verification evidence, and any
capability mismatch that remains visible.

## Trigger Tests

Should trigger:
- "Render this protein-DNA-RNA complex in py3Dmol."
- "Which renderer should we use for this molecular scene?"
- "Keep the ChimeraX and browser structure views visually consistent."
- "The bases are detached from the nucleic-acid backbone."

Should not trigger:
- "Orient this structure manually in ChimeraX and capture the pose."
- "Run ColabFold."
- "Interpret whether this mutation improves activity."

## Progressive Disclosure Resources

- `references/renderer-router.md`
- `references/py3dmol-rendering-contract.md`
- `references/cross-renderer-contract.md`
- `references/external-sources.md`
- `references/test-matrix.md`
- `scripts/verify-molecular-scene-contract.py`
- `scripts/verify-py3dmol-webgl.py`
- `scripts/audit-molecular-structure-visualization-skill.sh`
