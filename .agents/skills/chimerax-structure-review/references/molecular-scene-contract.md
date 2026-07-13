# Molecular Scene Contract

Use this contract when one atomic model contains protein, DNA, and RNA. Declare
the chain selections before changing the scene.

## Required Roles

| Role | Required declaration | Stable UI name |
| --- | --- | --- |
| Source model | Atomic model selection such as `#1` | `molecular_complex` |
| Protein | Explicit protein-chain selection | `protein_role` |
| DNA | Explicit DNA-chain selection | `dna_role` |
| RNA | Explicit RNA-chain selection | `rna_role` |

Do not split one deposited complex into new atomic models merely to improve the
Model Panel. Rename the source model and define named selectors for molecule
roles instead.

## Canonical Representation

Protein:
- show the protein cartoon;
- draw the molecular surface only on the protein selection;
- use 65 percent surface alpha as the general interactive default;
- increase surface transparency when the cartoon must remain visually primary;
- use an opaque surface for a quantitative or qualitative color-mapped surface
  unless transparency is itself part of the declared comparison;
- apply `transparency ... target s` after surface-color commands, because later
  coloring can reset alpha;
- show only the `sidechain` atom class for explicitly highlighted residues.

DNA and RNA:
- use distinct, stable colors;
- use `#B97700` for DNA and `#C84C5A` for RNA unless the artifact declares a different accessible palette;
- show the native nucleic cartoon with backbone suppression enabled;
- use ChimeraX `ladder` as the default nucleotide representation;
- color nucleotide representations with target `f` so DNA ladders match the
  gold DNA cartoon and RNA ladders match the salmon RNA cartoon;
- use connected nucleotide sticks only as an explicit atomic-inspection mode;
- do not use slab, tube/slab, or filled-ring representations by default;
- remove automatic 3D labels and hide missing-segment pseudobonds when they are not part of the evidence.

The key commands are:

```text
cartoon <nucleic_selection> suppressBackboneDisplay true
cartoon style nucleic xsect oval width 1.35 thick 0.28
cartoon tether nucleic shape cylinder sides 8 scale 0.65 opacity 1
show <nucleic_selection> atoms
nucleotides <nucleic_selection> ladder
```

The native cartoon plus ladder path is preferred. Explicitly show nucleotide
atoms before setting ladder mode; otherwise some scripted scenes retain only
the backbone cartoon. ChimeraX attaches ladder rungs to the cartoon at the C3'
tether position. Use `nucleotides
<selection> atoms` followed by complete nucleotide sticks only when inspecting
atomic connectivity.

Some deposited complexes do not produce a continuous native nucleic cartoon.
Confirm that failure visually before using `phosphate-ribbon` mode. The fallback
must show complete nucleotide atoms, not `sideonly`, so every base remains
attached to its sugar and phosphate. Give every generated ribbon a semantic
model name such as `dna_backbone` or `rna_backbone_1`.

## Camera And Framing

1. Start a visible graphical session.
2. Let the user orient the complex.
3. Capture the camera matrix and save a named view.
4. Reapply the matrix in generated scripts.
5. Fit the declared visible scene with a small pad, then apply one bounded zoom.
6. Render the final aspect ratio before judging margins or cropping.
7. Render fixed-size scientific frame series with `save` when the movie-record
   buffer does not clear the declared background across the full image. Check
   all four corners before encoding; do not repair bands with a global color
   replacement that would also remove black outlines or labels.

Do not infer the final camera from a headless smoke run. Each output aspect
ratio requires its own framing check even when artifacts share a pose matrix.

## Movie Contract

- name the source model and molecule-role selectors before recording;
- declare frame size and frame rate;
- capture PNG frames at the final dimensions with an explicit background;
- use `movie record` for ordinary command-script capture, or numbered `save`
  frames when a fixed offscreen movie buffer fails the background check;
- declare total rotation, rotation per scene, frames per scene, and hold frames;
- reset prior highlights before showing the next residue set;
- show highlighted protein side-chain atoms as connected sticks and color the
  matching surface patch;
- record the camera matrix, commands, dimensions, and output hash in a render
  manifest;
- describe the rotation as a communication view, not molecular motion.

## Failure Taxonomy

| Failure | Likely cause | Correction |
| --- | --- | --- |
| Ladder rungs are absent | The nucleotide mode was reset to atoms or the cartoon is hidden. | Restore the native cartoon and run `nucleotides <selection> ladder`. |
| Bases look detached in atomic-inspection mode | `sideonly` atoms were paired with a separate shape ribbon. | Use complete nucleotide atoms with the native cartoon, or pair a verified phosphate-ribbon fallback with complete nucleotide atoms. |
| Bases look like plates or slots | A slab, tube/slab, or filled-ring nucleotide mode is active. | Run `nucleotides <selection> ladder` for the default view. |
| DNA and RNA are indistinguishable | Shared or inherited color. | Color the declared DNA and RNA roles separately. |
| Surface hides the protein cartoon | Surface alpha is too opaque or a later color command reset alpha. | Apply the declared transparency after all surface-color commands. |
| Color-mapped surface looks washed out | Transparency mixes the surface colors with the background or cartoon. | Use an opaque surface for the color-mapped comparison and keep one explicit scale. |
| Ladder color differs from its cartoon | Color targeted atoms and cartoons but omitted nucleotide representations. | Recolor DNA or RNA with target `acf`. |
| Movie contains a black band | The fixed offscreen movie buffer was only partly cleared. | Capture numbered square frames with `save`, verify all four corners, then encode the checked frames. |
| Excess empty margin | Fit occurred before final dimensions or included hidden models. | Set dimensions first, fit visible models, then apply bounded zoom. |
| Model Panel shows generic ribbons | Fallback shape models were created without semantic names. | Rename every fallback ribbon by molecule role. |

## Deterministic Helper

Preview the canonical command sequence:

```bash
uv run python .agents/skills/chimerax-structure-review/scripts/chimerax-apply-complex-style.py \
  --dry-run \
  --protein-selection '#1/A' \
  --dna-selection '#1/D' \
  --rna-selection '#1/E,F' \
  --nucleic-selection '#1/D,E,F'
```

After review, replace `--dry-run` with `--session-manifest <path>` to apply the
same allowlisted commands to the visible session.

For a deposited complex whose native nucleotide cartoon is absent, dry-run the
explicit fallback:

```bash
uv run python .agents/skills/chimerax-structure-review/scripts/chimerax-apply-complex-style.py \
  --dry-run \
  --protein-selection '#1/A' \
  --dna-selection '#1/D' \
  --rna-selection '#1/E,F' \
  --nucleic-selection '#1/D,E,F' \
  --nucleic-display connected-atoms \
  --nucleic-backbone-mode phosphate-ribbon \
  --dna-phosphate-selection '#1/D@P' \
  --rna-phosphate-selection '#1/E@P' \
  --rna-phosphate-selection '#1/F@P'
```
