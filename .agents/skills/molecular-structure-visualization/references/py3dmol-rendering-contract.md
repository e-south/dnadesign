# py3Dmol Rendering Contract

Use the public `dnadesign.thread.structure_views` API for browser scenes:

```python
from dnadesign.thread.structure_views import (
    StructureViewModel,
    StructureViewMoleculeStyle,
    StructureViewSpec,
    render_structure_view_html,
)
```

Do not import backend internals from study notebooks.

## Input Contract

- The declared format must match the coordinate text passed to 3Dmol.
- Prefer the validated all-atom browser PDB when the source mmCIF has already
  been converted by a structured parser.
- Verify that conversion preserves `C4'` and enough base-ring atoms to compute
  one base centroid per nucleotide.
- Reject incomplete staged structures before rendering.

## Default Roles

Protein:
- native cartoon;
- optional VDW/SES surface on the protein selection only with `opacity=0.65`, off by default;
- sticks only for selected residues.

DNA and RNA:
- DNA `#B97700`; RNA `#C84C5A`;
- semantic style `backbone_ribbon_with_base_spokes`;
- one thin rectangular ribbon mesh through observed C4-prime coordinates per
  chain, with default width `1.35 A` and thickness `0.28 A`;
- one spoke from each C4-prime coordinate to its base-ring centroid;
- each ribbon and its spokes use the same chain color.

The ribbon must use observed molecular coordinates, preserve chain breaks, and
reject residues without a C4-prime anchor or base-ring coordinates. It must not
infer connectivity from residue centroids.

## Renderer-Specific Rules

- Do not expose 3Dmol's native nucleic cartoon by default; it adds base
  cylinders that do not match the shared ladder-like communication style.
- Do not apply sticks to every nucleotide atom; full rings and phosphate bonds
  obscure the ribbon and duplicate the backbone.
- Use public `addCustom` mesh geometry for the default flat ribbon and public
  `addCylinder` geometry for the base spokes.
- Do not use `addCurve` for the default nucleic backbone; it produces a round
  tube and diverges from the ChimeraX band aesthetic.
- Use the renderer version pinned by the owning project rather than an
  unversioned CDN default.
- Use stable named background colors such as `white` when renderer repaint
  behavior makes hexadecimal backgrounds unstable.
- Recolor or replace the existing style for highlights; do not layer duplicate
  cartoons or nucleotide sticks.

py3Dmol does not expose ChimeraX's `nucleotides ... ladder` command. Preserve
the same molecule roles and colors with a coordinate-derived ribbon and one
base-facing spoke per nucleotide.

## Runtime Verification

Use a GPU-backed browser and verify:

1. the WebGL canvas is nonblank;
2. DNA and RNA colors are distinct;
3. every nucleotide has one attached base spoke;
4. the ribbon is visibly wider than it is thick;
5. no base cylinders, full atom rings, or duplicate phosphate sticks are
   present;
6. protein-surface, DNA, and RNA controls affect only their declared roles;
7. highlights remain visible when the protein surface is enabled;
8. the browser console contains no renderer errors.

A static HTML export or HTTP 200 response does not prove these interactions.
Each rendered iframe exposes `window.__dnadesignStructureSceneAudit`; verify
its nucleotide, ribbon-mesh, vertex, triangle, segment, and spoke counts plus
the declared width and thickness alongside screenshot pixels.

## Camera And Interaction Contract

- Fit the complete set of visible molecule roles, then apply only a bounded
  presentation zoom. The initial frame must show the whole complex without the
  large empty margins of an unadjusted `zoomTo()` result.
- Persist camera state under a semantic, versioned key shared by views of the
  same structure family. Change the key version when the default framing or
  orientation changes so stale browser storage cannot override the fix.
- Keep display controls independent: hiding a reference protein must not hide
  reference DNA or RNA when their controls remain enabled.
- Use a reduced pointer-pan multiplier for trackpads. Verify that ordinary
  two-finger movement makes a small translation rather than moving the complex
  out of frame.
- Exercise every structure row, residue highlight, and display-state branch.
  Static command inspection is not sufficient because model indices and
  highlight targets change when models are hidden.
