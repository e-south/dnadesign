# External Sources

Retrieved: 2026-07-12

| Source | Authority | Used for |
| --- | --- | --- |
| 3Dmol.js AtomStyleSpec, https://3dmol.org/doc/AtomStyleSpec.html | Official API documentation. | Atom-scoped cartoon, stick, line, sphere, and surface styles. |
| 3Dmol.js CartoonStyleSpec, https://3dmol.org/doc/CartoonStyleSpec.html | Official API documentation. | Cartoon style fields and public API limits. |
| 3Dmol.js GLViewer, https://3dmol.org/doc/GLViewer.html | Official API documentation. | Public `addCustom` mesh and `addCylinder` shape methods. |
| 3Dmol.js cartoon renderer source, https://github.com/3dmol/3Dmol.js/blob/master/src/glcartoon.ts | Upstream implementation source. | Nucleic base-cylinder behavior associated with styled `N1` or `N3` atoms. |
| UCSF ChimeraX `nucleotides`, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/nucleotides.html | Official command documentation. | Ladder semantics, atom hiding, stubs, and cartoon tether positions. |
| UCSF ChimeraX `color`, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/color.html | Official command documentation. | Target `f` colors ring fill and nucleotide representations; ChimeraX transparency is the inverse of alpha. |
| UCSF ChimeraX system options, https://www.cgl.ucsf.edu/chimerax/docs/user/options.html | Official startup documentation. | `.cxc` script and command-line startup behavior. |

Use official APIs and checked-in public package facades as authority. Renderer
source inspection may explain behavior but must not become a private runtime
dependency.
