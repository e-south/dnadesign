# Cross-Renderer Contract

Match molecule roles, colors, framing intent, and evidence semantics across
renderers. Do not require identical primitives.

## Shared Source Preflight

Verify that every staged coordinate path retains:

- phosphate and backbone atoms, including `P`, `O5'`, and `O3'`;
- sugar atoms, including `C1'`, `C2'`, `C3'`, `C4'`, and `C5'`;
- `N9` for purines or `N1` for pyrimidines;
- the remaining base-ring atoms.

Atom loss during conversion is a data failure. Do not conceal it with custom
geometry.

## Semantic Mapping

| Role | ChimeraX | py3Dmol |
| --- | --- | --- |
| Protein | Native cartoon; optional protein-only surface at 65 percent alpha. | Native cartoon; optional protein-only surface at 65 percent alpha. |
| DNA | Gold native oval cartoon with a gold `ladder` nucleotide display. | Gold C4-prime rectangular ribbon mesh with one gold base spoke per nucleotide. |
| RNA | Salmon native oval cartoon with a salmon `ladder` nucleotide display. | Salmon C4-prime rectangular ribbon mesh with one salmon base spoke per nucleotide. |
| Highlight | Recolor selected cartoon, atoms, or surface patch. | Recolor the existing scoped representations and optional surface patch. |

The ChimeraX ladder and py3Dmol ribbon-with-spokes view are intentionally different
implementations. Describe them by renderer rather than claiming pixel-level
parity.

## Failure Routing

| Symptom | First owner |
| --- | --- |
| Missing sugar or attachment atoms | Coordinate conversion/materialization |
| Blank browser canvas | Browser/WebGL harness |
| py3Dmol base cylinders or duplicate backbone | py3Dmol style selection |
| ChimeraX ladder absent | ChimeraX nucleotide/cartoon state |
| GUI closes after launcher exits | ChimeraX session harness |
| Protein surface hides highlights | Renderer-specific surface selection/style |
