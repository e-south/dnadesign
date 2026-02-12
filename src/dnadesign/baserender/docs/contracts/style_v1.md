# Style v1 Contract

Style v1 controls typography, geometry, connectors, legend, and palette.

## Sources
Effective style is merged in this order:
1. `styles/presentation_default.yml`
2. optional preset file
3. inline `render.style.overrides`

## Key groups
- Typography: `font_*`, `font_size_*`, `dpi`
- Geometry: `padding_*`, `track_spacing`, `baseline_spacing`
- Feature box style: `kmer.*`
- Legend: `legend*`
- Connectors: `connectors`, `connector_*`
- Effects: `span_link_inner_margin_bp`
- Palette overrides: `palette`

## Strictness
- Unknown top-level style keys fail.
- Unknown `kmer.*` keys fail.
- Invalid value domains fail (e.g. non-positive dpi, bad align mode).
