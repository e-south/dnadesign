# Style v1 Contract

Style v1 controls typography, geometry, connectors, legend, and palette.

## Sources
Effective style is merged in this order:
1. `styles/presentation_default.yaml`
2. optional preset file
3. inline `render.style.overrides`

## Key groups
- Typography: `font_*`, `font_size_*`, `dpi`
- Geometry: `padding_*`, `track_spacing`, `baseline_spacing`
- Feature box style: `kmer.*`
- Motif logo style: `motif_logo.*`
- Legend: `legend*`
- Connectors: `connectors`, `connector_*`
- Effects: `span_link_inner_margin_bp`
- Palette overrides: `palette`

## Strictness
- Unknown top-level style keys fail.
- Unknown `kmer.*` keys fail.
- Unknown `motif_logo.*` keys fail.
- Unknown `motif_logo.scale_bar.*` keys fail.
- Invalid value domains fail (e.g. non-positive dpi, unsupported enum values).

## `motif_logo` keys
- `layout`: `stack | overlay`
- `lane_mode`: `follow_feature_track | independent`
- `display_mode`: `information | probability`
- `height_bits`: information-content ceiling (DNA default 2.0)
- `bits_to_cells`: vertical scale factor (sequence-cell units per bit)
- `y_pad_cells`: vertical gap between kmer band and nearest logo lane
- `letter_x_pad_frac`: horizontal letter inset in each base column
- `alpha_other`: alpha for non-observed bases in a column
- `alpha_observed`: alpha for observed base in a column
- `colors`: mapping for `A/C/G/T`
- `debug_bounds`: optional logo bounding-box overlay
- `scale_bar.enabled`: draw a motif y-scale reference bar
- `scale_bar.location`: `top_right | bottom_right`
- `scale_bar.font_size`: scale bar label font size
