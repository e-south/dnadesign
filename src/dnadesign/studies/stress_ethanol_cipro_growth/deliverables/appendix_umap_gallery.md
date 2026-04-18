# UMAP gallery

These panels are downstream orientation surfaces over the shared LatentDNA handoff. They help inspect the geometry of the eight realized candidate spaces, but they are not the decision rule for choosing a representation.

### reference_margin_gallery_synthetic_centroids | Synthetic proxy margin gallery

#### Plot details

**Data.** Each point is one promoter embedded in a candidate representation space. Unlike the wildtype reference-margin gallery, this plot uses synthetic cohort centroids instead of the real wildtype/control reference records.

**Definition.** The proxy margins are

$$
m_{\mathrm{eth}}^{\mathrm{syn}}(x)
=
\cos(z_x, c_{\mathrm{eth}})
-
\cos(z_x, c_{\mathrm{bg}})
$$

and

$$
m_{\mathrm{cipro}}^{\mathrm{syn}}(x)
=
\cos(z_x, c_{\mathrm{cipro}})
-
\cos(z_x, c_{\mathrm{bg}}).
$$

Here \(c_{\mathrm{eth}}\), \(c_{\mathrm{cipro}}\), and \(c_{\mathrm{bg}}\) are cohort centroids.

**Interpretation.** This is a proxy margin surface. It can show whether the cohort-centroid geometry has a similar orientation to the wildtype-reference surface, but it should not be treated as equivalent to the real-reference margin plot.

### appendix_umap_gallery | UMAP gallery

#### Plot details

**Data.** Each panel shows the full \(N = 157{,}164\)-row promoter population projected for one realized candidate representation space. The current projection artifacts are full-population fits, not notebook-level 1k-row samples.

**Definition.** The plotted coordinates are the persisted UMAP coordinates for each candidate space. Hue changes recolor the same coordinates; they must not trigger a new projection or a different sampled population.

**Interpretation.** These plots are orientation surfaces. Neighborhood layout can help identify broad geometry, density, and grouping patterns, but UMAP position is not the decision rule for selecting a representation.
