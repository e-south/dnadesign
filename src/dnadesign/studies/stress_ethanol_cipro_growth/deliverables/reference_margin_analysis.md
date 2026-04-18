# Reference-margin analysis

LatentDNA is a downstream comparison surface over the current `infer_batch_preparation` record. DenseGen remains the upstream source of cohort provenance and plan labels. This deliverable asks whether the shared `N = 157,164` promoter population lands near the expected wildtype references and whether that signal is still visible in the full embedding space.

### reference_margin_gallery_wildtype | Wildtype reference-margin gallery

#### Plot details

**Data.** Each point is one promoter from the merged \(N = 157{,}164\) promoter population, embedded in one candidate representation space. The reference anchors are the wildtype/control records present in the merged set: `spyP`, `sulAp`, and `J23105`.

**Definition.** For a promoter embedding \(z_x\), the signed wildtype margins are

$$
m_{\mathrm{eth}}(x) =
\cos(z_x, z_{\mathrm{spyP}}) -
\cos(z_x, z_{\mathrm{J23105}})
$$

and

$$
m_{\mathrm{cipro}}(x) =
\cos(z_x, z_{\mathrm{sulAp}}) -
\cos(z_x, z_{\mathrm{J23105}}).
$$

The x-axis is the ethanol reference margin and the y-axis is the ciprofloxacin reference margin.

**Interpretation.** Positive \(m_{\mathrm{eth}}\) means the promoter is closer to `spyP` than to `J23105` in that representation space. Positive \(m_{\mathrm{cipro}}\) means the promoter is closer to `sulAp` than to `J23105`. Points in the upper-right have larger signed margins toward both wildtype references relative to the `J23105` baseline.

### reference_neighbor_evidence | Reference-neighborhood evidence

#### Plot details

**Data.** This plot summarizes whether each candidate representation places the relevant reference records near the promoters in the original high-dimensional embedding space.

**Definition.** For promoter \(x_i\), let \(\mathcal{N}_k(x_i)\) be its \(k\)-nearest-neighbor set. The in-neighborhood rate is

$$
\mathrm{reference\_in\_knn\_rate}
=
\mathrm{mean}_i
\left[
\mathbf{1}\{
\mathrm{reference} \in \mathcal{N}_k(x_i)
\}
\right].
$$

The rank summary is

$$
\mathrm{reference\_neighbor\_rank\_median}
=
\mathrm{median}_i
\left[
\mathrm{rank}(\mathrm{reference}\mid x_i)
\right].
$$

Candidate summaries are aggregated across the scored reference tasks.

**Interpretation.** Higher `reference_in_knn_rate` means the relevant reference more often appears among local neighbors. Lower `reference_neighbor_rank_median` means the reference is reached earlier in the neighbor ordering. This is full-space evidence that complements the 2D reference-margin gallery.
