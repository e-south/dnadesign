# Context geometry audit

This deliverable measures what survives when the same promoter is embedded as a 60 bp anchor and as a 1 kb construct context. The paired comparisons are downstream LatentDNA checks over the shared handoff population, not a study-status authority.

### context_shift_reference_plane | Context-shift margin plane

#### Plot details

**Data.** Each point is an aligned anchor/context pair for the same promoter, keyed by `construct__anchor_id`. The anchor embedding comes from the 60 bp handoff and the context embedding comes from the 1 kb construct-context handoff.

**Definition.** The plotted coordinates are signed changes in the wildtype-reference margins:

$$
\Delta m_{\mathrm{eth}} =
m_{\mathrm{eth}}(\mathrm{full\_context}) -
m_{\mathrm{eth}}(\mathrm{anchor})
$$

and

$$
\Delta m_{\mathrm{cipro}} =
m_{\mathrm{cipro}}(\mathrm{full\_context}) -
m_{\mathrm{cipro}}(\mathrm{anchor}).
$$

**Interpretation.** Positive \(\Delta m_{\mathrm{eth}}\) means the promoter becomes more `spyP`-like relative to `J23105` in the full construct context. Positive \(\Delta m_{\mathrm{cipro}}\) means it becomes more `sulAp`-like relative to `J23105`. Values near zero mean the wildtype-margin readout is stable across added context.

### context_delta_distributions | Context-shift distributions

#### Plot details

**Data.** Each distribution summarizes anchor-to-context changes for matched promoter records. The matched anchor and context embeddings refer to the same promoter under different sequence context.

**Definition.** The plotted metrics are

$$
\mathrm{context\_self\_cosine}
=
\cos(z_{\mathrm{anchor}}, z_{\mathrm{context}}),
$$

$$
\mathrm{context\_shift\_l2}
=
\left\lVert
z_{\mathrm{context}} - z_{\mathrm{anchor}}
\right\rVert_2,
$$

plus the signed margin changes \(\Delta m_{\mathrm{eth}}\) and \(\Delta m_{\mathrm{cipro}}\).

**Interpretation.** Read the center, spread, and tails of each distribution. High `context_self_cosine` and low `context_shift_l2` indicate stable embeddings under added context. Wide or shifted \(\Delta m\) distributions indicate that the wildtype-margin readout changes when the construct context is included.

### context_geometry_summary | Context stability summary

#### Plot details

**Data.** Each row or point summarizes one candidate representation space across matched anchor/context promoter pairs.

**Definition.** The stability summaries include median `context_self_cosine`, median `context_shift_l2` distance,

$$
\mathrm{neighbor\_overlap\_fraction}
=
\frac{
|\mathcal{N}_a(x) \cap \mathcal{N}_b(x)|
}{k},
$$

and

$$
\mathrm{geometry\_distance\_correlation}
=
\mathrm{Spearman}(d_{\mathrm{anchor}}, d_{\mathrm{context}}).
$$

Here \(\mathcal{N}_a(x)\) and \(\mathcal{N}_b(x)\) are the anchor-space and context-space neighbor sets for the same promoter.

**Interpretation.** This is a compact metric panel, not a hidden combined score. Higher self-cosine, higher `neighbor_overlap_fraction`, and higher `geometry_distance_correlation` indicate more stable geometry across context. Lower `context_shift_l2` indicates smaller embedding displacement.
