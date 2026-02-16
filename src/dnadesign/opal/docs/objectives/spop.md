## sponging_percent_of_positive `spop`

> Archived note: `spop_v1` is not registered in the current OPAL objective registry.
> Treat this file as historical design context, not an active plugin contract.

**Intent.** IPTG increases expression of an msr/msd cassette with an extended hairpin. Constitutively expressed RT variants are tested for the ability to process these long hairpins to produce a mature retron. Mature retron DNA sponges TetR and derepresses the reporter, so RFP is a proxy for strand displacement and retron maturation. Our goal is to rank RT variants that deliver strong sponging, preferably at lower IPTG, without compromising growth.

---

#### 1) Channels and data shape

**Per variant $g$, per dose $j\in{0..J}$ (ascending), with replicates $r$:**

* OD600: $O_{g,j,r}$

* RFP: $R_{g,j,r}$
* Growth‑normalized RFP: $Z_{g,j,r}=R_{g,j,r}/(O_{g,j,r}+\epsilon_{od})$

**Variant‑level positive control (aTc):**

$O_{pos,g,r}$, $R_{pos,g,r}$, $Z_{pos,g,r}=R_{pos,g,r}/(O_{pos,g,r}+\epsilon_{od})$

Ceiling for derepression that does not depend on retron maturation.

**Replicate aggregation (medians across $r$):**

$\tilde O_{g,j}$, $\tilde R_{g,j}$, $\tilde Z_{g,j}$, and $\tilde O_{pos,g}$, $\tilde R_{pos,g}$, $\tilde Z_{pos,g}$.

If a variant’s aTc well is missing, use a plate‑level aTc median $\tilde Z_{pos}$ as fallback.

---

#### 2) Per‑dose Y label

Percent of positive, growth‑normalized:

$$
y_{g,j}=\frac{\tilde Z_{g,j}}{\tilde Z_{pos,g}+\epsilon_{pos}}
$$

$y_{g,j}$ is unitless and comparable across plates.

**Primary label vector:**

Ordered per‑dose series used for learning:
$$
Y^{dose}_g=[y_{g,0},...,y_{g,J}]
$$

**Optional diagnostic (raw RFP fraction):**
$$
y^{raw}_{g,j}=\frac{\tilde R_{g,j}}{\tilde R_{pos,g}+\epsilon_{pos}}
$$

---

#### 3) Viability per dose (relative to zero IPTG)

We compare each induced condition to the variant’s own uninduced growth.

* **Baseline per variant:** $B_g=\tilde O_{g,0}$.

* **Per‑dose viability factor** (one‑sided; only penalizes drops below the baseline):
  $$
  v_{g,j}=min\left(1,\frac{\tilde O_{g,j}}{B_g+\epsilon_{od}}\right)
  $$

Interpretation: $v_{g,j}=1$ means “as viable as uninduced”; values below 1 indicate induction‑associated growth loss for that variant.

---

#### 4) Dose set used for scoring

Exclude zero IPTG by default to avoid rewarding leakiness:

$$S=\{j:\,dose_j>0\}$$


---

#### 5) Potency and final score (cumulative aggregation)

We reward any observed sponging and, because the dose series is ascending, earlier turn‑on naturally accumulates more credit.

**Potency (cumulative average across scoring doses):**

  $$
  P_g=\frac{1}{|S|}\sum_{j\in S} y_{g,j}
  $$

  This increases whenever any $y_{g,j}$ increases and is higher when a variant turns on at lower doses (since those doses also contribute at higher levels downstream).

**Viability (average across scoring doses):**

  $$
  V_g=\frac{1}{|S|}\sum_{j\in S} v_{g,j}
  $$

**Final score** with viability weight $ \lambda\in[0,1] $:

  $$
  Score_g=P_g,\big((1-\lambda)+\lambda,V_g\big)
  $$

  $ \lambda=0 $ ignores viability. $ \lambda=1 $ scales potency by average viability. This is monotonic in RFP response and applies a tunable penalty when induced growth falls below the variant’s own zero‑IPTG baseline.

---

#### 6) Emissions

**Per‑variant (`kind="run_pred"`):**

* `pred__score_selected`: $Score_g$
* `pred__y_per_dose`: $[y_{g,0},...,y_{g,J}]$
* `qc__viability_per_dose`: $[v_{g,0},...,v_{g,J}]$
* `qc__baselines`: $B_g$ and `$ \tilde Z_{pos,g} $

**Per‑run (`kind="run_meta"`):**

* `obj__name`: "spop_v1"
* `selection__score_ref`: "pred__score_selected"
* `obj__lambda`: $\lambda$
* `obj__eps_od`: $\epsilon_{od}$
* `obj__eps_pos`: $\epsilon_{pos}$
* `obj__dose_vec`: original dose values (ascending, consistent units)
* `qc__reference_strain_zero`: $ \tilde O_{ref,0} $ if used

---

#### 7) Defaults

* $ \epsilon_{od}=10^{-8} $, $ \epsilon_{pos}=10^{-8} $
* $ S=\{j:\,dose_j>0\} $ (exclude zero IPTG)
* $ \lambda=0.5 $ (equal emphasis on potency and viability)
* If a variant lacks aTc, use plate‑level $ \tilde Z_{pos} $ for that variant

---

#### 8) Edge cases and QC

* Very small $ \tilde Z_{pos,g} $ can inflate ratios; rely on $ \epsilon_{pos} $ and flag for QC.
* Near‑zero OD is guarded by $ \epsilon_{od}$ ($ v_{g,j} $ will be small, reducing the score when $ \lambda>0 $).
* If no nonzero doses exist in $S$, scoring is undefined; emit null and flag.
* Record which baselines were used ($B_g$ vs reference strain vs plate zero‑IPTG median).

---

#### 9) Pipeline summary

1. **Plate to vectors.** Compute $\tilde O_{g,j}$, $\tilde R_{g,j}$, $\tilde Z_{g,j}$, the variant’s $\tilde Z_{pos,g}$, and the baseline $B_g$. Then compute \(y_{g,j}\) and \(v_{g,j}\).
2. **Vectors to scalar.** Choose $S$; compute \(P_g\), \(V_g\), and \(Score_g\).
