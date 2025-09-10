## `setpoint_fidelity_x_intensity_p95_v1`

**Intent.** Combine a model’s predicted **logic pattern** and **absolute fluorescent intensity** into a single score that rewards sequence designs that are both **right** (match the desired multi-condition setpoint) and **bright** (intense in the right conditions).

---

### 1. What the model predicts

The model predicts an **8-vector (Y)** per sequence design. The first four entries describe the **shape** of a two-factor logic response (bounded from 0 to 1). The last four capture **absolute fluorescent intensity** per state, but stored in **log2 space** for modeling stability.

$$
\underbrace{v_{00}, v_{10}, v_{01}, v_{11}}_{\text{logic in }[0,1]^4}\;,\;
\underbrace{y^{\star}_{00}, y^{\star}_{10}, y^{\star}_{01}, y^{\star}_{11}}_{\text{log2(abs. fluorescent intensity)}}
$$

* $v \in [0,1]^4$: **observed logic profile** in state order $[00,10,01,11]$.
* $y^\star \in \mathbb{R}^4$: **per-state absolute fluorescent intensity** in log2 space.

### 1.1 From experimental data → vec8

We start from raw fluorescent readouts for each state $i$:
$Y^{\mathrm{RFU}}_i$ (YFP) and $C^{\mathrm{RFU}}_i$ (CFP).

We also include, in every experiment, a **reference strain** with constitutive YFP to serve as an anchor—a per-experiment, per-state reference that removes gain/time/instrument drift while preserving a meaningful **effect size** of fluorescent intensity. Let

$$
A_{\mathrm{experiment},i} := \operatorname{mean}\{\text{references' } Y^{\mathrm{RFU}}_i\}.
$$

#### (a) Logic (fluorophore ratio → log2 → per-design min–max)

Using a dual fluorescent reporter allows us to [separate intrinsic from extrinsic noise](https://pmc.ncbi.nlm.nih.gov/articles/PMC3141918/) in our experiments. The YFP/CFP ratio cancels cell size/loading/day effects; log2 makes fold-changes symmetric; per-design min–max maps the four states onto a common $[0,1]$ scale so logic shapes are comparable across designs.

$$
r_i = \frac{Y^{\mathrm{RFU}}_i + \varepsilon}{C^{\mathrm{RFU}}_i + \varepsilon},
\qquad
u_i = \log_2(r_i)
$$

$$
u_{\min}=\min_i u_i,\quad u_{\max}=\max_i u_i
$$

$$
v_i = \frac{u_i - u_{\min}}{(u_{\max}-u_{\min})+\eta} \in [0,1]
$$

If $u_{\max}\approx u_{\min}$ (flat logic), set $v_i=\tfrac{1}{4}$ for all $i$ and **warn**.
($\varepsilon,\eta>0$ are small stabilizers; record them in metadata.)

#### (b) Absolute fluorescent intensity (reference-normalized → log2)

Absolute intensity carries the **effect size** or **scale** signal but raw RFUs drift by experiment/time. Dividing by a per-experiment reference strain removes this drift while keeping a meaningful, unitless intensity. We store log2 to stabilize the regression and avoid dominance by a few ultra-bright samples; we invert back to linear when ranking samples.

$$
y^{\mathrm{linear}}_i = \frac{Y^{\mathrm{RFU}}_i}{A_{\mathrm{plate},i} + \alpha}
\quad\text{(unitless, reference-normalized absolute fluorescence)}
$$

$$
y^\star_i = \log_2\!\big(y^{\mathrm{linear}}_i + \delta\big)
$$

**Vec8 label (stored):**

$$
Y = [\,v_{00}, v_{10}, v_{01}, v_{11},\; y^\star_{00}, y^\star_{10}, y^\star_{01}, y^\star_{11}\,]
$$

### 1.2 Modeling note (median–IQR robust scaling):

Random forest models handle mixed targets (our vec8: four bounded logic $v$ + four log-intensity $y^\star$), but with low sample counts the per-state log-intensities can have wide spreads, letting one state dominate split decisions. A affine, monotonic, and reversible median–IQR scaling puts the four intensity targets on a comparable scale so early fits aren’t skewed.

* Fit-time transform (applied to all training samples, per state): compute **campaign-cumulative training** median and IQR for each intensity target $y^\star_i$, then

$$
\tilde{y}^\star_i
= \frac{\,y^\star_i - \operatorname{median}_{\text{train}}(y^\star_i)\,}
         {\max\!\big(\operatorname{IQR}_{\text{train}}(y^\star_i),\,\varepsilon\big)}.
$$

This centers typical values near 0 and makes “1 unit” ≈ one IQR, reducing outlier leverage and balancing targets. (We do not scale $v$; it’s already in $[0,1]$.)

### 1.3 Inference-time inversion (undo scaling → undo log)

$$
\widehat{y}^\star_i
= \widehat{\tilde{y}}^\star_i \,\operatorname{IQR}_{\text{train}}(y^\star_i)
  + \operatorname{median}_{\text{train}}(y^\star_i),
\qquad
\widehat{y}^{\mathrm{linear}}_i
= \max\!\bigl(0,\; 2^{\,\widehat{y}^\star_i} - \delta\bigr).
$$

We return predictions to their original, linear intensity units so downstream steps use a meaningful scale. The scaling we applied at fit time is a uniform affine transform (same shift/scale for all samples), so reversing it—and then reversing the log—preserves ordering and simply restores interpretable magnitudes.

**Note on $\max(0,\cdot)$ and $\delta$:** The $\delta>0$ term is the same small offset used when taking $\log_2(y^{\mathrm{linear}}+\delta)$ to avoid $\log(0)$. Subtracting $\delta$ undoes that offset; the outer $\max(0,\cdot)$ guards against tiny negative values from numerical round-off, ensuring the recovered $\widehat{y}^{\mathrm{lin}}_i$ remains non-negative.

---

### 2. Inputs to the objective (selection time)

* **Predictions:** $\widehat{v}\in[0,1]^4$ and $\widehat{y}^{\mathrm{linear}}\in\mathbb{R}_{\ge 0}^4$.
* **Setpoint** (i.e., preference): $p\in[0,1]^4$. This can be binary setpoints (e.g., AND: $[0,0,0,1]$) or nuanced continuous ones (e.g., $[0.3,0.4,0.7,0.2]$).

Each candidate is scored with $p$, $\widehat{v}$, and $\widehat{y}^{\mathrm{linear}}$.

---

### 3. Logic fidelity

**Goal.** Compare the predicted logic vector $\widehat{v}\in[0,1]^4$ to the **setpoint** $p\in[0,1]^4$ using **root-mean-square error (RMSE)**, then turn that error into a **similarity** in $[0,1]$: 1 = perfect match, 0 = as bad as it can be for this setpoint.

**“Worst-case” error for this setpoint.** Inside the unit 4-cube $[0,1]^4$, the farthest point from $p$ is a **corner**. For each state $i$, choose whichever of $\{0,1\}$ is farther from $p_i$
(e.g., if $p_i=0.6$, distance to 0 is $0.6$, to 1 is $0.4$ ⇒ pick **0**).
That corner’s Euclidean distance from $p$ is

$$
D \;=\; \sqrt{\sum_{i=1}^{4} \max\!\big(p_i^2,\,(1-p_i)^2\big)}.
$$

Examples: $p=[0,0,0,1]\Rightarrow D=2$; $p=[0.5,0.5,0.5,0.5]\Rightarrow D=1$.

**Normalized RMSE → similarity.** With four states, $\mathrm{RMSE}(\widehat{v},p)=\tfrac{1}{2}\lVert \widehat{v}-p\rVert_2$ and $\mathrm{RMSE}_{\max}=\tfrac{D}{2}$. Report

$$
F_{\text{logic}}
\;=\; 1 \;-\; \frac{\mathrm{RMSE}(\widehat{v},p)}{\mathrm{RMSE}_{\max}}
\;=\; 1 \;-\; \frac{\lVert \widehat{v}-p\rVert_2}{D},
\qquad
F_{\text{logic}} \leftarrow \operatorname{clip}(F_{\text{logic}},\,0,\,1).
$$

$F_{\text{logic}}=1$ only when $\widehat{v}=p$; $F_{\text{logic}}=0$ at the setpoint’s farthest corner. A value like **0.7** means the joint error is 30% of the worst possible for this $p$ (i.e., you’re 70% of the way from “worst miss” to “perfect match”).

For valid $\widehat{v}\in[0,1]^4$, $\lVert \widehat{v}-p\rVert_2\!\le\! D$ so $F_{\text{logic}}\in[0,1]$ already; the clip only guards tiny numerical drift or rare predictions slightly outside $[0,1]$. Normalizing by $D$ makes scores directly comparable across binary or continuous setpoints.

---

### 4. Effect: fluorescent intensity **in the desired conditions** (with round-internal robust scaling)

**Plain-language idea:** Take a **weighted average** of the predicted per-state intensity, where the weights come from the **setpoint**—states you care about more get higher weight. Then scale that number using only this round’s labeled data so it lands in $[0,1]$ and stays resistant to outliers and day-to-day drift.

**Weights from the setpoint (turn setpoint into state weights):**

$$
P=\sum_i p_i,\qquad
w_i =
\begin{cases}
\dfrac{p_i}{P}, & P>0,\\[4pt]
0, & P=0.
\end{cases}
$$

**Raw effect (weighted average of predicted intensities):**

$$
E_{\text{raw}} = \sum_{i=1}^{4} w_i \,\widehat{y}^{\mathrm{lin}}_i
\quad\text{(equivalently } E_{\text{raw}}=\tfrac{p\cdot \widehat{y}^{\mathrm{lin}}}{\max(P,\epsilon)}\text{)}.
$$

* Increasing intensity where $p_i$ is large **always** increases $E_{\text{raw}}$.
* Intensity in states with $p_i=0$ does **not** increase $E_{\text{raw}}$.
* If $P=0$ (an “all-OFF” setpoint), define $E_{\text{raw}}=0$; the logic term will drive the score.

**Round-internal robust scaling (what/why/how):**
We compute a denominator **within the current active-learning round** using only **labeled rows** and the **current setpoint** $p$. This turns $E_{\text{raw}}$ into a unit-free $[0,1]$ value that:

* is **robust** to outliers (we use a high percentile),
* adapts to each round’s dynamic range (new plates/instruments),
* lets scores remain comparable **within the round** without hard clipping artifacts.

Let $E^{\text{(lab)}}_{\text{raw}}$ be the raw effects recomputed for the labeled rows under the current $p$. Then

$$
\text{denom} = \max\!\big(\operatorname{P95}\{E^{\text{(lab)}}_{\text{raw}}\},\, \epsilon\big),
\qquad
E_{\text{scaled}} = \operatorname{clip}\!\left(\frac{E_{\text{raw}}}{\text{denom}},\, 0,\; 1\right).
$$

(If very few labels exist early on, you may choose a slightly lower percentile; behavior remains monotone and robust.)

* **Effect term:** $E_{\text{raw}}$ is a **weighted average** determined by the setpoint; it increases only when intensity rises in states the setpoint favors; insensitive to intensity where $p_i=0$.


* **Round scaling:** $\operatorname{P95}$ on labeled $E_{\text{raw}}$ gives a robust, unit-free denominator; clipping to $[0,1]$ prevents distortions from extreme outliers.
---

### 5. Final scoring metric (no gates; tunable curvature)

$$
\boxed{\;\text{score} = \big(F_{\text{logic}}\big)^{\beta}\cdot \big(E_{\text{scaled}}\big)^{\gamma}\;}
\qquad \beta=\gamma=1.
$$

* $\beta>1$: emphasize logic correctness more strongly.
* $\gamma>1$: emphasize intensity in the desired conditions more strongly.

* **Final product:** Ensures “wrong logic” or “dim in desired states” both depress the score, while “right and intense” is rewarded.

---

### 6. Metric properties

* **Batch robustness.** Logic uses YFP/CFP ratios; intensity uses a **reference-strain anchor**; and selection scales the effect by a **within-round percentile** over labels.
* **Setpoint flexibility.** You can change $p$ anytime; the objective recomputes $E_{\text{raw}}$ using the already predicted $\widehat{y}^{\mathrm{lin}}$.
* **Continuity.** Everything is continuous; no threshold/gate discontinuities.

---

### 7. Concrete examples (state order $[00,10,01,11]$)

#### (a) AND-like setpoint

$p=[0,0,0,1]\Rightarrow w=[0,0,0,1]$.

* Candidate A: $\widehat{y}^{\mathrm{lin}}=[0.1,0.2,0.3,1.2]$ → $E_{\text{raw}}=1.2$ (all credit from $A{+}B$).
* Candidate B: $\widehat{y}^{\mathrm{lin}}=[0.8,0.9,0.7,0.2]$ → $E_{\text{raw}}=0.2$ (intensity in the wrong conditions doesn’t help).
  If B’s $\widehat{v}$ is also far from $p$, $F_{\text{logic}}$ is small → score stays low.

#### (b) Nuanced setpoint

$p=[0.3,0.4,0.7,0.2]\Rightarrow P=1.6,\; w\approx[0.1875,0.25,0.4375,0.125]$.

* C: $\widehat{y}^{\mathrm{lin}}=[0.2,0.8,0.2,0.8]$ → $E_{\text{raw}}\approx 0.35$.
* D: $\widehat{y}^{\mathrm{lin}}=[0.1,0.5,1.1,0.4]$ → $E_{\text{raw}}\approx 0.56$.
  Similar total signal; D wins because it is intense in the **high-$p$** state (01).

#### (c) “All-OFF” setpoint

$p=[0,0,0,0]\Rightarrow w=0\Rightarrow E_{\text{raw}}=0$.
Only proximity of $\widehat{v}$ to $p$ (being OFF everywhere) is rewarded.

---

### 8. Edge cases and guards

* **Tiny CFP or anchor:** add $\varepsilon,\alpha$.
* **Flat logic:** if $u_{\max}\approx u_{\min}$, set $v=\tfrac{1}{4}\mathbf{1}$.
* **Non-finite:** reject at ingestion.

---

### 9. Column slugs (diagnostics)

* `opal__{slug}__r{k}__logic_fidelity__l2_to_setpoint__normalized_0_1`
* `opal__{slug}__r{k}__effect_size__setpoint_weighted_yfp__p95_scaled_0_1`
* `opal__{slug}__r{k}__score__logic_x_intensity__beta{β}_gamma{γ}`

---

@e-south