## setpoint_fidelity_x_intensity `sfxi`

> **Scope.** This document describes the end‑to‑end selection scalar used by OPAL’s demo pipeline (logic fidelity × intensity). It spans ingest, Y‑ops, modeling, objective, and events.

**Objective intent:** Combine a model’s predicted **logic pattern** and **absolute fluorescent intensity** into a single score that rewards sequence designs that are both **right** (match the target setpoint) and **bright** (intense in the target conditions).

--- 

### 1. What the model predicts

The model predicts an **8-vector (Ŷ)** per input sequence (kept as `pred__y_hat_model`). The first four entries describe the **shape** of a two-factor logic response (bounded from 0 to 1). The last four capture **absolute fluorescent intensity** per state, but stored in **log2 space** for modeling stability.

$$
\underbrace{v_{00}, v_{10}, v_{01}, v_{11}}_{\text{logic in }[0,1]^4}\;,\;
\underbrace{y^{\star}_{00}, y^{\star}_{10}, y^{\star}_{01}, y^{\star}_{11}}_{\text{log2(abs. fluorescent intensity)}}
$$

* $v \in [0,1]^4$: **observed logic profile** in state order $[00,10,01,11]$.
* $y^\star \in \mathbb{R}^4$: **per-state absolute fluorescent intensity** in log2 space.

### 1.1 From experimental data → 8-vector

We start from raw fluorescent readouts for each state $i$:
$Y^{\mathrm{RFU}}_i$ (YFP/OD600) and $C^{\mathrm{RFU}}_i$ (CFP/OD600).

We also include, in every experiment, a **reference strain** with constitutive YFP to serve as an anchor—a per-experiment, per-state reference that removes gain/time/instrument drift while preserving a meaningful **effect size** of fluorescent intensity. Let

$$
A_{\mathrm{experiment},i} := \mathrm{mean}\{\text{references' } Y^{\mathrm{RFU}}_i\}.
$$

#### (a) Logic (fluorophore ratio → log2 → per-design min–max)

Using a dual fluorescent reporter allows us to [separate intrinsic from extrinsic noise](https://pmc.ncbi.nlm.nih.gov/articles/PMC3141918/) in our experiments. The YFP/CFP ratio cancels cell size effects; log2 makes fold-changes symmetric; per-design min–max maps the four states onto a common $[0,1]$ scale so logic shapes are comparable across designs.

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

Absolute intensity carries the **effect size** or **scale** signal but raw RFUs drift by experiment/time. Dividing by a per-experiment reference strain removes this drift while keeping a meaningful, unitless intensity. We store log2 to stabilize the regression and avoid dominance by possible ultra-bright samples; we invert back to linear when ranking samples.

$$
y^{\mathrm{linear}}_i = \frac{Y^{\mathrm{RFU}}_i}{A_{\mathrm{experiment},i} + \alpha}
\quad\text{(unitless, reference-normalized absolute fluorescence)}
$$

$$
y^\star_i = \log_2\!\big(y^{\mathrm{linear}}_i + \delta\big)
$$

**8-vector label (stored under `y_column_name`):**

$$
Y = [\,v_{00}, v_{10}, v_{01}, v_{11},\; y^\star_{00}, y^\star_{10}, y^\star_{01}, y^\star_{11}\,]
$$

### 1.2 Modeling note (median–IQR robust scaling):

Random forest (RF) models can handle mixed targets (our 8-vector: four bounded logic $v$ + four log-intensity $y^\star$), but with low sample counts the per-state log-intensities risks having large variance, letting one state potentially dominate RF-internal split decisions. An affine, monotonic, and reversible median–IQR scaling
 puts the four intensity targets on a comparable scale so early fits aren’t skewed.

* Fit-time transform (applied to all training samples, per state): compute campaign-cumulative training median and IQR for each intensity target $y^\star_i$, then

$$
\tilde{y}^\star_i
= \frac{\,y^\star_i - \mathrm{median}_{\text{train}}(y^\star_i)\,}
         {\max\!\big(\mathrm{IQR}_{\text{train}}(y^\star_i),\,\varepsilon\big)}.
$$

This centers typical values near 0 and makes “1 unit” ≈ one IQR, reducing outlier leverage and balancing targets. (We do not scale $v$; it’s already in $[0,1]$.)

### 1.3 Inference-time inversion (undo scaling → undo log)

After fitting the model, we return predictions to their original, linear intensity units so downstream steps use a meaningful scale. The scaling we applied at fit time is a uniform affine transform (same shift/scale for all samples), so reversing it—and then reversing the log—preserves ordering and simply restores interpretable magnitudes.

$$
\widehat{y}^\star_i
= \widehat{\tilde{y}}^\star_i \,\mathrm{IQR}_{\text{train}}(y^\star_i)
  + \mathrm{median}_{\text{train}}(y^\star_i),
\qquad
\widehat{y}^{\mathrm{linear}}_i
= \max\!\bigl(0,\; 2^{\,\widehat{y}^\star_i} - \delta\bigr).
$$

**Note on $\max(0,\cdot)$ and $\delta$:** The $\delta>0$ term is the same small offset used when taking $\log_2(y^{\mathrm{linear}}+\delta)$ to avoid $\log(0)$. Subtracting $\delta$ undoes that offset; the outer $\max(0,\cdot)$ guards against tiny negative values from numerical round-off, ensuring the recovered $\widehat{y}^{\mathrm{lin}}_i$ remains non-negative.

---

### 2. Inputs to the objective (selection time)

* **Predictions:** $\widehat{v}\in[0,1]^4$ and $\widehat{y}^{\mathrm{linear}}\in\mathbb{R}_{\ge 0}^4$.
* **Setpoint** (i.e., preference): $p\in[0,1]^4$. This can be binary setpoints (e.g., AND: $[0,0,0,1]$) or nuanced continuous ones (e.g., $[0.3,0.4,0.7,0.2]$).

Each candidate is scored with $p$, $\widehat{v}$, and $\widehat{y}^{\mathrm{linear}}$.

---

### 3. Logic fidelity

We compare the predicted logic vector $\widehat{v}\in[0,1]^4$ to the setpoint $p\in[0,1]^4$ using **root-mean-square error (RMSE)**, then turn that error into a **similarity** in $[0,1]$: 1 = perfect match, 0 = as bad as it can be for this setpoint.

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
F_{\text{logic}} \leftarrow \mathrm{clip}(F_{\text{logic}},\,0,\,1).
$$

$F_{\text{logic}}=1$ only when $\widehat{v}=p$; $F_{\text{logic}}=0$ at the setpoint’s farthest corner. A value like **0.7** means the joint error is 30% of the worst possible for this $p$ (i.e., you’re 70% of the way from “worst miss” to “perfect match”).

For valid $\widehat{v}\in[0,1]^4$, $\lVert \widehat{v}-p\rVert_2\!\le\! D$ so $F_{\text{logic}}\in[0,1]$ already; the clip only guards tiny numerical drift or predictions slightly outside $[0,1]$. Normalizing by $D$ makes scores directly comparable across binary or continuous setpoints.

---

### 4. Evaluating fluorescent intensity in target conditions

We compute a **weighted average** of the predicted per-state intensity, where the weights come from the **setpoint**—states you care about more get higher weight. Then we **rescale that single number** using only **this round’s labeled data** (i.e., the designs measured in the current active-learning batch and used to fit the model) so it lands in $[0,1]$ and stays resistant to experiment-to-experiment drift.

**Weights from the setpoint (turn setpoint into state weights).**
Let $p\in[0,1]^4$ and $P=\sum_i p_i$. Define

$$
w_i =
\begin{cases}
\dfrac{p_i}{P}, & P>0,\\[4pt]
0, & P=0.
\end{cases}
\quad\text{so that } w_i\ge 0 \text{ and } \sum_i w_i=1.
$$

This makes a simple average: each state’s contribution is proportional to how much the setpoint values it.

**Raw effect (weighted average of predicted intensities).**
With predicted linear intensities $\widehat{y}^{\mathrm{linear}}_i$,

$$
E_{\mathrm{raw}} = \sum_{i=1}^{4} w_i \,\widehat{y}^{\mathrm{linear}}_i
\quad\text{(equivalently } E_{\mathrm{raw}}=\tfrac{p\cdot \widehat{y}^{\mathrm{linear}}}{\max(P,\epsilon)}\text{, with } \epsilon>0 \text{ a small guard).}
$$

Raising intensity where $p_i$ is large **always** increases $E_{\mathrm{raw}}$; intensity where $p_i=0$ does **not**. 

If $P=0$ (an “all-OFF” setpoint), define $w=\mathbf{0}$ and set $E_{\mathrm{raw}}=0$; the score is then fully determined by the logic term.


**Round-internal robust scaling.**
We now map $E_{\mathrm{raw}}$ to $[0,1]$ using only **this round’s labeled designs**:

* The denominator, **$\mathrm{denom}$**, is the **95th percentile** of $\{E_{\mathrm{raw}}\}$ recomputed over the round’s labels under the current setpoint $p$, with a small floor $\epsilon>0$:

  $$
  \mathrm{denom} \;=\; \max\!\Big(\text{95th percentile of } \{E^{\mathrm{(round)}}_{\mathrm{raw}}\},\ \epsilon\Big).
  $$
* **Scaled effect:**

  $$
  E_{\mathrm{scaled}} \;=\; \min\!\Big(1,\ \max\!\big(0,\ \tfrac{E_{\mathrm{raw}}}{\mathrm{denom}}\big)\Big).
  $$

Using the **same-round** labeled set makes the scale **self-calibrating** to that experiment/day; the 95th percentile is robust to a few extreme bright wells (they map to ~1). The realized denominator is a **per-run constant** and must be snapshotted in the **round context / objective meta artifact** (referenced by `run_meta`), not duplicated per-ID. As a result, $E_{\mathrm{scaled}}$ is unit-free, bounded, and comparable **within the round**.


---

### 5. Final scoring metric

$$
\boxed{\;\text{score} = \big(F_{\text{logic}}\big)^{\beta}\cdot \big(E_{\text{scaled}}\big)^{\gamma}\;}
\qquad \beta=\gamma=1.
$$

* $\beta>1$: emphasize logic correctness more strongly.
* $\gamma>1$: emphasize intensity in the desired conditions more strongly.

* **Final product:** Ensures “wrong logic” or “dim in desired states” both depress the score, while “right and bright” is rewarded.

---

### 6. Metric properties

* **Batch robustness.** Logic uses YFP/CFP ratios; intensity uses a **reference-strain anchor**; and selection scales the effect by a **within-round percentile** over labels.
* **Setpoint flexibility.** You can change $p$ easily and start a different campaign; the objective recomputes $E_{\text{raw}}$ using the already predicted $\widehat{y}^{\mathrm{linear}}$.

---

### 7. Concrete examples (state order $[00,10,01,11]$)

#### (a) AND-like setpoint

$p=[0,0,0,1]\Rightarrow w=[0,0,0,1]$.

* Candidate A: $\widehat{y}^{\mathrm{linear}}=[0.1,0.2,0.3,1.2]$ → $E_{\text{raw}}=1.2$ (all credit from $A{+}B$).
* Candidate B: $\widehat{y}^{\mathrm{linear}}=[0.8,0.9,0.7,0.2]$ → $E_{\text{raw}}=0.2$ (intensity in the wrong conditions doesn’t help).
  If B’s $\widehat{v}$ is also far from $p$, $F_{\text{logic}}$ is small → score stays low.

#### (b) Nuanced setpoint

$p=[0.3,0.4,0.7,0.2]\Rightarrow P=1.6,\; w\approx[0.1875,0.25,0.4375,0.125]$.

* Candidate C: $\widehat{y}^{\mathrm{linear}}=[0.2,0.8,0.2,0.8]$ → $E_{\text{raw}}\approx 0.425$.
* Candidate D: $\widehat{y}^{\mathrm{linear}}=[0.1,0.5,1.1,0.4]$ → $E_{\text{raw}}\approx 0.675$.
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

### 9. Emissions

All outputs are namespaced in `events.parquet`.

**Per-ID predictions (`kind="run_pred"`)**

- `pred__y_obj_scalar: double`                    ← the SFXI score used for ranking
- `pred__y_hat_model: list<float>`                ← model-space vector
- `obj__logic_fidelity_l2_norm01: float ∈ [0,1]`  ← \(F_\text{logic}\)
- `obj__effect_scaled: float ∈ [0,1]`             ← \(E_\text{scaled}\)

**Per-run metadata (`kind="run_meta"`)**

- `obj__name="sfxi_v1"`
* `obj__params_hash`
- `sel__score_field="pred__y_obj_scalar"`  ← selection ranks on this field

Objective parameters surfaced to aid auditability without opening artifacts:

  - `obj__logic_exponent_beta: double`
  - `obj__intensity_exponent_gamma: double`
  - `obj__log2_offset_delta: double` (the delta used to invert log2)
  - `obj__setpoint_vec4: list<double>[4]`
  - `obj__scale_percentile_p: int` (e.g., 95)
  - `obj__scale_fallback_p: int`
  - `obj__scale_min_n: int`
  - `obj__scale_eps: double`
  - `obj__scale_denom_value: double`


**Recomputable at runtime (not persisted per-ID)**

- `E_raw = dot(w, yhat_linear)` where `yhat_linear` is recovered by **inverting** `pred__y_hat_model[4:8]` using transforms in `run_meta` (target normalizer stats + `obj__log2_offset_delta`).
- `F_logic` from `v_hat = pred__y_hat_model[0:4]` and the setpoint in `run_meta`.
- `D` (worst-case distance) and weights `w` derived from the setpoint in `run_meta`.

---

@e-south


