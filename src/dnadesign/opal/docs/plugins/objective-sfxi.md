## setpoint_fidelity_x_intensity `sfxi`

This page documents `sfxi_v1` objective behavior, equations, and emitted channels.
It explains how OPAL converts vec8 model outputs into selection-ready score and uncertainty channels.

### At a glance (plugin contract)

- Objective plugin: `sfxi_v1`
- Input shape: `y_pred` vec8 = `[logic(4), log2_intensity(4)]` in state order `[00,10,01,11]`
- Primary score channel: `sfxi` (maximize)
- Additional score channels: `logic_fidelity`, `effect_scaled`
- Uncertainty channel key (when available): `sfxi` (standard deviation of the scalar score)
- Uncertainty methods: `delta` (gradient delta-method) and `analytical` (restored analytical variance path)
- Uncertainty method parameter values: only `delta` or `analytical` are accepted
- Uncertainty method gating: `analytical` is valid only when `logic_exponent_beta == 1` and `intensity_exponent_gamma == 1` (exact equality)
- Uncertainty method default: if `uncertainty_method` is omitted or null, OPAL uses `analytical` when `beta=gamma=1`, otherwise `delta`
- Uncertainty with missing model std: when `y_pred_std` is absent, no uncertainty channel is emitted and method selection does not affect score computation
- Uncertainty with model std: required `y_pred_std` entries must be strictly `> 0`; non-positive required entries fail fast
- Uncertainty output contract: emitted scalar uncertainty is finite and strictly `> 0`, otherwise OPAL fails fast
- Std semantics: `y_pred_std` is interpreted as a standard deviation in objective units and may exceed `1`
- Scaling source: denominator is computed from current-round observed labels and persisted in run metadata
- Strictness: run fails if current-round labels are fewer than `scaling.min_n`
- Selection wiring:
  - Top-N: `selection.params.score_ref = "sfxi_v1/sfxi"`
  - EI: `selection.params.score_ref = "sfxi_v1/sfxi"` and `selection.params.uncertainty_ref = "sfxi_v1/sfxi"`

---

### 1. What the model predicts

The model predicts an **8-vector (Ŷ)** per input sequence (kept as `pred__y_hat_model`). The first four entries describe the **shape** of a two-factor logic response (bounded from 0 to 1). The last four capture **absolute fluorescent intensity** per state, but stored in **log2 space** for modeling stability.

$$
\underbrace{v_{00}, v_{10}, v_{01}, v_{11}}_{\text{logic in }[0,1]^4}\,\
\underbrace{y^{\star}_{00}, y^{\star}_{10}, y^{\star}_{01}, y^{\star}_{11}}_{\text{log2(abs. fluorescent intensity)}}
$$

* $v \in [0,1]^4$: **observed logic profile** in state order $[00,10,01,11]$.
* $y^\star \in \mathbb{R}^4$: **per-state absolute fluorescent intensity** in log2 space.

### 1.1 From experimental data → 8-vector

We start from raw fluorescent readouts for each state $i$:
$Y^{\mathrm{RFU}}_i$ (YFP/OD600) and $C^{\mathrm{RFU}}_i$ (CFP/OD600).

We also include, in every experiment, a **reference strain** with constitutive YFP to serve as an anchor—a per-experiment, per-state reference that removes gain/time/instrument drift while preserving a meaningful **effect size** of fluorescent intensity. Let

$$
A_{\mathrm{experiment},i} := \mathrm{mean}\{\text{references' } Y^{\mathrm{RFU}}_i\}
$$

### (a) Logic (fluorophore ratio → log2 → per-design min–max)

Using a dual fluorescent reporter allows us to [separate intrinsic from extrinsic noise](https://pmc.ncbi.nlm.nih.gov/articles/PMC3141918/) in our experiments. The YFP/CFP ratio cancels cell size effects; log2 makes fold-changes symmetric; per-design min–max maps the four states onto a common $[0,1]$ scale so logic shapes are comparable across designs.

$$
r_i = \frac{Y^{\mathrm{RFU}}_i + \varepsilon}{C^{\mathrm{RFU}}_i + \varepsilon}
\qquad
u_i = \log_2(r_i)
$$

$$
u_{\min}=\min_i u_i\quad u_{\max}=\max_i u_i
$$

$$
v_i = \frac{u_i - u_{\min}}{(u_{\max}-u_{\min})+\eta} \in [0,1]
$$

If $u_{\max}\approx u_{\min}$ (flat logic), set $v_i=\tfrac{1}{4}$ for all $i$ and **warn**. ($\varepsilon,\eta>0$ are small stabilizers and are recorded in metadata.)

### (b) Absolute fluorescent intensity (reference-normalized → log2)

Absolute intensity carries the **effect size** or **scale** signal but raw RFUs drift by experiment/time. Dividing by a per-experiment reference strain removes this drift while keeping a meaningful, unitless intensity. We store log2 to stabilize the regression and avoid dominance by possible ultra-bright samples; we invert back to linear when ranking samples.

$$
y^{\mathrm{linear}}_i = \frac{Y^{\mathrm{RFU}}_i}{A_{\mathrm{experiment},i} + \alpha}
\quad\text{(unitless, reference-normalized absolute fluorescence)}
$$

$$
y^\star_i = \log_2\ \big(y^{\mathrm{linear}}_i + \delta\big)
$$

**8-vector label (stored under `y_column_name`):**

$$
Y = [v_{00}, v_{10}, v_{01}, v_{11}, y^\star_{00}, y^\star_{10}, y^\star_{01}, y^\star_{11}]
$$

### 1.2 Modeling note (median–IQR robust scaling):

Random forest (RF) models can handle mixed targets (our 8-vector: four bounded logic $v$ + four log-intensity $y^\star$), but with low sample counts the per-state log-intensities risks having large variance, letting one state potentially dominate RF-internal split decisions. An affine, monotonic, and reversible median–IQR scaling puts the four intensity targets on a comparable scale so early fits aren’t skewed.

* Fit-time transform (applied to all training samples, per state): compute campaign-cumulative training median and IQR for each intensity target $y^\star_i$, then

$$
\tilde{y}^\star_i
= \frac{\,y^\star_i - \mathrm{median}_{\text{train}}(y^\star_i)\,}
         {\max\ \big(\mathrm{IQR}_{\text{train}}(y^\star_i),\varepsilon\big)}
$$

This centers typical values near 0 and makes “1 unit” ≈ one IQR, reducing outlier leverage and balancing targets. (We do not scale $v$; it’s already in $[0,1]$.)

### 1.3 Inference-time inversion (undo scaling → undo log)

After fitting the model, we return predictions to their original, linear intensity units so downstream steps use a meaningful scale. The scaling we applied at fit time is a uniform affine transform (same shift/scale for all samples), so reversing it—and then reversing the log—preserves ordering and simply restores interpretable magnitudes.

$$
\widehat{y}^\star_i = \widehat{\tilde{y}}^\star_i \,\mathrm{IQR}_{\text{train}}(y^\star_i)+\mathrm{median}_{\text{train}}(y^\star_i) \qquad \widehat{y}^{\mathrm{linear}}_i = \max\!\bigl(0,\; 2^{\,\widehat{y}^\star_i} - \delta\bigr)
$$

**Note on $\max(0,\cdot)$ and $\delta$:** The $\delta>0$ term is the same small offset used when taking $\log_2(y^{\mathrm{linear}}+\delta)$ to avoid $\log(0)$. Subtracting $\delta$ undoes that offset; the outer $\max(0,\cdot)$ guards against tiny negative values from numerical round-off, ensuring the recovered $\widehat{y}^{\mathrm{lin}}_i$ remains non-negative.

---

### 2. Inputs to the objective (selection time)

* **Predictions:** $\widehat{v}\in[0,1]^4$ and $\widehat{y}^{\star}\in\mathbb{R}^{4}$ (log2 intensity block from the vec8 model output).
* **Setpoint** (i.e., preference): $p\in[0,1]^4$. This can be binary setpoints (e.g., AND: $[0,0,0,1]$) or nuanced continuous ones (e.g., $[0.3,0.4,0.7,0.2]$).

Each candidate is scored with $p$, $\widehat{v}$, and $\widehat{y}^{\star}$; OPAL converts $\widehat{y}^{\star}$ to linear intensity internally before computing effect terms.

---

### 3. Logic fidelity

We compare the predicted logic vector $\widehat{v}\in[0,1]^4$ to the setpoint $p\in[0,1]^4$ using **root-mean-square error (RMSE)**, then turn that error into a **similarity** in $[0,1]$: 1 = perfect match, 0 = as bad as it can be for this setpoint.

**“Worst-case” error for this setpoint.** Inside the unit 4-cube $[0,1]^4$, the farthest point from $p$ is a **corner**. For each state $i$, choose whichever of $\{0,1\}$ is farther from $p_i$
(e.g., if $p_i=0.6$, distance to 0 is $0.6$, to 1 is $0.4$ ⇒ pick **0**).
That corner’s Euclidean distance from $p$ is

$$
D \;=\; \sqrt{\sum_{i=1}^{4} \max\!\big(p_i^2,\,(1-p_i)^2\big)}
$$

Examples: $p=[0,0,0,1]\Rightarrow D=2$; $p=[0.5,0.5,0.5,0.5]\Rightarrow D=1$

**Normalized RMSE → similarity.** With four states, $\mathrm{RMSE}(\widehat{v},p)=\tfrac{1}{2}\lVert \widehat{v}-p\rVert_2$ and $\mathrm{RMSE}_{\max}=\tfrac{D}{2}$

Report

$$
F_{\text{logic}}
\;=\; 1 \;-\; \frac{\mathrm{RMSE}(\widehat{v},p)}{\mathrm{RMSE}_{\max}}
\;=\; 1 \;-\; \frac{\lVert \widehat{v}-p\rVert_2}{D}
\qquad
F_{\text{logic}} \leftarrow \mathrm{clip}(F_{\text{logic}},\,0,\,1)
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
\dfrac{p_i}{P}, & P>0\\[4pt]
0, & P=0
\end{cases}
\quad\text{so that } w_i\ge 0 \text{ and } \sum_i w_i=1
$$

This makes a simple average: each state’s contribution is proportional to how much the setpoint values it.

**Raw effect (weighted average of predicted intensities).**
With predicted linear intensities $\widehat{y}^{\mathrm{linear}}_i$

$$
E_{\mathrm{raw}} = \sum_{i=1}^{4} w_i \,\widehat{y}^{\mathrm{linear}}_i
\quad\text{(equivalently } E_{\mathrm{raw}}=\tfrac{p\cdot \widehat{y}^{\mathrm{linear}}}{\max(P,\epsilon)}\text{, with } \epsilon>0 \text{ a small guard)}
$$

Raising intensity where $p_i$ is large **always** increases $E_{\mathrm{raw}}$; intensity where $p_i=0$ does **not**.

If $P=0$ (an “all-OFF” setpoint), define $w=\mathbf{0}$ and set $E_{\mathrm{raw}}=0$; the score is then fully determined by the logic term.


**Round-internal robust scaling.**
We now map $E_{\mathrm{raw}}$ to $[0,1]$ using only **this round’s labeled designs**:

* The denominator, **$\mathrm{denom}$**, is the **`scaling.percentile`th percentile** (default 95) of $\{E_{\mathrm{raw}}\}$ recomputed over the round’s labels under the current setpoint $p$, with a small floor $\epsilon>0$:

  $$
  \mathrm{denom} \;=\; \max\!\Big(\text{95th percentile of } \{E^{\text{(round)}}_{\mathrm{raw}}\},\ \epsilon\Big)
  $$
* **Scaled effect:**

  $$
  E_{\mathrm{scaled}} \;=\; \min\!\Big(1,\ \max\!\big(0,\ \tfrac{E_{\mathrm{raw}}}{\mathrm{denom}}\big)\Big)
  $$

Using the **same-round** labeled set makes the scale **self-calibrating** to that experiment/day; the 95th percentile is robust to a few extreme bright wells (they map to ~1). The realized denominator is a **per-run constant** and must be snapshotted in the **round context / objective meta artifact** (referenced by `run_meta`), not duplicated per-ID. As a result, $E_{\mathrm{scaled}}$ is unit-free, bounded, and comparable **within the round**.

**Strictness:** The objective requires at least `scaling.min_n` labeled designs in the **current round**. If there are fewer, the run fails with a clear error (no silent fallbacks). Lower `scaling.min_n` or add labels to proceed.


---

### 5. Final scoring metric

$$
\boxed{\;\text{score} = \big(F_{\text{logic}}\big)^{\beta}\cdot \big(E_{\text{scaled}}\big)^{\gamma}\;}
\qquad \beta=\gamma=1
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

### (a) AND-like setpoint

$p=[0,0,0,1]\Rightarrow w=[0,0,0,1]$.

* Candidate A: $\widehat{y}^{\mathrm{linear}}=[0.1,0.2,0.3,1.2]$ → $E_{\text{raw}}=1.2$ (all credit from $A{+}B$).
* Candidate B: $\widehat{y}^{\mathrm{linear}}=[0.8,0.9,0.7,0.2]$ → $E_{\text{raw}}=0.2$ (intensity in the wrong conditions doesn’t help).
  If B’s $\widehat{v}$ is also far from $p$, $F_{\text{logic}}$ is small → score stays low.

### (b) Nuanced setpoint

$p=[0.3,0.4,0.7,0.2]\Rightarrow P=1.6,\; w\approx[0.1875,0.25,0.4375,0.125]$.

* Candidate C: $\widehat{y}^{\mathrm{linear}}=[0.2,0.8,0.2,0.8]$ → $E_{\text{raw}}\approx 0.425$.
* Candidate D: $\widehat{y}^{\mathrm{linear}}=[0.1,0.5,1.1,0.4]$ → $E_{\text{raw}}\approx 0.675$.
  Similar total signal; D wins because it is intense in the **high-$p$** state (01).

### (c) “All-OFF” setpoint

$p=[0,0,0,0]\Rightarrow w=0\Rightarrow E_{\text{raw}}=0$.
Only proximity of $\widehat{v}$ to $p$ (being OFF everywhere) is rewarded.

---

### 8. Edge cases and guards
* **Tiny CFP or anchor:** add $\varepsilon,\alpha$.
* **Flat logic:** if $u_{\max}\approx u_{\min}$, set $v=\tfrac{1}{4}\mathbf{1}$.
* **Non-finite:** reject at ingestion.
* **Too few labels in round:** objective errors; lower `scaling.min_n` or add labels.
* **Analytical uncertainty constraints:** if `uncertainty_method=analytical`, OPAL fails fast unless `logic_exponent_beta=1` and `intensity_exponent_gamma=1`.
  Config parsing also rejects this invalid combination before runs start.
* **Uncertainty positivity:** if required model std entries are non-positive, or computed scalar uncertainty is non-positive, OPAL fails fast with a clear error.
* **Fractional-exponent derivatives:** for `0 < logic_exponent_beta < 1` or `0 < intensity_exponent_gamma < 1`, delta-method derivatives are singular at base `0`; OPAL fails fast if `F_logic <= 0` (beta case) or `E_scaled <= 0` (gamma case).
* **Analytical scope:** analytical uncertainty follows the `c5666a7` closed-form direction with log2-space moment correction (`ln(2)` scaling in the intensity moments), modeling `X = 2^Y` moments in log2 space. It remains a closed-form approximation and does not fully re-derive all nonlinear score branches (`max(0, 2^y - delta)` and clipping).

---

### 9. Emissions

All outputs are written to **ledger sinks** under `outputs/ledger/`:

- `outputs/ledger/predictions/` → per‑ID `run_pred` rows (one per candidate)
- `outputs/ledger/runs.parquet` → per‑run `run_meta` row (one per run)
- `outputs/ledger/labels.parquet` → per‑label `label` rows (one per ingest event)

Selection channels and diagnostics are distinct:

- channel refs (`score_ref`, `uncertainty_ref`) resolve only objective channels
- `obj__*` columns are analysis diagnostics and are not selectable channel refs

**Per‑ID predictions (`run_pred` rows)**

- `pred__score_selected: double` — the SFXI score used for ranking
- `pred__uncertainty_selected: double` — selected uncertainty channel (SFXI score standard deviation when available)
- `pred__y_hat_model: list<float>` — **objective‑space** vector (after Y‑ops inversion)
- `obj__logic_fidelity: float ∈ [0,1]` — $F_{\text{logic}}$
- `obj__effect_raw: float` — weighted intensity before scaling
- `obj__effect_scaled: float ∈ [0,1]` — $E_{\text{scaled}}$
- `obj__clip_lo_mask`, `obj__clip_hi_mask` — clipping flags for $E_{\text{scaled}}$
- `sel__rank_competition`, `sel__is_selected` — ranking + selection flags

**Per‑run metadata (`run_meta` row)**

- `objective__name = "sfxi_v1"`
- `objective__params` — includes setpoint, exponents, scaling config, log2 delta
- `objective__summary_stats` — includes `denom_used`, `denom_percentile`, score min/median/max, clip fractions
- `objective__denom_value`, `objective__denom_percentile` — convenience mirrors
- `selection__score_ref` — selected score channel ref (for this demo: `"sfxi_v1/sfxi"`)
- `selection__uncertainty_ref` — selected uncertainty channel ref (EI flows only)
- `selection__objective` (stores objective mode: `maximize|minimize`), `selection__tie_handling`
- `training__y_ops` — Y‑ops applied at fit time (inverted before objective)

`round_ctx.json` also records:

- `objective/sfxi_v1/denom_value`
- `objective/sfxi_v1/denom_percentile`
- full plugin contract audit trail under `core/contracts/...`

**Recomputable at runtime (not persisted per‑ID)**

- $E_{\text{raw}}$ from `pred__y_hat_model[4:8]` using the setpoint‑derived weights
- $F_{\text{logic}}$ from `pred__y_hat_model[0:4]` and the setpoint
- $D$ (worst‑case distance) from the setpoint
- $E_{\text{scaled}}$ via `objective__denom_value` (from `run_meta` or `round_ctx.json`)

---

### Appendix

**Why an 8-vector and not a 4-vector?**

The 8‑vector is minimal and justified for the above stated objective. It cleanly separates:

* **Shape (right):** ratio‑based, effect size‑invariant logic.
* **Scale (bright):** reference‑anchored absolute intensity.

To compute that, we keep 8 numbers per design:

* **4 logic numbers (`v`)** — the **shape** of the response, built from the **YFP/CFP ratio** and min–max scaled across the four states. The ratio cancels extrinsic noise.
* **4 intensity numbers (`y*`)** — the **absolute brightness** per state, anchored to a reference strain and stored in log2 for modeling stability.

One cannot recover the ratio‑based logic (`v`) from YFP intensities alone. If you try to create a “logic pattern” by min–max scaling YFP across states, you keep state‑specific capacity shifts that the YFP/CFP ratio is designed to remove. Dropping to 4 intensities asks the model to infer information that isn’t in the data; it will compute a different score.

---
