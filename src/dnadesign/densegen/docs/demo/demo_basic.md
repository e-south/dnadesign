## DenseGen demo (minimal)

This demo runs DenseGen end-to-end in a packaged workspace with two PWM artifacts (LexA + CpxR).

Prereqs: Python deps, MEME Suite (`fimo` on PATH), and a solver backend.

```bash
uv sync --locked
fimo --version
```

---

### 1) Enter the demo workspace

```bash
cd src/dnadesign/densegen/workspaces/demo_meme_two_tf
```

---

### 2) Validate config + solver

```bash
dense validate-config --probe-solver
```

---

### 3) Build Stage-A pools

```bash
dense stage-a build-pool --fresh
```

---

### 4) Run generation

```bash
dense run
```

---

### 5) Plot

```bash
dense plot --only stage_a_summary,stage_b_summary,run_health
```

---

### 6) Report (optional)

```bash
dense report --plots include
```

---

### 7) Reset the demo (optional)

```bash
dense campaign-reset
```

---

If you need a full walkthrough or workflow details, see:
- `guide/sampling.md`
- `guide/generation.md`
- `workflows/cruncher_pwm_pipeline.md`

---

@e-south
