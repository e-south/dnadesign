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

### 4) (Optional) Build Stage-B libraries (helper)

`dense run` will build libraries automatically when `library_source: build`.
This helper is useful for feasibility inspection or artifact replay workflows.

```bash
dense stage-b build-libraries --overwrite
```

---

### 5) Run generation

```bash
dense run
```

---

### 6) Plot

```bash
dense plot --only stage_a_summary,placement_map

# Run-health is better inspected than plotted:
# dense inspect run --events --library
```

---

### 7) Report (optional)

```bash
dense report --plots include
```

---

### 8) Reset the demo (optional)

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
