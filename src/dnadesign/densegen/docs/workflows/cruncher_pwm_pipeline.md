## Cruncher to DenseGen PWM workflow

Cruncher exports per‑motif JSON artifacts, and DenseGen consumes them for **Stage‑A sampling**. For a full, progressive walkthrough that uses Cruncher in its own workspace and then hands off to DenseGen, see the [demo](../demo/demo_basic.md).

### Contents
- [Overview](#overview) - what this handoff enables.
- [Minimal operator flow](#minimal-operator-flow) - cache → export → handoff.
- [DenseGen inputs](#densegen-inputs) - Stage‑A consumption points.

---

### Overview

Cruncher produces stable PWM artifacts (one JSON per motif) with explicit background + log‑odds. DenseGen treats these artifacts as a strict contract and uses them in **Stage‑A sampling** to build TFBS pools. Stage‑B sampling remains fully controlled by the DenseGen config.

---

### Minimal operator flow

Run Cruncher in its **own** workspace (Cruncher owns its configs and outputs). Cruncher resolves its config from CWD.

```bash
# From a Cruncher workspace (see cruncher demo docs)
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --hydrate
cruncher lock

# Export to a Cruncher-owned location
cruncher catalog export-densegen --set 1 --out outputs/densegen_motifs
cruncher catalog export-sites --set 1 --out outputs/densegen_sites.parquet
```

Copy the exports into the DenseGen workspace `inputs/` (DenseGen remains config‑centric):

```bash
cp outputs/densegen_sites.parquet <densegen_workspace>/inputs/
cp -R outputs/densegen_motifs <densegen_workspace>/inputs/motif_artifacts
```

---

### DenseGen inputs

Use the exported artifacts in Stage‑A sampling inputs:

- `type: pwm_artifact_set` for per‑motif JSON artifacts.
- `type: binding_sites` for the exported `binding_sites.csv` (optional).

See the config reference for exact fields and Stage‑A sampling knobs.

---

@e-south
