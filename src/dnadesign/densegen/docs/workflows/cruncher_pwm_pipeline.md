## Cruncher -> DenseGen (PWM handoff)

Cruncher exports per-motif JSON artifacts; DenseGen consumes them for Stage-A PWM sampling.
For a full walkthrough, see `docs/demo/demo_basic.md`.

---

### Minimal flow

```bash
# Cruncher workspace
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --hydrate
cruncher lock

# Export into a DenseGen workspace
cruncher catalog export-densegen --set 1 --densegen-workspace demo_meme_two_tf
cruncher catalog export-sites --set 1 --densegen-workspace demo_meme_two_tf
```

---

### DenseGen inputs

- `type: pwm_artifact_set` for PWM artifacts
- `type: binding_sites` for exported site tables (optional)

See `reference/config.md` for exact fields.

---

@e-south
