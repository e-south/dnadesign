## Motif artifact JSON contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


DenseGen can consume one JSON file per motif.
This keeps producer tooling and DenseGen decoupled:
- producer tools own parsing and conversion
- DenseGen only validates and consumes a stable artifact contract

Cruncher can emit these artifacts with `cruncher catalog export-densegen`.

### Contents

This section covers contents.
- [When to use this](#when-to-use-this) - where artifacts fit in DenseGen runs.
- [Contract rules](#contract-rules) - invariants DenseGen validates.
- [Required fields](#required-fields) - strict JSON keys.
- [Scoring rules](#scoring-rules) - log‑odds + background.
- [Example artifact](#example-artifact) - minimal JSON payload.
- [Config usage](#config-usage) - Stage‑A sampling entry point.

---

### When to use this

Artifact‑first PWM inputs are a decoupling contract: producers generate stable, versioned JSON, and DenseGen consumes them. This enables independent producers, reproducible ingestion, and clear provenance. DenseGen uses these artifacts in **Stage‑A sampling** to build TFBS pools from the PWM matrices.

---

### Contract rules

This section lists the rules DenseGen enforces for artifact ingestion.

- **One file per motif** (explicit paths; no directory scanning).
- **JSON-first**, no sidecar schema files.
- **Strict, fail-fast validation** to keep runs deterministic.
- **Both probabilities and log-odds** are required.

---

### Required fields

Top-level JSON object with the following required keys:

- `schema_version` (string) — currently `"1.0"`.
- `producer` (string) — name of the tool that created the artifact (e.g., `"cruncher"`).
- `motif_id` (string) — motif identifier.
- `alphabet` (string) — must be `"ACGT"`.
- `matrix_semantics` (string) — must be `"probabilities"`.
- `background` (object) — A/C/G/T background probabilities (sum to 1).
- `probabilities` (list of objects) — per-position PWM probabilities with A/C/G/T keys.
- `log_odds` (list of objects) — per-position log-odds values with A/C/G/T keys.

Optional keys (ignored by DenseGen but recommended for provenance):

- `tf_name`, `source`, `organism`, `provenance`, `checksums`, `tags`, `length`

---

### Scoring rules

DenseGen scores sampled candidates via **FIMO log-odds** using the PWM probabilities and
background. The `log_odds` field is validated and stored for provenance, but scoring is
performed by FIMO on the motif probabilities.

Log-odds values must be **finite** (no infinities). Provide log-odds as **log2(p/bg)**
to align with FIMO’s score scale. If your matrices contain zeros, apply pseudocounts
before emitting artifacts.

---

### Example artifact

This section shows a minimal valid motif artifact payload.

```json
{
  "schema_version": "1.0",
  "producer": "cruncher",
  "motif_id": "LexA",
  "alphabet": "ACGT",
  "matrix_semantics": "probabilities",
  "background": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
  "probabilities": [
    {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
    {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
    {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1}
  ],
  "log_odds": [
    {"A": 1.6781, "C": -1.3219, "G": -2.3219, "T": -2.3219},
    {"A": -1.3219, "C": 1.4854, "G": -1.3219, "T": -1.3219},
    {"A": -1.3219, "C": -1.3219, "G": 1.4854, "T": -1.3219}
  ],
  "provenance": {"source_url": "https://example.org", "citation": "Example et al. 2025"}
}
```

---

### Config usage

In `config.yaml`, reference the artifact explicitly and set **Stage‑A sampling** behavior there:

```yaml
inputs:
  - name: lexA
    type: pwm_artifact
    path: inputs/artifacts/lexA.json
    sampling:  # Stage‑A sampling
      strategy: stochastic
      n_sites: 250
      mining:
        batch_size: 5000
        budget:
          mode: fixed_candidates
          candidates: 1000000
          growth_factor: 1.25
      selection:
        policy: mmr
        rank_by: score_norm
        alpha: 0.35
        pool:
          min_score_norm: 0.75  # fraction of theoretical max log-odds score
          relevance_norm: minmax_raw_score
      tier_fractions: [0.001, 0.01, 0.09]
      length:
        policy: range
        range: [16, 20]
```

`fixed_candidates` is the recommended mining mode for predictable runtime behavior. Use
`tier_target` only when you intentionally want mining to continue until a tier target is met
or caps/time terminate early.

If `min` is below motif length, Stage‑A trims to a max-information window per candidate.
When you use MMR, cores must have consistent length; set a fixed trimming window if needed.

For end-to-end artifact handoff commands, use **[Cruncher to DenseGen PWM handoff](../howto/cruncher_pwm_pipeline.md)**.
