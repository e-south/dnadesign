## Motif artifact contract (JSON)

DenseGen can consume **per‑motif JSON artifacts** that encode a single PWM. This keeps DenseGen decoupled from parsing code e.g., from Cruncher, and DenseGen only reads the artifact path specified in `config.yaml`. Cruncher produces these artifacts via `cruncher catalog export-densegen` (implemented in `cruncher/src/app/motif_artifacts.py`).

### Contents
- [Context](#context) - why artifacts exist and where they fit.
- [Core principles](#core-principles) - contract invariants.
- [Required fields](#required-fields) - strict JSON keys.
- [Scoring semantics](#scoring-semantics) - log‑odds + background.
- [Example artifact](#example-artifact) - minimal JSON payload.
- [Config usage](#config-usage) - Stage‑A sampling entry point.

---

### Context

Artifact‑first PWM inputs are a decoupling contract: producers generate stable, versioned JSON, and DenseGen consumes them. This enables independent producers, reproducible ingestion, and clear provenance. DenseGen uses these artifacts in **Stage‑A sampling** to build TFBS pools from the PWM matrices.

---

### Core principles

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

### Scoring semantics

DenseGen scores sampled candidates via **FIMO log-odds** using the PWM probabilities and
background. The `log_odds` field is validated and stored for provenance, but scoring is
performed by FIMO on the motif probabilities.

Log-odds values must be **finite** (no infinities). Provide log-odds as **log2(p/bg)**
to align with FIMO’s score scale. If your matrices contain zeros, apply pseudocounts
before emitting artifacts.

---

### Example artifact

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
      n_sites: 200
      mining:
        batch_size: 5000
        budget:
          mode: tier_target
          target_tier_fraction: 0.001
          max_candidates: 200000
      length:
        policy: exact
```

Exact length is the default. To enable variable length, set `length.policy: range` and
provide `length.range: [min, max]` where `min >= motif_length`.

---

@e-south
