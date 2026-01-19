## Motif artifact contract (JSON)

DenseGen can consume **per-motif JSON artifacts** that encode a single PWM. This keeps
DenseGen decoupled from parsing code: any producer (Cruncher or external tooling) can
emit the contract, and DenseGen only reads the artifact path specified in `config.yaml`.
Cruncher produces these artifacts via `cruncher catalog export-densegen`
(implemented in `cruncher/src/app/motif_artifacts.py`).

### Core principles

- **One file per motif** (explicit paths; no directory scanning).
- **JSON-first**, no sidecar schema files.
- **Strict, fail-fast validation** to keep runs deterministic.
- **Both probabilities and log-odds** are required.

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

### Scoring semantics

DenseGen scores sampled candidates using **PWM log-odds** with the provided background.
Probabilities are used for sequence generation; log-odds are used for scoring and
thresholding.

Log-odds values must be **finite** (no infinities). DenseGen assumes log-odds are
computed with the natural log (ln) of `p/background`. If your matrices contain zeros,
apply pseudocounts before emitting artifacts.

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
    {"A": 1.1632, "C": -0.9163, "G": -1.6094, "T": -1.6094},
    {"A": -0.9163, "C": 1.0296, "G": -0.9163, "T": -0.9163},
    {"A": -0.9163, "C": -0.9163, "G": 1.0296, "T": -0.9163}
  ],
  "provenance": {"source_url": "https://example.org", "citation": "Example et al. 2025"}
}
```

### Config usage

In `config.yaml`, reference the artifact explicitly and set sampling behavior there:

```yaml
inputs:
  - name: lexA
    type: pwm_artifact
    path: inputs/artifacts/lexA.json
    sampling:
      strategy: stochastic
      n_sites: 200
      oversample_factor: 5
      score_percentile: 90
      length_policy: exact
```

Exact length is the default. To enable variable length, set `length_policy: range` and
provide `length_range: [min, max]` where `min >= motif_length`.

---

@e-south
