## cruncher

**cruncher** is a nucleic acid sequence design tool for generating short DNA sequences that jointly resemble multiple user-defined transcription factor motifs. Motifs can be represented as [position weight matrices (PWMs)](https://en.wikipedia.org/wiki/Position_weight_matrix).


### Contents

1. [Overview](#overview)
2. [Quickstart](#quickstart)
3. [More documentation](#more-documentation)

---

### Overview

A typical workflow looks like:

1. Fetch motif matrices and/or binding sites from a source (e.g., [RegulonDB](https://regulondb.ccg.unam.mx/), [JASPAR](https://jaspar.elixir.no/), or a local dataset).
2. Lock TF names to exact cached artifacts (motif IDs + hashes) for reproducibility.
3. Generate synthetic sequences (e.g., via [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)) using the locked motifs.
4. Analyze / visualize / report from run artifacts.

Cruncher optimizes short, fixed-length sequences to co-satisfy multiple TF PWMs and
returns a diverse elite set using TFBS-core MMR. For the full intent, lifecycle,
and config/architecture mapping, see `docs/guides/intent_and_lifecycle.md`.

Scoring is **FIMO-like**: cruncher builds log-odds PWMs against a 0-order
background, scans each candidate sequence to find the best window per TF
(optionally bidirectional), and can scale that best hit to a p-value using a
DP-derived null distribution (`score_scale: logp`). For `logp`, the tail
probability for the best window is converted to a sequence-level p via
`p_seq = 1 - (1 - p_win)^n_windows`. This is an internal implementation; cruncher
does not call the FIMO binary.

---

### Quickstart (happy path)

**cruncher** uses **uv** for Python deps and **pixi** for system binaries (MEME Suite).

```bash
# cd into a workspace
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf

# Optional: install system tools (MEME Suite) via pixi
pixi install

# Use pixi to keep MEME Suite on PATH; use uv for Python-only flows.
cruncher() { pixi run cruncher -- "$@"; }
# cruncher() { uv run cruncher "$@"; }

# Quick sanity check: list sources
cruncher sources list

# Local demo cache (no network required)
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR

# Optional: fetch curated sites from RegulonDB (network)
cruncher fetch sites --tf lexA --tf cpxR

# Reproducibility pinning
cruncher lock

# Inspect cached PWMs / logos (optional)
cruncher catalog pwms
cruncher catalog logos --set 1

# Optional: validate locked motifs
cruncher parse

# Optimization (fixed-length PT)
cruncher sample

# Diagnostics + plots (defaults to latest run)
cruncher analyze

# Optional: print a concise summary to stdout
cruncher analyze --summary

```

---

Notes:
- Analysis writes a canonical summary to `summary.json`, a human-readable
  entrypoint to `report.md`, and a detailed artifact manifest to
  `manifest.json` (all under each run directory, typically `outputs/sample/latest/`).
- Motif overlap is a feature, not a failure; overlap plots are descriptive only.

### More documentation

1. [Documentation index](docs/index.md)
2. [Two-TF demo (end-to-end)](docs/demos/demo_basics_two_tf.md)
3. [Campaign demo (multi-TF)](docs/demos/demo_campaigns_multi_tf.md)
4. [Densegen prep demo (three-TF)](docs/demos/demo_densegen_prep_three_tf.md)
5. [MEME Suite setup](docs/guides/meme_suite.md)
6. [Sampling + analysis](docs/guides/sampling_and_analysis.md)
7. [Configure your project](docs/reference/config.md)
8. [Ingesting and caching data](docs/guides/ingestion.md)
9. [CLI reference](docs/reference/cli.md)
10. [Architecture and artifacts](docs/reference/architecture.md)
11. [Package spec (developers)](docs/internals/spec.md)

---

@e-south
