## cruncher

**cruncher** is a nucleic acid sequence optimization tool for designing short DNA sequences that jointly resemble multiple user-defined transcription factor motifs. Motifs can be represented as [position weight matrices (PWMs)](https://en.wikipedia.org/wiki/Position_weight_matrix).


### Contents

1. [Overview](#overview)
2. [Quickstart](#quickstart)
3. [More documentation](#more-documentation)

---

### Overview

A typical workflow looks like:

1. Fetch motif matrices and/or binding sites from a source (e.g., [RegulonDB](https://regulondb.ccg.unam.mx/), [JASPAR](https://jaspar.elixir.no/)).
2. Lock TF names to exact cached artifacts (motif IDs + hashes) for reproducibility.
3. Generate synthetic sequences (e.g., via [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)) using the locked motifs, and then score them iteratively.
4. Analyze / visualize / report from run artifacts.

---

### Quickstart

```bash
# Initialize a workspace or jump into the demo
# cd into a workspace
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf

# Quick sanity check: list sources
cruncher sources list

# Network access (explicit)
cruncher fetch sites --tf lexA --tf cpxR

# Reproducibility pinning
cruncher lock

# Inspect cached PWMs / logos (optional)
cruncher catalog pwms
cruncher catalog logos --set 1

# Optional: validate + render logos
cruncher parse

# Optimization
cruncher sample

# Diagnostics + plots
cruncher analyze --latest

# Report (JSON + Markdown) for a specific run name
cruncher runs list
cruncher report <run_name>
```

---

### More documentation

1. [Documentation index](docs/index.md)
2. [CLI reference](docs/reference/cli.md)
3. [Two-TF end-to-end demo](docs/demos/demo_basics_two_tf.md)
4. [Campaign demo (multi-TF)](docs/demos/demo_campaigns_multi_tf.md)
5. [Configuring a project](docs/reference/config.md)
6. [Ingesting and caching external data](docs/guides/ingestion.md)
7. [Architecture and artifacts](docs/reference/architecture.md)
8. [Package specification for developers](docs/internals/spec.md)

---

@e-south
