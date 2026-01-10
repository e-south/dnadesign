## cruncher

**cruncher** is a nucleic acid sequence optimization tool for designing short DNA
sequences that score highly against multiple user-defined transcription factor
motifs. Motifs can be represented as [position weight matrices (PWMs)](https://en.wikipedia.org/wiki/Position_weight_matrix).


### Contents

1. [Overview](#overview)
2. [Quickstart](#quickstart)
3. [More documentation](#more-documentation)

---

### Overview

A typical workflow looks like:

1. Fetch motif matrices and/or binding sites from a source (e.g., [RegulonDB](https://regulondb.ccg.unam.mx/), [JASPAR](https://jaspar.elixir.no/)).
2. Lock TF names to exact cached artifacts (motif IDs + hashes) for reproducibility.
3. Sample sequences (e.g., [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)) using the locked motifs.
4. Analyze / visualize / report from run artifacts.

---

### Quickstart

```bash
# Initialize a workspace or jump into the demo
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo

# Option B: run from anywhere
cruncher --workspace demo sources list

# Quick sanity check: list sources
cruncher sources list

# Network access (explicit)
cruncher fetch sites --tf lexA --tf cpxR

# Reproducibility pinning
cruncher lock

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

1. [CLI reference](docs/cli.md)
2. [Two-TF end-to-end demo](docs/demo.md)
3. [Category campaign demo](docs/demo_campaigns.md)
4. [Configuring a project](docs/config.md)
5. [Ingesting and caching external data](docs/ingestion.md)
6. [Architecture and artifacts](docs/architecture.md)
7. [Package specification for developers](docs/spec.md)

---

@e-south
