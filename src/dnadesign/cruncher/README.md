## cruncher

**cruncher** designs short DNA sequences that score highly against a set of
transcription‑factor motifs (PWMs). It keeps data ingestion, optimization logic,
and reporting decoupled so sources and optimizers can evolve independently.

**Workflow:** fetch → lock → parse → sample → analyze → report.

---

### Quick start (RegulonDB example)

Lockfiles are mandatory for parse/sample (no implicit TF resolution). Analyze/report operate on
existing run artifacts and validate the lockfile captured in the run manifest.

```bash
# 1) Populate local catalog
cruncher fetch sites  --tf lexA --tf cpxR src/dnadesign/cruncher/config.yaml
cruncher fetch motifs --tf lexA --dry-run src/dnadesign/cruncher/config.yaml
cruncher catalog list src/dnadesign/cruncher/config.yaml

# 2) Lock TF names
cruncher lock src/dnadesign/cruncher/config.yaml

# 3) Preview targets + motifs, then parse
cruncher targets status src/dnadesign/cruncher/config.yaml
cruncher parse src/dnadesign/cruncher/config.yaml

# 4) Run optimizer
cruncher sample src/dnadesign/cruncher/config.yaml

# 5) Analyze + report
cruncher analyze src/dnadesign/cruncher/config.yaml
cruncher report  src/dnadesign/cruncher/config.yaml sample_<tfset>_<timestamp>
```

Source‑specific details (RegulonDB TLS, HT hydration, windowing rules) live in
`docs/ingestion.md` and `docs/troubleshooting.md`.

---

### What to read next

- `docs/demo.md` — end‑to‑end workflow (LexA + CpxR)
- `docs/cli.md` — concise command reference + examples
- `docs/config.md` — config schema + examples
- `docs/architecture.md` — component boundaries + run artifacts
- `docs/spec.md` — full requirements and design rationale

---

### Outputs at a glance

Each run directory contains:

- `config_used.yaml` — resolved runtime config + PWM summaries
- `run_manifest.json` — provenance, hashes, optimizer stats
- `run_status.json` — live progress (parse/sample)
- `sequences.parquet` — per‑draw sequences + per‑TF scores (if enabled)
- `trace.nc` — ArviZ trace (if enabled)
- `cruncher_elites_*/` — elite sequences (Parquet + JSON + YAML)
- `report.json` / `report.md` — generated summary (report stage)

---

### Project layout

```
dnadesign/
└─ cruncher/
   ├─ cli/        # Typer CLI entry point
   ├─ core/       # PWM/scoring/state/optimizers
   ├─ ingest/     # Source adapters + normalization
   ├─ store/      # Catalog cache + lockfiles
   ├─ services/   # fetch/lock/catalog/targets services
   ├─ workflows/  # parse/sample/analyze/report orchestration
   ├─ io/         # parsers + plots
   ├─ config/     # v2 config schema + loader
   ├─ docs/       # user and developer docs
   └─ tests/      # unit + integration tests
```
```
