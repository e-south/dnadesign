# Cruncher CLI

See `docs/README.md` for the docs map and reading order.

Cruncher is structured around an explicit lifecycle: **fetch → lock → parse/sample → analyze/report**.
Each command has one job and never performs hidden network access.

## Command groups

- `parse` — validate cached motifs and render PWM logos.
- `sample` — run MCMC optimization to design candidate sequences.
- `analyze` — generate diagnostics and plots for completed runs.
- `report` — summarize a sample run into report artifacts.
- `notebook` — generate an optional marimo notebook for analysis.
- `fetch` — fetch motifs or binding sites into the local cache.
- `catalog` — inspect cached motifs and binding sites.
- `lock` — resolve TF names to exact cached motif IDs.
- `cache` — inspect or verify cache integrity.
- `config` — summarize effective configuration settings.
- `sources` — list ingestion sources and capabilities.
- `status` — show a bird's-eye view of cache, targets, and runs.
- `targets` — check target readiness and catalog candidates.
- `optimizers` — list available optimizer kernels.
- `runs` — list, inspect, or watch past run artifacts.

## Help and hints

- Every command supports `--help` (for example: `cruncher fetch motifs --help`).
- When a command fails due to missing inputs or ambiguous targets, Cruncher prints a short hint with the next recommended command.

## Core lifecycle

- `cruncher fetch motifs --tf <name> --tf <name> --source regulondb <config>`
- `cruncher fetch motifs --motif-id <id> --source regulondb <config>`
- `cruncher fetch motifs --tf <name> --all <config>` (fetch all candidates)
- `cruncher fetch motifs --tf <name> --dry-run <config>` (preview remote matches)
- `cruncher fetch motifs --tf <name> --offline <config>` (verify cached only)
- `cruncher fetch motifs --tf <name> --update <config>` (force refresh)
- `cruncher fetch sites --tf <name> --source regulondb <config>`
- `cruncher fetch sites --motif-id <id> --source regulondb <config>`
- `cruncher fetch sites --tf <name> --dry-run <config>` (preview HT datasets)
- `cruncher fetch sites --tf <name> --genome-fasta path/to/genome.fna <config>` (hydrate sequences from coordinates)
- `cruncher fetch sites --tf <name> --dataset-id <id> <config>` (limit HT dataset; enables HT for this request)
- `cruncher fetch sites --tf <name> --hydrate <config>` (hydrate cached sites without refetching)
- `cruncher fetch sites --tf <name> --offline <config>`
- `cruncher fetch sites --tf <name> --update <config>`
- `cruncher lock <config>`
- `cruncher status <config>` (overview of cache, targets, and recent runs)
- `cruncher targets list <config>` (show configured TF sets)
- `cruncher targets status <config> [--pwm-source sites] [--site-kind curated]`
- `cruncher targets candidates <config>` (show catalog candidates per TF)
- `cruncher targets candidates --fuzzy [--min-score 0.6] [--limit 10] <config>`
- `cruncher targets stats <config>` (site-length stats + PWM lengths for each target)
- `cruncher parse <config>`
- `cruncher sample <config>`
- `cruncher analyze <config>` (requires `analysis.runs`; use `--run` or `--latest` to override)
- `cruncher analyze --run <sample_run> <config>` (repeat `--run` for multiple)
- `cruncher analyze --latest <config>` (explicitly analyze latest sample run)
- `cruncher analyze --tf-pair lexA,cpxR <config>` (override pairwise plots)
- `cruncher analyze --list-plots <config>` (preview which plots will run)
- `cruncher report <config> <run_name>` (write `report.json` + `report.md`)
- `cruncher notebook [--analysis-id <id>|--latest] <run_dir>` (generate a marimo notebook)
- `cruncher runs list <config> [--stage sample]`
- `cruncher runs list <config> [--stage sample] [--full] [--json]`
- `cruncher runs show <config> <run_name> [--json]`
- `cruncher runs latest <config> [--stage sample] [--json]`
- `cruncher runs watch <config> <run_name>`
- `cruncher runs rebuild-index <config>`

Notes:

- Offline fetch is strict: if multiple cached motifs match a TF, use `--all` (motifs) or `--motif-id` to disambiguate.
- `fetch motifs` and `fetch sites` render a summary table by default; use `--paths` to print raw cache paths.
- If HT sites only provide coordinates, Cruncher hydrates sequences via NCBI by default (`ingest.genome_source=ncbi`).
- Use `--genome-fasta` to override and hydrate against a local FASTA (offline runs).
- `--genome-fasta` overrides `ingest.genome_source` and `ingest.genome_fasta` when provided.
- Use `cruncher targets stats` to inspect site lengths (curated or HT) and set `motif_store.site_window_lengths`.
- When multiple `regulator_sets` are configured, Cruncher runs each set independently and creates separate run folders.
- Each `cruncher analyze` call creates a new `analysis/<analysis_id>/` folder under the sample run directory.
- `cruncher runs show` prints the artifact registry plus analysis IDs, counts, and notebook path (when available).
- `cruncher notebook` requires the optional `marimo` dependency (install via `uv add --group notebooks marimo`).
- Pairwise plots only render when `analysis.tf_pair` is explicitly set.

## Catalog inspection

- `cruncher catalog list <config> [--tf <name>] [--source <source>] [--organism <name>]`
- `cruncher catalog list <config> [--tf <name>] [--include-synonyms]`
- `cruncher catalog search <config> <query> [--source <source>] [--organism <name>] [--regex] [--case-sensitive] [--fuzzy] [--min-score 0.6] [--limit 25]`
- `cruncher catalog resolve <config> <tf> [--source <source>] [--organism <name>] [--include-synonyms]`
- `cruncher catalog show <config> <source>:<motif_id>`
- `cruncher cache stats <config>`
- `cruncher cache verify <config>`

## Source adapters

- `cruncher sources list`
- `cruncher sources info regulondb <config>`
- `cruncher sources datasets regulondb <config> [--tf <name>]`

## Optimizers

- `cruncher optimizers list`

## Config

- `cruncher config summary <config>`

## Global options

- `--log-level INFO|DEBUG` (or set `CRUNCHER_LOG_LEVEL`)
