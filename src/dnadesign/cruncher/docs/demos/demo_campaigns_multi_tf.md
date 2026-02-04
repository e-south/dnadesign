## cruncher demo 2

**Jointly maximizing a sequence based on >2 TFs, categories, and campaigns.**

This demo walks through a process of running category-based sequence optimization campaigns, with a focus on campaign selection (site counts + PWM quality), derived configs, and multi-TF runs.

Scoring is **FIMO-like** (internal implementation): cruncher uses PWM log‑odds
scanning against a 0‑order background, takes the best window per TF (optionally
both strands), and can convert that best hit to a p‑value via a DP‑derived null
distribution (`score_scale: logp`, with `p_seq = 1 − (1 − p_win)^n_windows`).

### Demo instance

- **Workspace**: `src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/`
- **Config**: `config.yaml`
- **Output root**: `outputs/` (relative to the workspace; runs live under `outputs/<stage>/<run_name>/`)
- **Motif flow**: fetch sites → discover MEME/STREME motifs → lock/sample using those matrices

### Data provenance (demo inputs)

This demo uses multiple local and remote sources so you can see how site merging works beyond RegulonDB:

- **Local DAP-seq motifs + training sites**: O'Malley et al. 2021 (DOI: 10.1038/s41592-021-01312-2), bundled as MEME files under `inputs/local_motifs/` via the `demo_local_meme` source.
- **Local BaeR ChIP-exo binding sites**: Choudhary et al. 2020 (DOI: 10.1128/mSystems.00980-20), processed FASTA in `dnadesign-data/primary_literature/Choudhary_et_al/processed/BaeR_binding_sites.fasta` ingested as a site-only source.
- **Curated/HT sites**: RegulonDB (as configured under `ingest.regulondb`).

### Enter the demo workspace

The demo workspace lives under `src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/`.

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf
CONFIG="$PWD/config.yaml"

# Option B: run from anywhere
CONFIG=src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/config.yaml

# Choose a runner (uv is the default in this repo; pixi is optional).
cruncher() { uv run cruncher "$@"; }
# cruncher() { pixi run cruncher -- "$@"; }

# Optional: widen tables to avoid truncation in rich output.
export COLUMNS=160

# Note: when using pixi tasks, put -c/--config after the subcommand (pixi inserts `--`).
# pixi install

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit -c.
```

Fail fast on external dependencies (MEME Suite):

```bash
cruncher doctor -c "$CONFIG"
```

If it reports missing tools, install MEME Suite (pixi or system install) or set `motif_discovery.tool_path`.
See the [MEME Suite guide](../guides/meme_suite.md) for details.

This workspace also registers the demo local DAP-seq source (`demo_local_meme`)
used in the two-TF demo. It only covers LexA + CpxR; for the full multi-source
walkthrough (curated + DAP-seq sites + merge), start with
[demo_basics_two_tf.md](demo_basics_two_tf.md).

## Campaigns configured in the demo workspace

The demo config includes three campaign flavors so you can see selection and expansion in context:

- `demo_pair` — small two-category campaign (Stress + Envelope), good for a baseline end-to-end run
- `demo_categories` — multi-category expansion across Category1/2/3 (broad coverage)
- `demo_categories_best` — same as `demo_categories` but with selectors (filters by site count / source preference)

Category definitions (from the demo config):

- Category1: CpxR, BaeR
- Category2: LexA, RcdA, Lrp, Fur
- Category3: Fnr, Fur, AcrR, SoxR, SoxS, Lrp

## Inspect categories and campaign expansion (optional)

List targets by category or campaign:

```bash
cruncher targets list --category Category1 -c "$CONFIG"
cruncher targets list --campaign demo_pair -c "$CONFIG"
```

`demo_categories` expands to many sets; use it once your cache is warmed and you are ready for larger runs.

## Validate campaign definitions (no cache required)

Validate campaign wiring (category membership, selectors, and constraints) without hitting the catalog:

```bash
cruncher campaign validate --campaign demo_categories_best --no-selectors --no-metrics -c "$CONFIG"
```

This prints category totals/selected counts plus the campaign_id.

## Fetch curated sites for campaign TFs

Fetch curated RegulonDB sites for all TFs implied by the campaign:

```bash
cruncher fetch sites --campaign demo_categories --no-selectors --update -c "$CONFIG"
```
Output includes per‑TF site counts and any warnings about invalid coordinates.

## Inspect cached inventory + group stats

Summarize cached inventory and check per-set stats:

```bash
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"
cruncher targets stats --campaign demo_pair -c "$CONFIG"
```
These summaries confirm cached counts and site‑length ranges before selection.

For `demo_categories`, the stats table is much larger; use it when you are ready to inspect all expanded sets.

These stats are the first selection signal: **site counts** drive `selectors.min_site_count`, and length ranges
help decide whether to tune `motif_store.site_window_lengths`.

## Evaluate PWM quality (information content)

Compute PWM info-bits for a representative subset before you scale up. This helps decide whether to add a
`selectors.min_info_bits` gate for the campaign:

```bash
cruncher catalog pwms --tf lexA --tf cpxR --source regulondb -c "$CONFIG"
```

If you want to enforce information content, add `selectors.min_info_bits` to the
`demo_categories_best` campaign in `config.yaml` and re-run `campaign validate`.

## Optional: align sites with MEME Suite (recommended before multi-source metrics)

If you want aligned PWMs (e.g., when combining variable-length sites), run MEME Suite
on cached sites first. For the small `demo_pair` baseline, store each tool in its own
`source_id`:

Discovery uses cached sites and writes results under `.cruncher/<workspace>/discoveries/`.
By default it replaces earlier discoveries for the same TF/source to avoid cache bloat.
Use `--keep-existing` to retain historical runs.

```bash
cruncher discover check -c "$CONFIG"
cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme -c "$CONFIG"
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --meme-prior addone --source-id meme_suite_meme -c "$CONFIG"
```

Tip: if each sequence represents one site, prefer MEME with `--meme-mod oops` and `--meme-prior addone`.

How to read the outputs:

- In `discover check`, “ok” means the binary resolved and the version was read.
To compare both tools without manual IDs:

```bash
cruncher catalog pwms --source meme_suite_streme --set 1 -c "$CONFIG"
cruncher catalog pwms --source meme_suite_meme --set 1 -c "$CONFIG"
cruncher catalog logos --source meme_suite_streme --set 1 -c "$CONFIG"
cruncher catalog logos --source meme_suite_meme --set 1 -c "$CONFIG"
```

The demo config already prefers `meme_suite_meme` then `meme_suite_streme`, so after discovery
you can run `cruncher lock` and proceed to sampling without editing IDs.

If you need shorter PWMs for optimization, add a window constraint before sampling:

```yaml
motif_store:
  pwm_window_lengths:
    lexA: 15
    cpxR: 15
```

```bash
cruncher catalog pwms --set 1 -c "$CONFIG"
```

For larger campaigns, run discovery on a curated shortlist first (or per-category),
then re-run `campaign validate --metrics` so selector metrics reflect the aligned motifs.

## Apply selectors and generate a derived config

Apply selectors to keep the strongest candidates and preview what got filtered
(include metrics so min-site-count / info-bit checks can run):

```bash
cruncher campaign validate --campaign demo_categories_best --metrics --show-filtered -c "$CONFIG"
```

Example output:

```bash
   Campaign validation: demo_categories_best
┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Category  ┃ Total ┃ Selected ┃ Filtered TFs ┃
┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Category1 │ 2     │ 1        │ baeR         │
│ Category2 │ 4     │ 4        │ -            │
│ Category3 │ 6     │ 5        │ soxR         │
└───────────┴───────┴──────────┴──────────────┘
campaign_id: 639a75038757d81acbfcc86a61a8a3a58d214f5d187ea4933894708c94fb50f2
```

Use `--no-metrics` if you want to validate category wiring without reading the local catalog.

Generate a derived config from the campaign:

```bash
DERIVED=campaign.demo_categories_best.yaml
cruncher campaign generate --campaign demo_categories_best --out "$DERIVED" -c "$CONFIG"
```

Example output:

```bash
<workspace>/campaign.demo_categories_best.yaml
<workspace>/campaign.demo_categories_best.campaign_manifest.json
```

The derived config lives inside the workspace so outputs remain workspace-scoped. The companion manifest records per-TF metrics (site counts, plus info bits when `selectors.min_info_bits` is enabled).

Example (manifest excerpt; info_bits is `null` unless you enable the selector):

```bash
{
  "targets": {
    "lexA": {"site_count": 49, "info_bits": null},
    "cpxR": {"site_count": 154, "info_bits": null}
  }
}
```

## Run a multi-TF optimization (baseline run)

The base workspace config includes a small 3-TF set (`lexA`, `cpxR`, `fur`) for a baseline end-to-end run. Use it to validate the multi-TF pipeline before running a large campaign-derived config.

If you ran MEME/STREME discovery above, `lock` will prefer those motifs because the demo config
lists `meme_suite_meme`/`meme_suite_streme` first in `source_preference`.

```bash
cruncher lock -c "$CONFIG"
cruncher parse -c "$CONFIG"
cruncher sample -c "$CONFIG"
cruncher sample --no-auto-opt -c "$CONFIG"
cruncher runs list --stage sample -c "$CONFIG"
cruncher runs latest --stage sample --set-index 1 -c "$CONFIG"
cruncher runs best --stage sample --set-index 1 -c "$CONFIG"
```

For diagnostics and tuning guidance, see the
[sampling + analysis guide](../guides/sampling_and_analysis.md).

Example output (runs list, abridged):

```bash
                                                                         Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                                             ┃ Stage  ┃ Status    ┃ Created                          ┃ Motifs ┃ Regulator set      ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 20260110_161355_df8fcb │ sample │ completed │ 2026-01-10T21:13:55.663756+00:00 │ 3      │ set1:lexA,cpxR,fur │ matrix     │
│ 20260110_161221_d63ebd │ sample │ completed │ 2026-01-10T21:12:22.000163+00:00 │ 3      │ set1:lexA,cpxR,fur │ matrix     │
│ 20260110_125635_0affef │ sample │ completed │ 2026-01-10T17:56:39.263742+00:00 │ 3      │ set1:lexA,cpxR,fur │ matrix     │
└──────────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┴────────┴────────────────────┴────────────┘
```

## Inspect run artifacts and outputs

```bash
cruncher runs show 20260110_161355_df8fcb -c "$CONFIG"
```

Example output (abridged):

```bash
run: 20260110_161355_df8fcb
stage: sample
status: completed
created_at: 2026-01-10T21:13:55.663756+00:00
motif_count: 3
regulator_set: {'index': 1, 'tfs': ['lexA', 'cpxR', 'fur']}
pwm_source: matrix
run_dir: <workspace>/outputs/sample/lexA-cpxR-fur_20260110_161355_df8fcb
artifacts:
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Stage  ┃ Type     ┃ Label                                  ┃ Path              ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ sample │ config   │ Resolved config (config_used.yaml)     │ meta/config_used.yaml        │
│ sample │ trace    │ Trace (NetCDF)                         │ artifacts/trace.nc            │
│ sample │ table    │ Sequences with per-TF scores (Parquet) │ artifacts/sequences.parquet   │
│ sample │ table    │ Elite sequences (Parquet)              │ artifacts/elites.parquet      │
│ sample │ json     │ Elite sequences (JSON)                 │ artifacts/elites.json         │
│ sample │ metadata │ Elite metadata (YAML)                  │ artifacts/elites.yaml         │
└────────┴──────────┴────────────────────────────────────────┴───────────────────┘
```

## Render PWM logos (visual QA)

```bash
cruncher catalog logos --set 1 -c "$CONFIG"
```

Example output (abridged; third row omitted):

```bash
Rendered PWM logos
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ Length ┃ Bits  ┃ Output                                                       ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA │ regulondb │ RDBECOLITFC00214 │ 15     │ 10.36 │ /path/to/.../outputs/logos/catalog/set1_lexA-cpxR-fur_20260110_161412_a1b2c3 │
│ cpxR │ regulondb │ RDBECOLITFC00170 │ 11     │ 3.63  │ /path/to/.../outputs/logos/catalog/set1_lexA-cpxR-fur_20260110_161412_a1b2c3 │
└──────┴───────────┴──────────────────┴────────┴───────┴──────────────────────────────────────────────────────────────────────┘
Logos saved to /path/to/.../outputs/logos/catalog/set1_lexA-cpxR-fur_20260110_161412_a1b2c3
```

For live progress during sampling:

```bash
cruncher runs watch <run_name> -c "$CONFIG"
cruncher runs watch <run_name> --plot -c "$CONFIG"
```

## Analyze + report (pairwise plots)

Pairwise plots require a TF pair. For the 3-TF demo run, pick a pair and pass `--tf-pair`:

```bash
cruncher analyze --latest --tf-pair lexA,cpxR -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
```

Example output (analyze):

```bash
Random baseline: 100%|██████████| 25/25 [00:00<00:00, 11676.79it/s]
Analysis outputs → <workspace>/outputs/sample/lexA-cpxR-fur_20260110_161355_df8fcb/analysis
  summary: <workspace>/outputs/sample/lexA-cpxR-fur_20260110_161355_df8fcb/analysis/summary.json
  analysis_id: 20260110T211410Z_1febb3
Next steps:
  cruncher runs show <workspace>/config.yaml 20260110_161355_df8fcb
  cruncher notebook --latest <workspace>/outputs/sample/lexA-cpxR-fur_20260110_161355_df8fcb
  cruncher analyze --summary <workspace>/config.yaml
```

If you're running via `pixi`, prefix those next-step commands with `pixi run cruncher --`.

For a compact diagnostics checklist and tuning guidance, see the
[sampling + analysis guide](../guides/sampling_and_analysis.md).

## Open the run notebook (optional, real-time exploration)

```bash
cruncher notebook --latest <workspace>/outputs/sample/lexA-cpxR-fur_20260110_161355_df8fcb
```

## Optional: campaign-level summary

Aggregate many runs (pairs + facets across runs):

```bash
# Summarize only the analyzed runs (repeat --runs as needed)
cruncher campaign summarize --campaign demo_categories_best --runs "$RUN" -c "$DERIVED"
```

Summary outputs include `campaign_summary.csv`, `campaign_best.csv`, and plots such as `best_jointscore_bar.png`, `tf_coverage_heatmap.png`, `joint_trend.png`, and `pareto_projection.png`.
Use `--skip-missing` if you summarize a larger run set where some analyses are incomplete.

To explore the campaign summary in a notebook (requires the summary artifacts above):

```bash
cruncher campaign notebook --campaign demo_categories_best -c "$DERIVED"
```

## Notes

- Large campaigns can generate many regulator sets. If you need to reduce runtime, trim
  `regulator_sets` in the generated config to a smaller subset before sampling.
- If you want a default TF pair for repeated analyses, set `analysis.tf_pair`
  in the generated config; otherwise pass `--tf-pair` on each analyze call.
- `selectors.min_info_bits` requires PWMs to be buildable. If you switch to
  `pwm_source: sites` and your site lengths vary, set `motif_store.site_window_lengths`
  per TF (or stay in matrix mode). Discovery uses raw sites unless
  `motif_discovery.window_sites=true`.
- The demo config pre-populates `site_window_lengths` for the expanded TF list so
  site-derived PWMs are available if you opt in. This does **not** affect MEME/STREME
  discovery unless you enable `motif_discovery.window_sites=true`.

## Where outputs live

- `<out_dir>/` - this demo writes to `outputs/`.
- Campaign summaries land under `outputs/campaigns/<campaign_id>/`.

## See also

- Command reference: [CLI reference](../reference/cli.md)
- Config knobs: [Config reference](../reference/config.md)
- Ingestion details: [Ingestion guide](../guides/ingestion.md)
