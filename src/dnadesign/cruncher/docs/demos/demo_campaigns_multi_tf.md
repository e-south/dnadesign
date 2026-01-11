## cruncher demo 2

**Jointly maximizing a sequence based on >2 TFs, categories, and campaigns.**

This demo walks through a process of running category-based sequence optimization campaigns, with a focus on campaign selection (site counts + PWM quality), derived configs, and multi-TF runs.

### Demo instance

- **Workspace**: `src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/`
- **Config**: `config.yaml`
- **Output root**: `runs/` (relative to the workspace)

### Enter the demo workspace

The demo workspace lives here:

- `src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/`

You can either `cd` into the workspace or run from anywhere. This demo uses its
own workspace so its outputs stay isolated.

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf
CONFIG=config.yaml

# Option B: run from anywhere
CONFIG=src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/config.yaml

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit -c.
# Commands are shown as `uv run cruncher ...` so they work out of the box.
```

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
uv run cruncher -c "$CONFIG" targets list --category Category1
uv run cruncher -c "$CONFIG" targets list --campaign demo_pair
```

Example output (Category1):

```bash
   Category
   targets:
  Category1
┏━━━━━┳━━━━━━┓
┃ Set ┃ TF   ┃
┡━━━━━╇━━━━━━┩
│   1 │ cpxR │
│   1 │ baeR │
└─────┴──────┘
```

Example output (demo_pair campaign):

```bash
   Campaign
   targets:
  demo_pair
┏━━━━━┳━━━━━━┓
┃ Set ┃ TF   ┃
┡━━━━━╇━━━━━━┩
│   1 │ lexA │
│   1 │ cpxR │
└─────┴──────┘
```

`demo_categories` expands to many sets; use it once your cache is warmed and you are ready for larger runs.

## Validate campaign definitions (no cache required)

Validate campaign wiring (category membership, selectors, and constraints) without hitting the catalog:

```bash
uv run cruncher -c "$CONFIG" campaign validate --campaign demo_categories_best --no-selectors --no-metrics
```

Example output:

```bash
      Campaign validation:
      demo_categories_best
┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
┃ Category  ┃ Total ┃ Selected ┃
┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
│ Category1 │ 2     │ 2        │
│ Category2 │ 4     │ 4        │
│ Category3 │ 6     │ 6        │
└───────────┴───────┴──────────┘
campaign_id: bcca78ce83989eb24a25f5bc93ebda1e02d118420475ed778a35352e591242af
```

## Fetch curated sites for campaign TFs

Fetch curated RegulonDB sites for all TFs implied by the campaign:

```bash
uv run cruncher -c "$CONFIG" fetch sites --campaign demo_categories --no-selectors --update
```

Example output (abridged, INFO log level):

```bash
16:07:02 INFO     Fetching binding sites from regulondb for TFs=['acrR', 'baeR', 'cpxR', 'fnr', 'fur', 'lexA', 'lrp', 'rcdA', 'soxR', 'soxS'] motif_ids=[]
         INFO     Fetching binding sites for TF 'acrR'
         INFO     Fetching binding sites for TF 'baeR'
16:07:03 INFO     Fetching binding sites for TF 'cpxR'
16:07:04 WARNING  Skipping curated site RDBECOLIBSC04560: invalid curated binding-site coordinates
         INFO     Fetching binding sites for TF 'fnr'
16:07:05 INFO     Fetching binding sites for TF 'fur'
16:07:06 INFO     Fetching binding sites for TF 'lexA'
         INFO     Fetching binding sites for TF 'lrp'
16:07:08 INFO     Fetching binding sites for TF 'rcdA'
         INFO     Fetching binding sites for TF 'soxR'
         INFO     Fetching binding sites for TF 'soxS'
                                         Fetched binding-site sets
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ Kind    ┃ Dataset ┃ Method ┃ Sites ┃ Total ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ lrp  │ regulondb │ RDBECOLITFC00014 │ curated │ -       │ -      │ 219   │ 219   │ 14.7     │ 2026-01-10 │
│ rcdA │ regulondb │ RDBECOLITFC00048 │ curated │ -       │ -      │ 15    │ 15    │ 10.0     │ 2026-01-10 │
│ acrR │ regulondb │ RDBECOLITFC00065 │ curated │ -       │ -      │ 11    │ 11    │ 18.9     │ 2026-01-10 │
│ soxR │ regulondb │ RDBECOLITFC00071 │ curated │ -       │ -      │ 7     │ 7     │ 18.0     │ 2026-01-10 │
│ fur  │ regulondb │ RDBECOLITFC00093 │ curated │ -       │ -      │ 217   │ 217   │ 18.6     │ 2026-01-10 │
│ fnr  │ regulondb │ RDBECOLITFC00128 │ curated │ -       │ -      │ 152   │ 152   │ 14.3     │ 2026-01-10 │
│ cpxR │ regulondb │ RDBECOLITFC00170 │ curated │ -       │ -      │ 154   │ 154   │ 15.3     │ 2026-01-10 │
│ baeR │ regulondb │ RDBECOLITFC00182 │ curated │ -       │ -      │ 4     │ 4     │ 20.0     │ 2026-01-10 │
│ soxS │ regulondb │ RDBECOLITFC00201 │ curated │ -       │ -      │ 44    │ 44    │ 20.0     │ 2026-01-10 │
│ lexA │ regulondb │ RDBECOLITFC00214 │ curated │ -       │ -      │ 49    │ 49    │ 19.5     │ 2026-01-10 │
└──────┴───────────┴──────────────────┴─────────┴─────────┴────────┴───────┴───────┴──────────┴────────────┘
```

## Inspect cached inventory + group stats

Summarize cached inventory and check per-set stats:

```bash
uv run cruncher -c "$CONFIG" sources summary --source regulondb --scope cache
uv run cruncher -c "$CONFIG" targets stats --campaign demo_pair
```

Example output (cache summary, abridged):

```bash
        Cache overview
      (source=regulondb)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric            ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ entries           │ 10      │
│ sources           │ 1       │
│ TFs               │ 10      │
│ motifs            │ 0       │
│ site sets         │ 10      │
│ sites (seq/total) │ 872/872 │
│ datasets          │ 0       │
└───────────────────┴─────────┘
```

Example output (demo_pair stats):

```bash
                                                               Campaign targets: demo_pair
┏━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Set ┃ TF   ┃ Source    ┃ Motif ID         ┃ Kind    ┃ Matrix len ┃ Sites (seq/total) ┃ Mean len ┃ Len min/max ┃ Len source ┃ Dataset ┃ Method ┃ Genome ┃
┡━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│   1 │ lexA │ regulondb │ RDBECOLITFC00214 │ curated │ -          │ 49/49             │ 19.5     │ 15/20       │ sequence   │ -       │ -      │ -      │
│   1 │ cpxR │ regulondb │ RDBECOLITFC00170 │ curated │ -          │ 154/154           │ 15.3     │ 11/19       │ sequence   │ -       │ -      │ -      │
└─────┴──────┴───────────┴──────────────────┴─────────┴────────────┴───────────────────┴──────────┴─────────────┴────────────┴─────────┴────────┴────────┘
```

For `demo_categories`, the stats table is much larger; use it when you are ready to inspect all expanded sets.

These stats are the first selection signal: **site counts** drive `selectors.min_site_count`, and length ranges
let you decide if `motif_store.site_window_lengths` needs tuning for any TFs with variable sites.

## Evaluate PWM quality (information content)

Compute PWM info-bits for a representative subset before you scale up. This helps decide whether to add a
`selectors.min_info_bits` gate for the campaign:

```bash
uv run cruncher -c "$CONFIG" catalog pwms --tf lexA --tf cpxR --source regulondb
```

Example output:

```bash
                                        PWM summary
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ PWM source ┃ Length ┃ Bits  ┃ n sites ┃ Site sets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩
│ lexA │ regulondb │ RDBECOLITFC00214 │ sites      │ 15     │ 10.36 │ 49      │ 1         │
│ cpxR │ regulondb │ RDBECOLITFC00170 │ sites      │ 11     │ 3.63  │ 154     │ 1         │
└──────┴───────────┴──────────────────┴────────────┴────────┴───────┴─────────┴───────────┘
```

If you want to enforce information content, add `selectors.min_info_bits` to the
`demo_categories_best` campaign in `config.yaml` and re-run `campaign validate`.

## Apply selectors and generate a derived config

Apply selectors to keep the strongest candidates and preview what got filtered
(include metrics so min-site-count / info-bit checks can run):

```bash
uv run cruncher -c "$CONFIG" campaign validate --campaign demo_categories_best --metrics --show-filtered
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
uv run cruncher -c "$CONFIG" campaign generate --campaign demo_categories_best --out "$DERIVED"
```

Example output:

```bash
/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/campaign.demo_categories_best.yaml
/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/campaign.demo_categories_best.campaign_manifest.json
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

```bash
uv run cruncher -c "$CONFIG" lock
uv run cruncher -c "$CONFIG" parse
uv run cruncher -c "$CONFIG" sample
uv run cruncher -c "$CONFIG" runs list --stage sample
```

Example output (runs list, abridged):

```bash
                                                                         Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                                             ┃ Stage  ┃ Status    ┃ Created                          ┃ Motifs ┃ Regulator set      ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb │ sample │ completed │ 2026-01-10T21:13:55.663756+00:00 │ 3      │ set1:lexA,cpxR,fur │ sites      │
│ sample_set1_lexA-cpxR-fur_20260110_161221_d63ebd │ sample │ completed │ 2026-01-10T21:12:22.000163+00:00 │ 3      │ set1:lexA,cpxR,fur │ sites      │
│ sample_set1_lexA-cpxR-fur_20260110_125635_0affef │ sample │ completed │ 2026-01-10T17:56:39.263742+00:00 │ 3      │ set1:lexA,cpxR,fur │ sites      │
└──────────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┴────────┴────────────────────┴────────────┘
```

## Inspect run artifacts and outputs

```bash
uv run cruncher -c "$CONFIG" runs show sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
```

Example output (abridged):

```bash
run: sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
stage: sample
status: completed
created_at: 2026-01-10T21:13:55.663756+00:00
motif_count: 3
regulator_set: {'index': 1, 'tfs': ['lexA', 'cpxR', 'fur']}
pwm_source: sites
run_dir: /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/runs/sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
artifacts:
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Stage  ┃ Type     ┃ Label                                  ┃ Path              ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ sample │ config   │ Resolved config (config_used.yaml)     │ config_used.yaml  │
│ sample │ trace    │ Trace (NetCDF)                         │ trace.nc          │
│ sample │ table    │ Sequences with per-TF scores (Parquet) │ sequences.parquet │
│ sample │ table    │ Elite sequences (Parquet)              │ elites.parquet    │
│ sample │ json     │ Elite sequences (JSON)                 │ elites.json       │
│ sample │ metadata │ Elite metadata (YAML)                  │ elites.yaml       │
└────────┴──────────┴────────────────────────────────────────┴───────────────────┘
```

## Render PWM logos (visual QA)

```bash
uv run cruncher -c "$CONFIG" catalog logos --set 1
```

Example output (abridged; third row omitted):

```bash
Rendered PWM logos
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ Length ┃ Bits  ┃ Output                                                       ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA │ regulondb │ RDBECOLITFC00214 │ 15     │ 10.36 │ /path/to/.../logos_set1_lexA-cpxR-fur_20260110_161412_a1b2c3 │
│ cpxR │ regulondb │ RDBECOLITFC00170 │ 11     │ 3.63  │ /path/to/.../logos_set1_lexA-cpxR-fur_20260110_161412_a1b2c3 │
└──────┴───────────┴──────────────────┴────────┴───────┴──────────────────────────────────────────────────────────────────────┘
Logos saved to /path/to/.../logos_set1_lexA-cpxR-fur_20260110_161412_a1b2c3
```

For live progress during sampling:

```bash
uv run cruncher -c "$CONFIG" runs watch <run_name>
```

To write a live metrics plot alongside the watch loop:

```bash
uv run cruncher -c "$CONFIG" runs watch <run_name> --plot
```

This writes `live_metrics.jsonl` and `live/live_metrics.png` under the run directory.

## Analyze + report (pairwise plots)

Pairwise plots require a TF pair. For the 3-TF demo run, pick a pair and pass `--tf-pair`:

```bash
uv run cruncher -c "$CONFIG" analyze --latest --tf-pair lexA,cpxR
uv run cruncher -c "$CONFIG" report sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
```

Example output (analyze):

```bash
Random baseline: 100%|██████████| 25/25 [00:00<00:00, 11676.79it/s]
Analysis outputs → /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/runs/sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb/analysis
  summary: /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/runs/sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb/analysis/summary.json
  analysis_id: 20260110T211410Z_1febb3
Next steps:
  cruncher runs show /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/config.yaml sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
  cruncher notebook --latest /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/runs/sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
  cruncher report /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/config.yaml sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
```

If you're running via `uv`, prefix those next-step commands with `uv run`.

## Open the run notebook (optional, real-time exploration)

```bash
uv run cruncher notebook --latest /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/runs/sample_set1_lexA-cpxR-fur_20260110_161355_df8fcb
```

## Optional: campaign-level summary

Aggregate many runs (pairs + facets across runs):

```bash
# Summarize only the analyzed runs (repeat --runs as needed)
uv run cruncher -c "$DERIVED" campaign summarize --campaign demo_categories_best --runs "$RUN"
```

Summary outputs include `campaign_summary.csv`, `campaign_best.csv`, and plots such as `best_jointscore_bar.png`, `tf_coverage_heatmap.png`, `joint_trend.png`, and `pareto_projection.png`.
Use `--skip-missing` if you summarize a larger run set where some analyses are incomplete.

To explore the campaign summary in a notebook (requires the summary artifacts above):

```bash
uv run cruncher -c "$DERIVED" campaign notebook --campaign demo_categories_best
```

## Notes

- Large campaigns can generate many regulator sets. If you need to reduce runtime, trim
  `regulator_sets` in the generated config to a smaller subset before sampling.
- If you want a default TF pair for repeated analyses, set `analysis.tf_pair`
  in the generated config; otherwise pass `--tf-pair` on each analyze call.
- `selectors.min_info_bits` requires PWMs to be buildable. For site-based PWMs
  with variable site lengths, set `motif_store.site_window_lengths` per TF (or
  switch to matrix-based sources) before enabling that selector.
- The demo config pre-populates `site_window_lengths` for the expanded TF list
  so multi-TF parse/sample runs work without extra edits.

## Where outputs live

- `<out_dir>/` - this demo writes to `runs/`.
- Campaign summaries land under `runs/campaigns/<campaign_id>/`.

## See also

- Command reference: [CLI reference](../reference/cli.md)
- Config knobs: [Config reference](../reference/config.md)
- Ingestion details: [Ingestion guide](../guides/ingestion.md)
