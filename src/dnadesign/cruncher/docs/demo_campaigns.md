## cruncher demo (category campaigns + multi-TF)

This walkthrough extends the demo workspace to run category-based campaigns and N>2 TF optimizations. Start with the two-TF demo first ([demo.md](demo.md)) to confirm your cache, lockfiles, and basic parse/sample/analyze flow. For live sampling and validation UX, see [demo_progressive.md](demo_progressive.md).

Captured outputs below were generated on **2026-01-09** using `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200` to avoid truncated tables (unless noted otherwise). Expect timestamps and counts to differ in your environment.

### Enter the demo workspace

The demo workspace lives here:

- `src/dnadesign/cruncher/workspaces/demo/`

You can either `cd` into the workspace (auto-detects `config.yaml`), or run from anywhere using the workspace-aware flags:

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo

# Option B: run from anywhere
cruncher --workspace demo sources list
cruncher --config src/dnadesign/cruncher/workspaces/demo/config.yaml sources list
```

### Campaigns configured in the demo workspace

The demo config includes a small pairwise campaign and a larger multi-category campaign:

- `demo_pair` (Stress + Envelope)
- `demo_categories` (Category1/2/3, no selectors)
- `demo_categories_best` (Category1/2/3 with quality selectors)

Category definitions (from the demo config):

- Category1: CpxR, BaeR
- Category2: LexA, RcdA, Lrp, Fur
- Category3: Fnr, Fur, AcrR, SoxR, SoxS, Lrp

List targets by category or campaign:

```bash
cruncher targets list --category Stress
cruncher targets list --category Category1
cruncher targets list --campaign demo_pair
cruncher targets list --campaign demo_categories
```

### Fetch sites for the campaigns

Fetch curated RegulonDB sites for all TFs implied by the campaign:

```bash
cruncher fetch sites --campaign demo_categories --no-selectors
```

Optional DAP-seq local source:

- If you have the O'Malley DAP-seq MEME files locally, add a `local_sources`
  entry (see `docs/config.md`) with `extract_sites: true`.
- Then fetch from that source instead of RegulonDB:

```bash
cruncher fetch sites --source omalley_ecoli_meme --campaign demo_categories --no-selectors
```

Summarize what is available:

```bash
cruncher sources summary --source regulondb --scope cache
cruncher targets stats --campaign demo_categories
```

### Apply selectors and generate a derived config

Apply selectors to keep the strongest candidates and generate a derived config:

```bash
cruncher campaign generate --campaign demo_categories_best --out config.demo_categories_best.yaml
```

The companion manifest (`config.demo_categories_best.campaign_manifest.json`) records per-TF metrics (site counts, plus info bits if you enable that selector).

### Run a multi-dimensional optimization and plot facets

```bash
cruncher lock config.demo_categories_best.yaml
cruncher parse config.demo_categories_best.yaml
cruncher sample config.demo_categories_best.yaml
cruncher analyze --latest --plots pairgrid config.demo_categories_best.yaml
```

### Optional: campaign-level summary

Aggregate many runs (pairs + facets across runs):

```bash
cruncher campaign summarize --campaign demo_categories_best --skip-missing
```

Summary outputs include `campaign_summary.csv`, `campaign_best.csv`, and plots such as `best_jointscore_bar.png`, `tf_coverage_heatmap.png`, `joint_trend.png`, and `pareto_projection.png`.

### Notes

- Large campaigns can generate many regulator sets. For a quick demo, trim
  `regulator_sets` in the generated config to a smaller subset.
- If you want pairwise plots for a specific TF pair, update `analysis.tf_pair`
  in the generated config before running `cruncher analyze`.
- `selectors.min_info_bits` requires PWMs to be buildable. For site-based PWMs
  with variable site lengths, set `motif_store.site_window_lengths` per TF (or
  switch to matrix-based sources) before enabling that selector.
- The demo config pre-populates `site_window_lengths` for the expanded TF list
  so multi-TF parse/sample runs work without extra edits.
