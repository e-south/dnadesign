## Cruncher to DenseGen (PWM handoff)

Below is a walkthrough that uses **Cruncher** commands to fetch binding sites for select transcription factors. **Cruncher** then exports per-motif JSON artifacts which DenseGen consumes for Stage-A PWM sampling.

---

### Three-TF prep (lexA, cpxR, baeR)

This workflow fetches TFBS, runs STREME discovery, and exports DenseGen-ready artifacts.

Assumptions for this example:

- **lexA** and **cpxR** have local DAP-seq MEME files (demo_local_meme).
- **baeR** is sourced from RegulonDB curated sites.
- lexA/cpxR may also have curated RegulonDB sites; you can include them if you
  want discovery to merge across sources.

The dedicated config lives at `src/dnadesign/cruncher/workspaces/densegen_prep_three_tf/config.yaml`.

```yaml
cruncher:
  out_dir: outputs  # workspace-local outputs
  regulator_sets:  # three TFs for discovery + export
    - [lexA, cpxR, baeR]

  motif_store:
    source_preference: [meme_suite_streme, demo_local_meme, regulondb]
    combine_sites: true                # merge per-TF sites across sources for discovery
    site_window_lengths:               # fixed windows if site-derived PWMs are used
      lexA: 20  # bp window
      cpxR: 20  # bp window
      baeR: 20  # bp window

  motif_discovery:
    tool: streme                       # prefer STREME explicitly
    source_id: meme_suite_streme       # must match source_preference

  ingest:
    regulondb:
      curated_sites: true              # baeR curated sites
      ht_sites: false                  # keep HT off in this walkthrough
    local_sources:
      - source_id: demo_local_meme     # local DAP-seq MEME files
        root: inputs/local_motifs      # demo motifs directory
        patterns: ["*.txt"]            # MEME text files
        format_map: {".txt": "MEME"}   # explicit parser mapping
        extract_sites: true            # include MEME BLOCKS sites
        tf_name_strategy: stem         # TF names from filenames

  parse:
    plot:
      logo: true                       # enable PWM logos for parse
      bits_mode: information           # logo scale
      dpi: 150                         # plot resolution
```

#### Fetch sources + run STREME (preferred)

```bash
# Option A: cd into the dedicated cruncher workspace
cd src/dnadesign/cruncher/workspaces/densegen_prep_three_tf
CONFIG="$PWD/config.yaml"

# Option B: run from anywhere in the repo
# CONFIG=src/dnadesign/cruncher/workspaces/densegen_prep_three_tf/config.yaml

# Use pixi as the default runner (avoid alias collisions).
unalias cruncher 2>/dev/null
cruncher() { pixi run cruncher -- "$@"; }

# Local DAP-seq motifs + sites (lexA/cpxR only).
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites  --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"

# RegulonDB curated sites (baeR).
cruncher fetch sites --tf baeR --update -c "$CONFIG"

# Optional: if you want lexA/cpxR discovery to include curated sites too.
# cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"

# Verify MEME Suite before discovery.
cruncher doctor -c "$CONFIG"

# STREME discovery (preferred) so all three TFs have consistent PWMs.
cruncher discover motifs --tf lexA --tf cpxR --tf baeR --tool streme --source-id meme_suite_streme -c "$CONFIG"

# Pin exact motif IDs/hashes for reproducibility.
cruncher lock -c "$CONFIG"
```

If any TF has zero sites, `discover motifs` and `lock` will fail.
Stop and resolve the missing source before proceeding (for example, add a
local site set for baeR or adjust your RegulonDB query).

#### Export into a DenseGen workspace

```bash
cruncher catalog export-densegen --set 1 --densegen-workspace demo_meme_two_tf -c "$CONFIG"
cruncher catalog export-sites   --set 1 --densegen-workspace demo_meme_two_tf -c "$CONFIG"
```

These commands write motif JSONs for **lexA**, **cpxR**, and **baeR** under
`src/dnadesign/densegen/workspaces/demo_meme_two_tf/inputs/motif_artifacts/`,
which the DenseGen demo config references directly.

---

### DenseGen inputs

- `type: pwm_artifact_set` for PWM artifacts
- `type: binding_sites` for exported site tables (optional)

See `reference/config.md` for exact fields.

---

@e-south
