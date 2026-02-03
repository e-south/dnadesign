## Cruncher to DenseGen (PWM handoff)

Below is a walkthrough that uses **Cruncher** commands to fetch binding sites for select transcription factors. **Cruncher** then exports per-motif JSON artifacts which DenseGen consumes for Stage-A PWM sampling.

---

### Three-TF prep (lexA, cpxR, baeR)

This workflow fetches TFBS, runs MEME discovery, and exports DenseGen-ready artifacts.
It also documents provenance for the demo sources beyond RegulonDB.

Assumptions for this example:

- **lexA** and **cpxR** have local DAP-seq MEME files (demo_local_meme) under
  `src/dnadesign/cruncher/workspaces/densegen_prep_three_tf/inputs/local_motifs/`.
- **baeR** comes from a processed ChIP-exo FASTA (Choudhary et al. 2020, DOI:
  10.1128/mSystems.00980-20) stored in the sibling repo `dnadesign-data` under
  `primary_literature/Choudhary_et_al/processed/BaeR_binding_sites.fasta`.
- RegulonDB curated sites are used as a supplement (baeR by default; lexA/cpxR
  optionally) so discovery can merge across sources.

The dedicated config lives at `src/dnadesign/cruncher/workspaces/densegen_prep_three_tf/config.yaml`.

```yaml
cruncher:
  out_dir: outputs  # workspace-local outputs
  regulator_sets:  # three TFs for discovery + export
    - [lexA, cpxR, baeR]

  motif_store:
    source_preference: [meme_suite_meme, meme_suite_streme, demo_local_meme, regulondb]
    combine_sites: true                # merge per-TF sites across sources for discovery
    site_window_lengths:               # fixed windows if site-derived PWMs are used
      lexA: 20  # bp window
      cpxR: 20  # bp window
      baeR: 20  # bp window

  motif_discovery:
    tool: meme                         # prefer MEME explicitly
    meme_mod: oops                     # each site is one motif
    meme_prior: addone                 # stabilize sparse site sets
    source_id: meme_suite_meme         # must match source_preference

  ingest:
    regulondb:
      curated_sites: true              # curated RegulonDB sites
      ht_sites: false                  # keep HT off in this walkthrough
    local_sources:
      - source_id: demo_local_meme     # local DAP-seq MEME files
        root: inputs/local_motifs      # demo motifs directory
        patterns: ["*.txt"]            # MEME text files
        format_map: {".txt": "MEME"}   # explicit parser mapping
        extract_sites: true            # include MEME BLOCKS sites
        tf_name_strategy: stem         # TF names from filenames
    site_sources:
      - source_id: baer_chip_exo
        description: Choudhary et al. BaeR ChIP-exo binding sites (processed FASTA)
        path: ../../../../../../dnadesign-data/primary_literature/Choudhary_et_al/processed/BaeR_binding_sites.fasta
        tf_name: BaeR
        record_kind: chip_exo
        organism:
          name: Escherichia coli
          strain: K-12 MG1655
          assembly: NC_000913.3
        citation: "Choudhary et al. 2020 (DOI: 10.1128/mSystems.00980-20)"
        source_url: https://doi.org/10.1128/mSystems.00980-20
        tags:
          assay: chip_exo
          doi: 10.1128/mSystems.00980-20

  parse:
    plot:
      logo: true                       # enable PWM logos for parse
      bits_mode: information           # logo scale
      dpi: 150                         # plot resolution
```

#### Fetch sources + run MEME

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

# BaeR ChIP-exo sites (Choudhary et al. FASTA).
cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"

# RegulonDB curated sites (supplemental; merges with local sources).
cruncher fetch sites --tf baeR --update -c "$CONFIG"

# Optional: if you want lexA/cpxR discovery to include curated sites too.
# cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"

# Verify MEME Suite before discovery.
cruncher doctor -c "$CONFIG"

# MEME discovery (preferred) so all three TFs have consistent PWMs.
cruncher discover motifs --tf lexA --tf cpxR --tf baeR --tool meme --meme-mod oops --meme-prior addone --source-id meme_suite_meme -c "$CONFIG"

# Render PWM logos for grounding/QA.
cruncher catalog logos --source meme_suite_meme --set 1 -c "$CONFIG"

# Pin exact motif IDs/hashes for reproducibility.
cruncher lock -c "$CONFIG"
```

If any TF has zero sites, `discover motifs` and `lock` will fail.
Stop and resolve the missing source before proceeding (for example, verify the
ChIP-exo FASTA path for baeR or adjust your RegulonDB query).

Logos are saved under `outputs/logos/catalog/` (the command prints the exact path).

#### Export into a DenseGen workspace

```bash
cruncher catalog export-densegen --set 1 --densegen-workspace demo_meme_three_tfs --overwrite -c "$CONFIG"
cruncher catalog export-sites   --set 1 --densegen-workspace demo_meme_three_tfs --overwrite -c "$CONFIG"
```

These commands write motif JSONs for **lexA**, **cpxR**, and **baeR** under
`src/dnadesign/densegen/workspaces/demo_meme_three_tfs/inputs/motif_artifacts/`,
which the DenseGen demo config references directly.

If you regenerate motifs, make sure the DenseGen config points at the newly
exported motif IDs (or update from `artifact_manifest.json`).

---

### DenseGen inputs

- `type: pwm_artifact_set` for PWM artifacts
- `type: binding_sites` for exported site tables (optional)

See `reference/config.md` for exact fields.

---

@e-south
