## Cruncher to DenseGen (PWM handoff)

This runbook shows how to produce DenseGen-ready PWM artifacts with Cruncher.

Goal:
- fetch TF site sources
- run motif discovery
- export one JSON artifact per motif into a DenseGen workspace

DenseGen then consumes those JSON files through `type: pwm_artifact_set`.

---

### Scope of this example

This example prepares three TFs: `lexA`, `cpxR`, `baeR`.

Assumptions:

- `lexA` and `cpxR` local DAP-seq MEME files are available under
  `src/dnadesign/cruncher/workspaces/densegen_prep_three_tf/inputs/local_motifs/`
- `baeR` sites come from processed ChIP-exo FASTA
- RegulonDB curated sites are used as supplemental input

Cruncher config path used below:
`src/dnadesign/cruncher/workspaces/densegen_prep_three_tf/config.yaml`

---

### Example Cruncher config (three-TF prep)

```yaml
cruncher:
  out_dir: outputs
  regulator_sets:
    - [lexA, cpxR, baeR]

  motif_store:
    source_preference: [meme_suite_meme, meme_suite_streme, demo_local_meme, regulondb]
    combine_sites: true
    site_window_lengths:
      lexA: 20
      cpxR: 20
      baeR: 20

  motif_discovery:
    tool: meme
    meme_mod: oops
    meme_prior: addone
    source_id: meme_suite_meme

  ingest:
    regulondb:
      curated_sites: true
      ht_sites: false
    local_sources:
      - source_id: demo_local_meme
        root: inputs/local_motifs
        patterns: ["*.txt"]
        format_map: {".txt": "MEME"}
        extract_sites: true
        tf_name_strategy: stem
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
        citation: "Choudhary et al. 2020 (DOI: 10.1128/mSystems.00980-20"
        source_url: https://doi.org/10.1128/mSystems.00980-20
        tags:
          assay: chip_exo
          doi: 10.1128/mSystems.00980-20

  parse:
    plot:
      logo: true
      bits_mode: information
      dpi: 150
```

---

### 1) Fetch sources and run motif discovery

```bash
# Move into the dedicated Cruncher workspace.
cd src/dnadesign/cruncher/workspaces/densegen_prep_three_tf

# Save config path for reuse.
CONFIG="$PWD/config.yaml"

# Pull local DAP-seq motifs/sites for lexA and cpxR.
pixi run cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher fetch sites  --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"

# Pull BaeR ChIP-exo sites.
pixi run cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"

# Pull supplemental curated RegulonDB sites for baeR.
pixi run cruncher fetch sites --tf baeR --update -c "$CONFIG"

# Optional: include curated sites for lexA/cpxR too.
# pixi run cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"

# Verify external tooling (MEME Suite) is healthy.
pixi run cruncher doctor -c "$CONFIG"

# Run MEME discovery for all three TFs.
pixi run cruncher discover motifs --tf lexA --tf cpxR --tf baeR --tool meme --meme-mod oops --meme-prior addone --source-id meme_suite_meme -c "$CONFIG"

# Render motif logos for quick QA.
pixi run cruncher catalog logos --source meme_suite_meme --set 1 -c "$CONFIG"

# Lock exact motif IDs/hashes for reproducibility.
pixi run cruncher lock -c "$CONFIG"
```

If any TF has zero sites, discovery and lock steps will fail.
Resolve missing sources before continuing.

---

### 2) Export artifacts into DenseGen workspace

```bash
# Export DenseGen PWM artifact JSON files from catalog set 1.
pixi run cruncher catalog export-densegen --set 1 --densegen-workspace demo_sampling_baseline -c "$CONFIG"

# Export site tables (optional but useful for cross-checking).
pixi run cruncher catalog export-sites --set 1 --densegen-workspace demo_sampling_baseline --overwrite -c "$CONFIG"
```

These commands write motif JSON files under:
`src/dnadesign/densegen/workspaces/demo_sampling_baseline/inputs/motif_artifacts/`

If you regenerate motifs, make sure DenseGen uses the newly exported IDs.
`catalog export-densegen` cleans existing selected-TF artifact JSON files by default;
use `--no-clean` if you need to keep previous files.

---

### 3) DenseGen side: how artifacts are used

DenseGen input types used in this pattern:

- `type: pwm_artifact_set` (required)
- `type: binding_sites` (optional)

See:
- [../reference/config.md](../reference/config.md)
- [../reference/motif_artifacts.md](../reference/motif_artifacts.md)

### Demo plan intent (`demo_sampling_baseline`)

The packaged DenseGen demo includes a `background_pool` input for neutral 16-20 bp parts.
Plan-scoped pooling then builds four libraries:

- `controls` (background only; sigma70 spacer 16-18 bp)
- `ethanol` (CpxR/BaeR + background; sigma70 spacer 16-20 bp)
- `ciprofloxacin` (LexA + background; sigma70 spacer 16-18 bp)
- `ethanol_ciprofloxacin` (LexA + CpxR/BaeR + background; sigma70 spacer 16-20 bp)

This structure yields both monotypic and heterotypic designs while reducing accidental motif carryover.

---

@e-south
