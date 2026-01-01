# Demo: RegulonDB LexA + CpxR (end-to-end)

This walk-through shows how to discover, fetch, lock, parse, sample, and inspect
Cruncher outputs using **LexA** and **CpxR** from RegulonDB.

## 0) Pick a demo config

Use the example config:

- `docs/examples/regulondb_ecoli.yaml`

It’s tuned for fast local runs. Increase `draws/tune` for real experiments.

## 1) Preview available motifs (remote)

```bash
cruncher fetch motifs --tf lexA --tf cpxR --dry-run docs/examples/regulondb_ecoli.yaml
```

If you see SSL errors, set `ingest.regulondb.ca_bundle` to a known CA bundle
(loaded alongside certifi). Only disable `verify_ssl` as a last resort.

## 2) Fetch curated binding sites (PWM built from sites)

```bash
cruncher fetch sites  --tf lexA --tf cpxR docs/examples/regulondb_ecoli.yaml
```

Each command prints a compact summary table of what was cached.

## 3) Inspect what’s cached

```bash
cruncher catalog list docs/examples/regulondb_ecoli.yaml
cruncher targets status docs/examples/regulondb_ecoli.yaml
cruncher targets stats docs/examples/regulondb_ecoli.yaml
cruncher targets candidates --fuzzy docs/examples/regulondb_ecoli.yaml
cruncher catalog show docs/examples/regulondb_ecoli.yaml regulondb:RDBECOLITFC00214
```

## 4) Lock targets (required)

```bash
cruncher lock docs/examples/regulondb_ecoli.yaml
```

Lockfiles live at `.cruncher/locks/<config>.lock.json` and pin exact motif IDs.

## 5) Parse (logos)

```bash
cruncher parse docs/examples/regulondb_ecoli.yaml
```

Logos are written under `results/parse_set<index>_<tfset>_<timestamp>/`.

## 6) Sample (MCMC optimization)

```bash
cruncher sample docs/examples/regulondb_ecoli.yaml
cruncher runs list docs/examples/regulondb_ecoli.yaml
```

For live progress:

```bash
cruncher runs watch docs/examples/regulondb_ecoli.yaml <run_name>
```

Sample run directories include the regulator set index (e.g., `sample_set1_lexA-cpxR_20250101_120000`).

## 7) Analyze + report

```bash
cruncher analyze docs/examples/regulondb_ecoli.yaml
cruncher report docs/examples/regulondb_ecoli.yaml <sample_run_name>
```

## 8) Optional: use alignment matrices (if available)

This demo uses `pwm_source: sites` because LexA/CpxR alignments are not always present.
If you have alignment matrices, set `motif_store.pwm_source: matrix` and run:

```bash
cruncher fetch motifs --tf lexA --tf cpxR docs/examples/regulondb_ecoli.yaml
```

## 9) Optional: HT binding sites (RegulonDB)

Enable HT retrieval in the config:

```yaml
ingest:
  regulondb:
    ht_sites: true
    ht_binding_mode: tfbinding   # tfbinding | peaks
```

Then validate:

```bash
cruncher fetch sites --dry-run --tf lexA --tf cpxR docs/examples/regulondb_ecoli.yaml
cruncher fetch sites --tf lexA --tf cpxR --limit 5 docs/examples/regulondb_ecoli.yaml
```

If HT datasets return peaks without sequences, hydrate them using a reference genome:

```bash
cruncher fetch sites --tf lexA --tf cpxR docs/examples/regulondb_ecoli.yaml
```

Cruncher hydrates coordinate-only peaks via NCBI by default (`ingest.genome_source=ncbi`).
To run offline, provide a local FASTA and set `genome_source=fasta`.
Cruncher fails fast if no TFBinding **or** peaks are returned, so validate early.
If HT site lengths vary, use `cruncher targets stats` and set
`motif_store.site_window_lengths` per TF or dataset before building PWMs.
