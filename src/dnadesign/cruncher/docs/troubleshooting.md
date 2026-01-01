# Troubleshooting

- Missing lockfile: run `cruncher lock <config>` before `cruncher parse/sample/analyze`.
- Target readiness: run `cruncher targets status <config>` to see missing matrices or sites.
- Target file missing: `targets status` reports `missing-matrix-file` / `missing-sites-file` if catalog points to missing cache files. Run `cruncher fetch ...` or `cruncher cache verify`.
- Lockfile/config mismatch: if `pwm_source` or TFs change, re-run `cruncher lock <config>`.
- Offline mode: `cruncher fetch ... --offline` only validates cache and fails if data is missing.
- Offline genome hydration: ensure `.cruncher/genomes/<accession>/` exists before running `fetch sites --offline`.
- Offline ambiguity: if multiple cached entries match a TF, use `--motif-id` or `--all` (motifs only).
- Missing motifs in catalog: run `cruncher fetch motifs --tf ... <config>`.
- Missing sites in catalog: run `cruncher fetch sites --tf ... <config>`.
- Sites have no sequences: HT peaks may be coordinate-only; Cruncher hydrates via NCBI by default. Re-run `cruncher fetch sites` (or `--hydrate`) after setting `ingest.genome_source=ncbi`.
- Site lengths vary: run `cruncher targets stats <config>` and set `motif_store.site_window_lengths` per TF or dataset.
- Genome FASTA mismatch: if hydration fails, ensure FASTA contig names match the reference accession or set `ingest.contig_aliases` (e.g., `chr -> U00096.3`).
- Missing assembly/contig accession: HT sites must provide `referenceGenome` / `assemblyGenomeId` for NCBI hydration.
- Ambiguous TF: use `cruncher catalog resolve <tf>` or `cruncher fetch sites --motif-id <id>` to disambiguate.
- Missing run manifest: rerun `cruncher parse` or `cruncher sample` (manifests are written per run).
- Missing report: run `cruncher report <config> <batch_name>` after sampling.
- Analyze/report fail on missing sequences: ensure `sample.save_sequences=true`.
- Missing trace.nc: ensure `sample.save_trace=true` and install `netCDF4` or `h5netcdf`.
- RegulonDB errors: check connectivity; GraphQL schema changes can break adapters.
- SSL verification failures: set `ingest.regulondb.ca_bundle` to a known CA bundle (loaded alongside certifi).
  Disable `verify_ssl` only as a last resort.
- As of January 1, 2026, RegulonDB has occasionally served an incomplete chain (missing intermediate).
  Supplying the intermediate via `ca_bundle` keeps verification secure and repeatable in CI.
- To find the issuer URL: `openssl s_client -connect regulondb.ccg.unam.mx:443 -showcerts </dev/null | openssl x509 -noout -text | rg "CA Issuers"`.
- Alignment matrix missing: set `ingest.regulondb.motif_matrix_source: sites` to build PWMs from curated binding sites.
- PWM from sites failure: increase TFBS count or set `motif_store.allow_low_sites: true` only if you accept weak PWMs.
