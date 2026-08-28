## Storage adoption audit and RegulonDB shared-root tracer

**Date:** 2026-08-28
**Status:** shared-root topology viable; deep interoperability blocked
**Scope:** copy-first audit and tracer only; no caller cutover or source retirement

### Decision

Keep the existing dataset-specific USR storage objects as cold recovery
snapshots. The compatibility-preserving external topology is one storage object
containing a complete shared USR root:

```text
storage/stores/usr/<shared-root>/
  storage.object.json
  registry.yaml
  <dataset-a>/
  <dataset-b>/
```

The RegulonDB tracer proves that this topology preserves current USR and
Construct behavior and supports LatentDNA shallow inspection. It does not yet
prove the full Construct to Infer to LatentDNA route because the retained native
and core60 datasets lack canonical Infer sidecars. Keep the tracer cold and
non-authoritative until those sidecars are recovered or regenerated and deep
validation passes without reading the embedded root.

Do not introduce a multi-root USR catalog on current evidence. The shared-root
contract works at the owner-tool layer and avoids a cross-tool API migration.

### Baseline receipt

- Live source revision: `4ce87f8`.
- Adoption worktree revision and `origin/main`: `d3de39a`.
- Two quiescence passes agreed on 34,226 ignored regular files and
  58,669,760,759 bytes.
- The per-candidate closure contains 95 non-overlapping buckets and covers the
  same 34,226 files and 58,669,760,759 bytes. Its framed aggregate digest is
  `sha256:996240a791dd58664d73a06526ea25dc5f162e80822cd15de77a1637e3af0cb0`.
- No symlinks, multiply linked regular files, or dataless cloud placeholders
  were observed under `src/dnadesign`.
- The two RegulonDB datasets had no open file handles during the source check.
- Nested Git repositories remain present at `archived/decoydesigner` and
  `prototypes/deepDNAshape`; verified source-recovery objects already protect
  their unpublished state.

The machine-readable candidate ledger is
[2026-08-28-storage-adoption-candidates.json](2026-08-28-storage-adoption-candidates.json).
Each bucket records a file count, exact byte total, and a path-plus-content
digest. The largest candidates are:

| Candidate | Files | Bytes | Disposition |
| --- | ---: | ---: | --- |
| `latentdna/workspaces/stress_ethanol_cipro_growth` | 3,581 | 21,893,767,934 | Reconcile study config before adoption |
| `studies/units/eco1_rt_repack` | 16,708 | 11,338,517,657 | Existing foldcheck copy stays cold and unknown-legacy |
| `usr/datasets/usr_prom_eth_cip_opal_candidates` | 15 | 5,306,932,643 | Blocked by missing historical registry receipt |
| `usr/datasets/construct_prom_eth_cip_context` | 32 | 4,805,777,345 | Migrate only as part of the stress shared root |
| `latentdna/workspaces/rt_lnrna_sponging_construct_triage` | 820 | 3,166,859,303 | Reconcile study config before adoption |
| `archived/seqcnn` | 399 | 2,590,580,883 | Mixed archive; split ownership before adoption |
| `archived/preprocessing` | 299 | 1,292,212,438 | Mixed archive; split ownership before adoption |
| `usr/datasets/usr_prom_eth_cip_anchor` | 35 | 1,252,190,004 | Migrate only as part of the stress shared root |
| `usr/datasets/archived` | 190 | 999,702,626 | Legacy dataset family; classify individually |
| largest Cruncher workspace | 237 | 922,391,554 | Selective cold-workspace candidate |
| `archived/densepromoters` | 365 | 674,991,450 | Mixed archive; split ownership before adoption |
| DenseGen stress workspace | 1,841 | 241,964,838 | Selective adoption after USR tracer |

The ledger keeps caches, Finder metadata, small ignored source, batch outputs,
and residual tool material in explicit accounting buckets rather than silently
dropping them.

### Consumer graph

The machine-readable caller ledger is
[2026-08-28-storage-adoption-consumers.json](2026-08-28-storage-adoption-consumers.json).
It scanned tracked and ignored local text/config surfaces while excluding Git
metadata, environments, bytecode caches, node modules, and nested worktrees.

| Repository | Unique matching files | Role in cutover |
| --- | ---: | --- |
| live dnadesign tree | 1,796 | Tool defaults, runbooks, ignored workspaces, legacy archives |
| Research Studies | 565 | Dataset contracts, study configs, review and readiness surfaces |
| Reader | 7 | Legacy USR-path references; include in negative-path checks |
| dnadesign-data | 0 | No matching caller surface found |

The important runtime edges are:

```text
Research Studies config
  -> one shared USR root
  -> native + core60 dataset directories
  -> Construct validation/dry run
  -> Infer sequence-view aliases and vector/scalar payloads
  -> LatentDNA source adapters and deep validation
  -> study-owned review and readiness surfaces
```

Stress-study USR, OPAL, DenseGen, and LatentDNA callers also depend on one
shared root. They remain out of this tracer and must not be cut over until the
RegulonDB deep gate passes.

### RegulonDB tracer receipt

`storage/stores/usr/regulondb-shared-root-tracer` contains the current shared
registry plus complete copies of:

- `usr_regulondb_native_promoters`;
- `usr_regulondb_native_promoter_core60`.

Before storage metadata was added, the 37 copied source files matched the source
subset at
`sha256:f683bcd38b2d85716e3c6a8c536b121105c50e35be0f2eb99bf5bc3aa5a4a674`.
The published cold object has 38 declared resources, 7,955,814 bytes, and
manifest digest
`sha256:f303347862830b1bb2ff68a3a60b74fbf8210776fb100b2b9fbb52df7c864dd6`.

The existing LatentDNA cold object now carries a durable shared-root consumer
test under `provenance/`. Its refreshed receipt declares 455 resources,
137,680,357 bytes, and manifest digest
`sha256:336453870f541857e7d9528c51091b168af8236457074e11aa373790720c435e`.

### Acceptance results

| Gate | Result | Evidence |
| --- | --- | --- |
| Byte preservation | Pass | Exact 37-file source/destination subset digest; originals retained |
| Storage integrity | Pass | Tracer inventory plus explicit validation; LatentDNA refresh plus explicit validation; storage root verifies 22 objects |
| USR owner validity | Pass | Both datasets list and strictly validate under current-or-frozen registry policy |
| Construct interoperation | Pass | Workspace doctor, project validation, and dry run against the external root |
| Infer config validity | Pass | Native and core60 pinned configs validate without loading Evo2 |
| Infer completeness | Blocked | Native: 6,364 vectors and 6,364 scalars missing; core60: 6,362 and 6,362 missing; zero reusable |
| LatentDNA shallow inspection | Pass | 3,182 native and 3,181 core60 rows resolved only from the external root |
| LatentDNA deep validation | Blocked | All four representation views are missing because canonical Infer sidecars are absent |
| Study repository | Pass with router bug | `study-workspace validate`: 5 studies, 24 work items, 30 artifacts; no live RegulonDB status provider is registered |
| Caller cutover | Not attempted | Old source remains available; no config or operator binding changed |
| Independent recovery | Not met | Copies remain inside the same Dropbox failure domain |
| Retirement | Not authorized | No embedded source removed |

### Sidecar recovery result

No canonical RegulonDB `feature_aliases.parquet`, `feature_vectors.parquet`,
`feature_scalar_aliases.parquet`, or `feature_scalars.parquet` was found in local
stores, materializations, recovery archives, or a narrow Trash search. Retained
LatentDNA matrices and row tables preserve some vector materializations, and
historical manifests preserve expected vector-sidecar digests, but the full
alias schema and log-likelihood scalar payloads are not locally reconstructable.

A non-interactive SCC lookup failed at authentication before reading remote
bytes. Remote sidecars are therefore uninspected, not absent. If an interactive
SCC session cannot recover identity-matching sidecars, regenerate them with the
official pinned Infer recipes; never synthesize scalar payloads from the
LatentDNA materializations.

### Bugs and operational findings

1. Dropbox changes only a newly atomically published manifest's `ctime` shortly
   after publication. The storage code now performs one bounded 0.5-second
   retry only for the exact whole-object quiescence error. Digest mismatches,
   undeclared files, changed inputs, and every other semantic error still fail
   immediately and roll back.
2. The documented `ops runbook fill-infer` route resolves the Research
   Studies-owned runbook relative to the dnadesign checkout and reports it
   missing. Direct owner-tool validation works; the router needs a separate
   path-resolution repair before it can serve as acceptance evidence.
3. Finder recreated `storage/stores/.DS_Store`. It was moved recoverably to
   Trash, after which the storage root verified 22 objects: 12 stores,
   9 workspaces, and 1 tool cache.

### Protected exclusions and next gate

Dense Arrays, LigandMPNN/HOP, the DenseGen Dense Arrays showcase, and all
stress/OPAL migration surfaces were untouched. No archive-wide move is allowed.

The next safe increment is:

1. recover the exact RegulonDB sidecars from SCC or another retained receipt;
2. otherwise regenerate them with pinned native and core60 Infer recipes;
3. rerun exact Infer completion and LatentDNA deep validation against only the
   external shared root;
4. repair and test the cross-repository `ops` runbook resolver;
5. only after those pass, trace the complete stress-study shared USR family.

Retirement remains a separate authorization and additionally requires a
checksum-verified recovery copy outside Dropbox plus a successful restore drill.
