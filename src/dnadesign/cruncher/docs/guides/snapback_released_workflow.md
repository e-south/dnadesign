## Released-product snapback workflow

**Owner:** dnadesign-maintainers
**Doc kind:** guide
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-21
**Applies to:** `uv run cruncher snapback released-design|released-target-search|released-show`
**Last verified:** 2026-04-21
**Primary artifacts:** released-product reports, projection JSON, pre-event site records, released-design summary table

### Contents
- [Why this exists](#why-this-exists)
- [Current scope](#current-scope)
- [Core contract boundary](#core-contract-boundary)
- [Minimal explicit spec](#minimal-explicit-spec)
- [Standard command sequence](#standard-command-sequence)
- [Target-search behavior](#target-search-behavior)
- [Failure modes](#failure-modes)

### Why this exists

The released-product lane answers a different question from the preserved-site snapback lane:

> can a two-stage precursor carry one nickase site plus one downstream release-enzyme site such that the retained post-release product satisfies the final snapback geometry?

This lane exists because the final retained object may be smaller than the recognition sites required to build it. The explicit evaluation target is the rebased retained product, not the full precursor.

### Current scope

Current scope:

- `released-design` validates one explicit `single_nick_released_snapback_v1` precursor spec and writes one released-product bundle
- `released-target-search` searches nickase plus release-enzyme combinations for exact or near retained-product hits
- `released-show` revalidates one released-product bundle and fails fast on drift
- only `retained_side: upstream` is supported in v1
- only `stage_order: nick_then_release` is supported in v1
- the effective cap loop remains fixed at `3 nt`

Current non-scope:

- no released-product solve from arbitrary authored input
- no thermodynamic folding prediction
- no retron processivity or RT scoring
- no simultaneous digest chemistry modeling
- no multi-release or multi-nick workflows

Type IIS enzymes are modeled here as release enzymes, not nickases. The initial built-in release preset is `type_iis_release_v1`.

### Core contract boundary

The released-product lane is a sibling of the existing snapback explicit, solve, and preserved-site target-search lanes.

Key semantics:

- **precursor space** contains the nickase site, the downstream release site, and any sacrificial downstream sequence
- **retained-product space** is the upstream product after nick then release, rebased to coordinate `0`
- **pre-nick preservation** means the nickase recognition site must remain valid until the nick event
- **pre-release preservation** means the release recognition site must remain valid until the release event
- neither site is required to survive in the final retained product unless the spec forbids that loss

This lane stays geometry-first and construction-first. It is not a thermodynamic predictor and not a retron biology engine.

### Minimal explicit spec

```yaml
released_snapback:
  schema_version: 1
  kind: single_nick_released_snapback_v1
  name: demo_released_origin_033
input:
  precursor_top_strand: AACGTTGTTCCAA
nick_stage:
  nickase_variant_id: Nx.Exact7
  catalog:
    additional_paths: [inputs/nickases/local.nickases.yaml]
  normalized_to_top_strand_nick: true
  require_site_sequence_preserved_pre_nick: true
release_stage:
  release_variant_id: Re.Exact
  catalog:
    additional_paths: [inputs/release_enzymes/local.release.yaml]
  retained_side: upstream
  stage_order: nick_then_release
  require_site_sequence_preserved_pre_release: true
final_target:
  nick_boundary_from_left: 0
  paired_bp: 3
  cap_nt: 3
constraints:
  allow_post_release_loss_of_nickase_site: true
  allow_post_release_loss_of_release_site: true
  require_nick_survives_in_retained_product: true
  require_release_site_downstream_of_nick: true
  require_complete_downstream_fragment_separation: true
output:
  run_dir: outputs/released_design
```

See the checked-in demo at
[`../../workspaces/demo_snapback/configs/snapback/demo_released_origin_033.released.snapback.yaml`](../../workspaces/demo_snapback/configs/snapback/demo_released_origin_033.released.snapback.yaml).

### Standard command sequence

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/demo_snapback
cruncher() { uv run cruncher "$@"; }

# Materialize the checked-in released-product demo bundle.
cruncher snapback released-design \
  --spec configs/snapback/demo_released_origin_033.released.snapback.yaml \
  --force-overwrite

# Inspect the released-product bundle and integrity checks.
cruncher snapback released-show --run outputs/released_design

# Search paired nickase plus release-enzyme combinations against the same toy overlays.
cruncher snapback released-target-search \
  --workspace-root . \
  --nick-additional-path inputs/nickases/local.nickases.yaml \
  --release-additional-path inputs/release_enzymes/local.release.yaml \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --json
```

### Target-search behavior

`released-target-search` is target-first and does not write a bundle.

The search loop is:

1. enumerate feasible nickase placements
2. enumerate feasible downstream release placements
3. project the retained product
4. evaluate the final geometry in retained-product space
5. rank exact hits and near hits separately

The report surfaces:

- exact hits
- near hits
- blocker counts by reason
- pre-truncation and post-truncation hit counts

When no sources are provided, the command defaults to `neb_nicking_v1` plus `thermo_nicking_v1` for nickases and `type_iis_release_v1` for release enzymes.

### Failure modes

Common fail-fast cases:

- the spec path is outside `configs/snapback/`
- `stage_order` is not `nick_then_release`
- `retained_side` is not `upstream`
- the effective cap loop is not exactly `3 nt`
- no pre-nick site matches the requested normalized boundary
- no release site produces a fully separated downstream sacrificial fragment
- the retained product would lose the required nick
- the retained product rebases cleanly but still fails the reused explicit snapback geometry checks

For the file-by-file bundle layout, use [`../reference/released_snapback_artifacts.md`](../reference/released_snapback_artifacts.md).
For release-enzyme catalog fields and the built-in Type IIS preset, use [`../reference/release_enzyme_catalogs.md`](../reference/release_enzyme_catalogs.md).
