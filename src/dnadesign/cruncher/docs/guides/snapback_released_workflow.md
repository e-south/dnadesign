## Released-product snapback workflow

**Owner:** dnadesign-maintainers
**Doc kind:** guide
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-22
**Applies to:** `uv run cruncher snapback released-design|released-target-search|released-solve|released-show`
**Last verified:** 2026-04-22
**Primary artifacts:** released-product search/solve reports, projection JSON, pre-event site records, released-design summary table, and per-hit origin-anchored exposed-bottom plots

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

> can a two-stage precursor carry one nickase site plus one downstream release-enzyme site such that the exposed post-release bottom strand satisfies the final snapback geometry?

This lane exists because the final exposed bottom strand may be smaller than the recognition sites required to build it. The explicit evaluation target is the rebased exposed bottom strand, not the full precursor.

### Current scope

Current scope:

- `released-design` validates one explicit `single_nick_released_snapback_v1` precursor spec and writes one released-product bundle
- `released-target-search` searches nickase plus release-enzyme combinations for exact or near exposed-bottom hits
- `released-solve` materializes ranked exact or near exposed-bottom hits and can emit one origin-anchored plot per hit
- `released-show` revalidates one released-product bundle and fails fast on drift
- only `retained_side: upstream` is supported in v1
- only `stage_order: nick_then_release` is supported in v1
- the effective cap loop remains fixed at `3 nt`
- default released-product operational policy excludes nickases carrying `FREQUENT_CUTTER` unless the operator explicitly opts in

Current non-scope:

- no thermodynamic folding prediction
- no retron processivity or RT scoring
- no simultaneous digest chemistry modeling
- no multi-release or multi-nick workflows

Type IIS enzymes are modeled here as release enzymes, not nickases. The initial built-in release preset is `type_iis_release_v1`.

### Core contract boundary

The released-product lane is a sibling of the existing snapback explicit, solve, and preserved-site target-search lanes.

Key semantics:

- **precursor space** contains the nickase site, the downstream release site, and any sacrificial downstream sequence
- **final-geometry space** is the exposed post-release bottom strand after nick then release, rebased so the nick boundary remains the origin coordinate
- the exposed bottom strand is stored in left-to-right precursor coordinate order, so the rendered bottom row reads `3' -> 5'`
- **pre-nick preservation** means the nickase recognition site must remain valid until the nick event
- **pre-release preservation** means the release recognition site must remain valid until the release event
- neither site is required to survive in the post-release product unless the spec forbids that loss

This lane stays geometry-first and construction-first. It is not a thermodynamic predictor and not a retron biology engine.

### Minimal explicit spec

```yaml
released_snapback:
  schema_version: 1
  kind: single_nick_released_snapback_v1
  name: local_exact033
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
  require_nick_survives_in_retained_product: false
  require_release_site_downstream_of_nick: true
  require_complete_downstream_fragment_separation: true
output:
  run_dir: outputs/released_design
```

For the checked-in operational workspace, use
[`../../workspaces/de033/runbook.md`](../../workspaces/de033/runbook.md).
`de033` now carries both the whole-catalog search/solve surface and one pinned
explicit downstream-`BspQI` bundle at
[`../../workspaces/de033/configs/snapback/de033.released.snapback.yaml`](../../workspaces/de033/configs/snapback/de033.released.snapback.yaml),
which currently resolves to `Nt.BsmAI + BspQI`.

The search and solve surfaces also collapse redundant exact and near hits to
one representative per exposed post-nick `stem + cap` geometry. That keeps the
operator-facing hit list focused on distinct foldback outcomes rather than many
enzyme-pair aliases for the same `0/3/3` object.

The released-product lane also rejects internal-cut nickases whose matched
recognition span crosses the active-strand origin. A hit must place the
pre-nick recognition site fully upstream of the exposed foldback strand rather
than merely rebasing the nick event to boundary `0`.

### Standard command sequence

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/de033
cruncher() { uv run cruncher "$@"; }

# Probe the real dual-enzyme 0/3/3 space before writing a bundle.
cruncher snapback released-target-search \
  --workspace-root . \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --json

# Materialize whole-catalog hit bundles and emit one origin-anchored plot per hit.
cruncher snapback released-solve \
  --workspace-root . \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --run-dir outputs/released_solve \
  --materialize-top-k 8 \
  --render-format pdf \
  --emit-renders \
  --force-overwrite \
  --json

# Materialize the checked-in released-product bundle.
```

### Target-search behavior

`released-target-search` is target-first and does not write a bundle.

The search loop is:

1. enumerate feasible nickase placements
2. enumerate feasible downstream release placements
3. project the exposed bottom strand plus only the top-prefix fragment, if any,
   that remains left of the nick
4. evaluate the final geometry on the exposed bottom strand
5. rank exact hits and near hits separately

The report surfaces:

- exact hits
- near hits
- blocker counts by reason
- pre-truncation and post-truncation hit counts

Released-product search no longer permits any release-site geometry to begin
left of logical origin `0`. Nickase geometry is stricter than a generic
rebase but slightly looser than an absolute nonnegative clamp: after
top-strand normalization, the engine may omit a left-of-origin nickase prefix
only when that omitted block is a single contiguous fully degenerate `N`
segment. If any protected base would fall left of origin, the candidate is
rejected rather than rebased.

`released-solve` reuses the same whole-catalog search, then materializes the
top ranked exact hits when any exist or the top ranked near hits otherwise. The
solve bundle writes one released-product hit bundle per hit under
`outputs/released_solve/analysis/materialized_hits/hit_<rank>/`, snapshots the
target-search hit plus projection/site payloads, and can render
`plots/released_hit_triptych.<fmt>` inside each hit bundle when
`--emit-renders` is enabled. The solve plot keeps `Nick / origin` at the left
boundary, shows the exposed bottom strand as the scored object, and renders the
foldback panel as the same active bottom strand returning on itself rather than
as a reverse-complement annotation. All emitted plot coordinates are
non-negative; omitted left-of-origin nickase prefix positions are never drawn,
and no surviving site or fragment is allowed to extend left of origin.

Exact released-product hits are multi-invariant, not length-only. The solver
now requires the nick to arise from a real pre-nick nickase recognition site at
the requested boundary, requires any top-prefix fragment that survives left of
the nick to stay Watson-Crick paired across that residual duplex overlap, and
requires the origin-anchored stem/foldback return on the active bottom strand
to remain Watson-Crick paired as well.

Provide at least one explicit nickase source and one explicit release-enzyme source.
Demo-only catalog entries are excluded by default.
Nickases carrying `FREQUENT_CUTTER` are also excluded by default; use
`--allow-frequent-cutter-nickases` only for policy-comparison or historical audits.
The checked-in operational `de033` runbook resolves the whole local built-in
nickase preset surface as `neb_nicking_v1 + thermo_nicking_v1`.

### Failure modes

Common fail-fast cases:

- the spec path is outside `configs/snapback/`
- `stage_order` is not `nick_then_release`
- `retained_side` is not `upstream`
- the effective cap loop is not exactly `3 nt`
- no pre-nick site matches the requested normalized boundary
- no release site produces a fully separated downstream sacrificial fragment
- the selected nickase carries a disallowed warning code such as `FREQUENT_CUTTER`
- a legacy retained-top nick-survival constraint is enabled for an exposed-bottom-strand released-design bundle
- the released-product plot context cannot be derived from the exposed bottom-strand geometry

For the file-by-file bundle layout, use [`../reference/released_snapback_artifacts.md`](../reference/released_snapback_artifacts.md).
For release-enzyme catalog fields and the built-in Type IIS preset, use [`../reference/release_enzyme_catalogs.md`](../reference/release_enzyme_catalogs.md).
