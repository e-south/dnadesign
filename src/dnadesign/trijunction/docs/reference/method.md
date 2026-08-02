# TriJunction Method Reference

**Type:** reference
**Scope:** paper-inspired three-way-junction geometry, string-level
design objectives, and reconstruction evidence
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

TriJunction is an independent planning implementation inspired by the
Sidewinder three-way-junction method and its pooled-oligo extension:

- Robinson *et al.*, “Construction of complex and diverse DNA sequences using
  DNA three-way junctions,” *Nature* (2026),
  [doi:10.1038/s41586-025-10006-0](https://doi.org/10.1038/s41586-025-10006-0).
- Robinson *et al.*, “One-pot parallel Sidewinder construction from oligo
  pools,” *bioRxiv* preprint (2026),
  [doi:10.64898/2026.05.01.722326](https://doi.org/10.64898/2026.05.01.722326).

This reference describes exactly what the current implementation computes. It
does not claim equivalence with the authors' PyWinder software, reproduce their
experiments, or show that a plan will work in the laboratory.

## Geometry and Locus Enumeration

For a target string `S` of length `N`, the planning profile declares:

| Symbol | Request field | Meaning |
| --- | --- | --- |
| `L` | `oligo_length` | nominal strand-length geometry |
| `b` | `barcode_length` | barcode length |
| `t` | `toehold_length` | toehold length |
| `R` | `search_range` | number of candidate offsets at each locus |

TriJunction derives the fragment stride and trailing-domain limit as

```text
f = L - 2b
c_max = L - b
```

The base start for locus `m`, indexed from zero, is

```text
p_0 = L - b - t
p_m = p_0 + mf
```

Locus `m` contains the `R` complete candidate windows

```text
S[p_m + r : p_m + r + t],  r in {0, ..., R - 1}.
```

The first locus exists only when
`p_0 + (R - 1) + t <= N`. After enumerating locus `m`, planning stops when

```text
N - (p_m + t) <= c_max.
```

Otherwise, it advances to `p_(m+1)`. This stopping rule bounds the last
barcode-bearing strand by `L`; the candidate offset bounds first and internal
strands by `L + R - 1`. Truncated candidates are errors; a target with no
complete locus cannot be planned by v1. This stopping rule deliberately differs
from the pooled paper's literal next-locus predicate. The policy table below
records the difference.

## Selected Domains and Strand Orientation

Let `a_i` be the selected start for junction `i`, with junctions indexed
`i = 1, ..., n`. Define the selected toehold and target domains as

```text
t_i = S[a_i : a_i + t]
D_0 = S[0 : a_1]
D_j = S[a_j + t : a_(j+1)]       for 1 <= j < n
D_n = S[a_n + t : N].
```

Let `b_i` be the barcode assigned to junction `i`, juxtaposition denote
concatenation, and `rc(x)` denote reverse complement. Every sequence below,
including every order row, is written in the physical **5-prime to 3-prime**
direction.

For the barcode-bearing strands:

```text
B_0 = D_0 t_1 b_1
B_j = rc(b_j) D_j t_(j+1) b_(j+1)    for 1 <= j < n
B_n = rc(b_n) D_n.
```

The papers call the paired complementary oligos “coding oligos.” TriJunction
calls them **complement strands** because an input target need not encode a
protein. For those strands:

```text
K_0 = rc(D_0)
K_j = rc(D_j) rc(t_j)                 for 1 <= j <= n.
```

The target reconstruction proof checks

```text
D_0 t_1 D_1 ... t_n D_n = S,
```

and the complement-strand proof checks that ligation in reverse fragment order

```text
K_n K_(n-1) ... K_0 = rc(S).
```

Each junction record also names its left and right fragments, `t_i`,
`rc(t_i)`, `b_i`, `rc(b_i)`, complement-nick geometry, and the declared
complement-end preparation. These are sequence-geometry proofs, not
chemical-readiness or assembly-yield claims.

## Sequence Search Scores

All searches operate within one declared physical pool. Target, locus, and
candidate identities are sorted before search; the request seed derives
separate pool-and-stage seeds.

The toehold and barcode searches are maximin-oriented: a larger distance for
the least-separated pair is better. V1 ranks both minimum and mean pairwise
distance, gives the minimum-distance rank twice the weight, and then applies
stable lexical tie-breaking. The formulas below are authoritative.

### Toehold path

TriJunction selects one candidate per locus. For a sequence position normalized
to `u` in `[0, 1]`, the directional edit weight is

```text
w(u) = 1 + exp(-u).
```

The v1 dynamic program weights substitutions and deletions at the source
position, insertions at the target position, and exact matches at zero. Values
are quantized to fixed-point nanounits. Pair distance is the minimum of the two
directional scores.

For each sampled path, TriJunction records the minimum and mean pairwise
distance. The papers do not fully specify rank normalization or tie handling.
V1 assigns equal values the same descending dense rank, normalizes the distinct
ranks to `[0, 1]`, and computes

```text
1.0 * rank(minimum) + 0.5 * rank(mean).
```

The largest score wins, followed by stable lexical tie-breaking. Search always
includes the first-candidate baseline and then applies the declared seeded
iteration budget.

### Barcode pool and subset

The generator requests
`barcode_pool_factor * number_of_selected_toeholds` candidates. A candidate
must satisfy the declared GC and homopolymer bounds and all of these string
constraints, including reverse complements:

- no duplicate barcode or reverse-complement duplicate;
- no shared `q`-mer with a selected toehold, where
  `q = barcode_toehold_k`;
- no self or cross-barcode shared `k`-mer, where
  `k = barcode_pair_k`.

Generation stops at the requested pool size or fails when the declared attempt
budget is exhausted. It never weakens a constraint. Barcode subsets are scored
by minimum and mean conventional Levenshtein distance and use the same
`1.0/0.5` dense-rank aggregation and stable tie-breaking as the toehold path.

### Toehold-to-barcode matching

For each one-to-one assignment, TriJunction forms `t_i + b_i` and measures the
longest common substring for every pair. It minimizes the worst pairwise value.
When `n <= 8` and `n!` fits within the iteration budget, it evaluates every
permutation. Otherwise, it evaluates the sorted baseline plus stable seeded
samples. The smallest lexicographic assignment resolves equal scores.

## Recovery Evidence

Every target declares either target-specific or universal recovery primers.
For forward binding sequence `F`, reverse binding sequence `R`, and exact
5-prime extensions `F_ext` and `R_ext`, the order rows contain

```text
forward_order = F_ext F
reverse_order = R_ext R.
```

The terminal geometry proof uses only the binding sequences:

- `F = S[0 : len(F)]`;
- `R = rc(S[N - len(R) : N])`;
- the recorded intervals are `[0, len(F))` and `[N - len(R), N)`;
- the first and last fragment identities are explicit; and
- the expected core recovery product is exactly `S`.

The evidence also records the exact extension-bearing strands implied by that
declaration:

```text
extended_top    = F_ext S rc(R_ext)
extended_bottom = R_ext rc(S) rc(F_ext)
extended_bottom = rc(extended_top).
```

Target-specific binding pairs must not resolve another target in the same
pool. Universal mode requires one complete pair—including extensions—for the
whole pool, and the order table lists that pair once with every consuming
target in `target_ids`. These checks do not predict primer efficiency,
off-target behavior outside the declared pool, or PCR success.

An extension may carry a user-supplied adapter or restriction-site sequence,
including a sequence intended for later Type IIS cloning. TriJunction preserves
the exact 5-prime DNA but does not identify enzymes, add spacers, infer cleavage
sites, prove post-cleavage overhangs, or plan later cloning.

## How V1 Differs from the Papers

The papers establish the scientific method and motivate the sequence-design
objectives. TriJunction makes additional software choices so each request is
reproducible, inspectable, and rejected explicitly when a constraint cannot be
met.

| Topic | Paper-stated behavior | TriJunction v1 contract |
| --- | --- | --- |
| Terminal locus | The pooled Methods text stops when `N - p_(m+1) <= c_max`. For some lengths that leaves a terminal barcode-bearing oligo longer than the stated geometry. | V1 stops when `N - (p_m + t) <= c_max`, retaining another junction when necessary so the last barcode-bearing strand is at most `L` and every candidate-offset order is at most `L + R - 1`. Edge-case tests preserve this deliberate correction. |
| Toehold selection scope | The pooled procedure selects a toehold set target by target before global barcode design. | V1 jointly optimizes all toehold loci that share one declared physical pool. Adding a target to that pool may therefore change existing target junctions; the stronger cross-target maximin objective is intentional and tested. |
| Substring exclusion | The pooled method starts with `q = floor(t / 2)` for barcode-to-toehold exclusion and `k = max(floor(b / 4), q + 1)` for barcode-to-barcode exclusion, including reverse complements. | `barcode_toehold_k` and `barcode_pair_k` are fixed request fields. The paper-derived values are documented starting points, never inferred at runtime. |
| Constraint pressure | The pooled method requires at least `5|T|` admissible barcodes. If its generator returns fewer, the described software reruns while alternating constraint relaxation: increment `k` on even iterations (or whenever `q >= t`) and increment `q` on eligible odd iterations, halting if the threshold still cannot be met. | Attempt and iteration budgets are explicit. Candidate exhaustion fails under the declared `q` and `k`; TriJunction never changes them automatically. A reviewed replacement request may declare different values. |
| Barcode generation | The pooled Methods text describes seqwalk generation of a maximally sized shared-substring-constrained pool. | V1 uses seeded rejection sampling with explicit GC and homopolymer filters and stops at the declared candidate count. Exhaustion can be a conservative false negative even when another generator could find a feasible pool; it fails rather than changing algorithms or constraints silently. |
| Stochastic search | The design methods use stochastic or sampled selection. | Canonical ordering, stage-specific derived seeds, a baseline candidate, fixed-point scoring, and stable tie-breaking make a request reproducible. |
| Weighted edit distance and rank aggregation | The pooled method gives `w(u) = 1 + exp(-u)` but does not fully specify directional insertion/deletion coordinates, rank normalization, or tie policy. | V1 names the directional recurrence, symmetrizes by the minimum direction, gives equal values a shared descending dense rank normalized to `[0, 1]`, aggregates as `1.0 * minimum-rank + 0.5 * mean-rank`, and records how many paths it evaluated. |
| Thermodynamics and performance | The publications report experimental assemblies and method validation. | `thermodynamic_screening` is always `not_run`; string checks do not imply thermodynamic orthogonality, assembly yield, or laboratory validity. |
| Recovery and ordering | The pooled paper describes construct-specific (its term) and universal PCR recovery and experimental oligo use. | V1 names the mode `target_specific` and records binding geometry, exact 5-prime extensions, and vendor-neutral order rows in a new, digest-verified bundle; it does not execute a protocol, interpret downstream adapters, or place an order. |

For schema fields, commands, and publication rules, continue to the
[contract reference](contracts.md). For full authorship and implementation
scope, see [sources and scope](sources.md). For resource-sensitive request
shapes and optional BaseRender review, see [scale and quality
review](../guides/scale-and-review.md).
