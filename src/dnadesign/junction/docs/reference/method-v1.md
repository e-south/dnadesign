---
doc_id: junction-method-v1
title: junction method v1
type: reference
scope: paper-inspired three-way-junction geometry, string objectives, and reconstruction evidence
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-09
---

# Method v1

junction is an independent planning implementation inspired by the
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

The coordinate model below follows the pooled preprint's string-generator
notation. The Nature paper describes the earlier hand- and NUPACK-guided design
context, not this complete software contract.

## Geometry and Locus Enumeration

For a target string `S` of length `N`, the planning profile declares:

| Symbol | Request field | Meaning |
| --- | --- | --- |
| `L` | `nominal_fragment_oligo_length` | nominal fragment-oligo geometry |
| `b` | `barcode_length` | barcode length |
| `t` | `toehold_length` | toehold length |
| `R` | `search_range` | number of candidate offsets at each locus |

junction derives the fragment stride and trailing-domain limit as

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

The first locus exists only when every candidate is complete and leaves a
nonempty terminal target domain:

```text
p_0 + (R - 1) + t < N,
```

which is equivalent to `N >= L - b + R`. After enumerating locus `m`, planning
stops when

```text
N - (p_m + t) <= c_max.
```

Otherwise, it advances to `p_(m+1)` only when that locus also has complete
candidates with a nonempty terminal domain. This stopping rule bounds the last
barcode-bearing strand by `L`; the candidate offset bounds first and internal
strands by `L + R - 1`. Truncated candidates and empty domains are errors; a
target with no complete valid locus cannot be planned by v1. This stopping rule
deliberately differs from the pooled paper's literal next-locus predicate. The
last complement strand can reach `L - b + t`, so the request ceiling must cover
`max(L + R - 1, L - b + t)`. The policy table below records the difference.

Here `L` is a coordinate parameter, not a promise that every ordered strand is
`L` bases long. The Nature paper instead uses `L` for the physical input-oligo
length and derives a coding capacity of `L - 2b`. The v2 request therefore
spells the field `nominal_fragment_oligo_length`. Candidate offsets can yield
fragment orders up to `max(L + R - 1, L - b + t)`, and terminal fragment
orders can be shorter than `L`. The caller-declared
`minimum_fragment_oligo_length` and `max_oligo_length` bound the actual emitted
fragment strings.

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

The papers call the paired complementary oligos “coding oligos.” junction
calls them **complement strands** because an input target need not encode a
protein. For those strands:

```text
K_0 = rc(D_0)
K_j = rc(D_j) rc(t_j)                 for 1 <= j <= n.
```

The target reconstruction check requires

```text
D_0 t_1 D_1 ... t_n D_n = S,
```

and the complement-sequence check requires the reverse fragment order

```text
K_n K_(n-1) ... K_0 = rc(S).
```

Each junction record also names its left and right fragments, `t_i`,
`rc(t_i)`, `b_i`, `rc(b_i)`, complement-nick sequence layout, and the declared
complement-end preparation. These are string checks, not evidence of ligation,
chemical readiness, or assembly yield.

## Sequence Search Scores

All searches operate within one declared assembly group. Target, locus, and
candidate identities are sorted before search; the request seed derives
separate assembly-group-and-stage seeds.

The toehold and barcode searches are maximin-oriented: a larger distance for
the least-separated pair is better. V1 ranks both minimum and mean pairwise
distance, gives the minimum-distance rank twice the weight, and then applies
stable lexical tie-breaking. The formulas below define the v1 implementation.

### Toehold path

`junction` selects one candidate per locus. For zero-based position `i` in a
toehold of length `t`, the pooled preprint defines `u = i / (t - 1)` and the
directional edit penalty

```text
w(u) = 1 + exp(-u).
```

The v1 dynamic program weights substitutions and deletions at the source
position, insertions at the target position, and exact matches at zero. Values
are quantized to fixed-point nanounits. Pair distance is the minimum of the two
directional scores.

Each sampled path follows the preprint's local construction rule. The planner
randomly orders the loci for that trial, chooses the first locus's candidate
uniformly, and then considers each remaining locus in that order. If the
partial path is `v_1, ..., v_q`, a candidate `u_j` has the ideal weight

```text
omega(u_j) = 1 + sum(exp(d(v_l, u_j)) for l in 1..q).
```

The implementation does not evaluate the large exponentials directly. At each
selection step it finds the local maximum distance `M` across that trial's
candidate-to-prior pairs and multiplies every ideal weight by `exp(-M)`. It
therefore evaluates the equivalent shifted form

```text
exp(-M) + sum(exp(d(v_l, u_j) - M) for l in 1..q).
```

Each exponential contribution is scaled by `10^12` and rounded to the nearest
integer with ties to even. Integer weighted sampling then uses those local
fixed-point weights. The common shift preserves the ideal relative weights
before rounding; the fixed-point conversion is a local reproducibility choice,
not a formula stated by the preprint.

For each sampled path, `junction` records the minimum and mean pairwise
distance. The pooled preprint does not fully specify rank normalization or tie
handling.
V1 assigns equal values the same descending dense rank, normalizes the distinct
ranks to `[0, 1]`, and computes

```text
1.0 * rank(minimum) + 0.5 * rank(mean).
```

The largest score wins, followed by stable lexical tie-breaking. Search always
includes the lexicographically first feasible path in addition to the declared
number of seeded sampled paths.

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

For each one-to-one assignment, junction forms `t_i + b_i` and measures the
longest common substring for every pair. It minimizes the worst pairwise value.
When `n <= 8` and `n!` fits within the iteration budget, it evaluates every
permutation. Otherwise, it evaluates the sorted baseline plus stable seeded
samples. The pooled preprint chooses randomly among equally minimal-LCS
mappings; v1 instead uses the smallest lexical assignment so replay is
deterministic.

## Recovery Evidence

Every target declares either target-specific or universal recovery primers.
For forward binding sequence `F`, reverse binding sequence `R`, and exact
5-prime extensions `F_ext` and `R_ext`, the order rows contain

```text
forward_order = F_ext F
reverse_order = R_ext R.
```

The terminal primer check uses only the binding sequences:

- `F = S[0 : len(F)]`;
- `R = rc(S[N - len(R) : N])`;
- the recorded intervals are `[0, len(F))` and `[N - len(R), N)`;
- the first and last fragment identities are explicit; and
- the expected unextended target is exactly `S`.

The evidence also records the exact extension-bearing strands implied by that
declaration:

```text
extended_top    = F_ext S rc(R_ext)
extended_bottom = R_ext rc(S) rc(F_ext)
extended_bottom = rc(extended_top).
```

Target-specific binding pairs must not resolve another target in the same
assembly group. Universal mode requires one complete pair—including
extensions—for the whole group, and the order table lists that pair once with
every consuming target in `target_ids`. These checks do not predict primer
efficiency, off-target behavior outside the declared group, or PCR success.

The pooled paper's universal route also used shared priming regions, variable
buffers to equalize product lengths, internal Type IIS sites, and downstream
removal and hierarchical assembly. V1 does not design or validate those
features. Its `universal` mode means only one exact caller-supplied primer pair
and consolidated order rows.

An extension may carry a user-supplied adapter or restriction-site sequence,
including a sequence intended for later Type IIS cloning. junction preserves
the exact 5-prime DNA but does not identify enzymes, add spacers, infer cleavage
sites, prove post-cleavage overhangs, or plan later cloning.

## How V1 Differs from the Papers

The papers establish the scientific method and motivate the sequence-design
objectives. junction makes additional software choices so each request is
reproducible, inspectable, and rejected explicitly when a constraint cannot be
met.

| Topic | Paper-stated behavior | junction v1 contract |
| --- | --- | --- |
| Terminal locus and order length | The pooled Methods text stops when `N - p_(m+1) <= c_max` and states final oligo lengths in `[L - R + 1, L + R - 1]`. For some target lengths, the literal predicate leaves a terminal barcode-bearing oligo longer than the stated geometry. | V1 stops from the current locus, retains another junction when needed, requires a nonempty terminal domain for every candidate, and covers both the `L + R - 1` offset bound and `L - b + t` terminal-complement bound. This correction can produce a terminal fragment order shorter than the preprint's stated lower bound. The v2 request makes the caller declare `minimum_fragment_oligo_length`; infeasible candidate paths are removed before path ranking and barcode work. |
| Toehold selection scope | The pooled procedure selects a toehold set target by target before global barcode design. | V1 uses a different joint, cross-target-constrained search for all loci in one assembly group. Adding a target may therefore change existing assignments. No comparative laboratory result establishes either search as superior. |
| Substring exclusion | The pooled method starts with `q = floor(t / 2)` for barcode-to-toehold exclusion and `k = max(floor(b / 4), q + 1)` for barcode-to-barcode exclusion, including reverse complements. | `barcode_toehold_k` and `barcode_pair_k` are fixed request fields. The paper-derived values are documented starting points, never inferred at runtime. |
| Constraint pressure | The pooled method requires at least `5|T|` admissible barcodes. If its generator returns fewer, the described software reruns while alternating constraint relaxation: increment `k` on even iterations (or whenever `q >= t`) and increment `q` on eligible odd iterations, halting if the threshold still cannot be met. | Attempt and iteration budgets are explicit. Candidate exhaustion fails under the declared `q` and `k`; junction never changes them automatically. A reviewed replacement request may declare different values. |
| Barcode generation | The pooled Methods text describes seqwalk generation of a maximally sized shared-substring-constrained pool. | V1 uses seeded rejection sampling with explicit GC and homopolymer filters and stops at the declared candidate count. Exhaustion can be a conservative false negative even when another generator could find a feasible pool; it fails rather than changing algorithms or constraints silently. |
| Stochastic search | The design methods use stochastic or sampled selection. | Canonical ordering, stage-specific derived seeds, a lexicographically first feasible baseline path, fixed-point scoring, and stable tie-breaking make a request reproducible. |
| Weighted edit distance and rank aggregation | The pooled method gives `w(u) = 1 + exp(-u)` but does not fully specify directional insertion/deletion coordinates, rank normalization, or tie policy. | V1 names the directional recurrence, symmetrizes by the minimum direction, gives equal values a shared descending dense rank normalized to `[0, 1]`, aggregates as `1.0 * minimum-rank + 0.5 * mean-rank`, and records how many paths it evaluated. |
| Optional thermodynamic filter | The Nature work uses NUPACK-guided or prevalidated barcode choices. The pooled preprint uses string generation and describes an optional thermodynamic filter, inspired by prior NUPACK-based work, for ranking completed candidate sets. | No thermodynamic backend runs. `thermodynamic_screening` is always `not_run`; string checks do not imply thermodynamic orthogonality. |
| Performance and experiments | The publications report timings and experimental results for their implementations and tested designs. | No PyWinder speed or output equivalence is claimed. Software checks do not imply assembly yield or laboratory validity. |
| Recovery and ordering | The pooled paper describes construct-specific (its term) and universal PCR recovery and experimental oligo use. It also warns that PCR can favor shorter products and reports a higher observed junction-misconnection rate in its universal experiment than in the highlighted construct-specific condition. | V1 names the mode `target_specific`, checks exact terminal primer strings, preserves 5-prime extensions, and writes vendor-neutral rows. It does not model amplification bias, design primers, reproduce the paper's universal buffer/Type-IIS architecture, execute a protocol, or place an order. |

For request fields, use the [request contract](request.md). For commands and
publication, use [Artifacts, API, and errors](artifacts-api-and-errors.md).
For full authorship and implementation scope, see [Sources and
scope](sources.md). For resource-sensitive requests, see
[Scale](../guides/scale.md).
