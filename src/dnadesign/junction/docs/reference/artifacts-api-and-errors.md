---
doc_id: junction-artifacts-api-errors
title: junction artifacts, API, and errors
type: reference
scope: public operations, plan identity, publication, and verification
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Artifacts, API, and errors

## Public operations

The CLI and `dnadesign.junction` Python facade expose one request-preparation
route and four design operations. The Python facade also exposes one optional
sequence-comparison plot:

| Operation | Result | Writes? |
| --- | --- | --- |
| `request_from_sequences(...)` | Canonical `JunctionRequest` from normalized sequence records and explicit policy. | No. |
| `preflight(request)` | Short `PlanSummary` after a complete design. | No. |
| `plan(request)` | Complete immutable `JunctionPlan`. | No. |
| `build(request, destination=...)` | `PublishedJunctionBundle` after create-only publication and replay verification. | Yes. |
| `verify(bundle)` | `BundleVerification` after offline semantic replay. | No. |
| `plot_sequence_dissimilarity(request_or_plan, assembly_group_id=...)` | Matplotlib figure for Junction's search metrics. | No. |
| `render_sequence_dissimilarity_svg(request_or_plan, assembly_group_id=...)` | Canonical SVG bytes for the same diagnostic. | No. |

The request argument may be a validated `JunctionRequest` or a JSON/YAML path.
`preflight`, `plan`, and `build` each run the complete design independently;
calling one does not cache work for another.

Both sequence-comparison functions accept a request or an existing
`JunctionPlan`. Pass an existing plan when you have already computed one. Use
`render_sequence_dissimilarity_svg(...)` when byte-for-byte reproducibility
matters. The plot belongs to Junction because it evaluates Junction's string
metrics; BaseRender remains the reusable nucleotide and topology renderer.

The CLI mirrors the names:

```text
junction request --base-request REQUEST (--sequence DNA | --input FILE) --primer-binding-length NT
junction preflight REQUEST [--format text|json]
junction plan REQUEST [--format text|json]
junction build REQUEST --output NEW_DIRECTORY [--format text|json]
junction verify BUNDLE [--format text|json]
```

`junction request` writes canonical JSON to standard output. It accepts one
raw sequence, one whitespace-tolerant text sequence, or one or more FASTA
records. `--base-request` contributes only the existing request's seed, planning
profile, and order policy. The target list is replaced. The command derives
terminal binding strings at the explicit length; it does not assess PCR.

Python callers can use `sequence_record(...)` or
`load_sequence_records(...)`, then pass the result to
`request_from_sequences(...)`. The resulting request goes through the same
`preflight`, `plan`, `build`, and `verify` operations as a hand-authored
request.

```python
from dnadesign.junction import build, load_request, load_sequence_records, request_from_sequences

base = load_request("reviewed-request.yaml")
records = load_sequence_records("targets.fasta")
request = request_from_sequences(
    records,
    planning=base.planning,
    order_policy=base.order_policy,
    seed=base.seed,
    primer_binding_length=20,
)
bundle = build(request, destination="outputs/junction/design-01")
```

Successful preflight reports `status: planned`, `validation_scope:
string_only`, and `thermodynamic_screening: not_run` with target, assembly
group, junction, and order counts.

## Failure contract

After command dispatch, configuration, design, and bundle failures exit with
status `1`. JSON mode writes to standard error:

```json
{
  "status": "error",
  "error": {
    "code": "design_error",
    "message": "human-readable failure detail",
    "retryable": false
  }
}
```

`error.code` is `config_error`, `design_error`, `bundle_error`, or the fallback
`junction_error`. `retryable: false` means an identical request should not be
blindly retried. Typer handles command-line syntax errors before dispatch and
uses its own error presentation.

Python callers receive `JunctionConfigError`, `JunctionDesignError`, or
`JunctionBundleError`, all derived from `JunctionError`.

## Plan identity

The plan schema is `dnadesign.junction.plan.v1`; the current algorithm is
`dnadesign.junction.string.v1`. Canonical request order, a caller seed, and
assembly-group-derived stage seeds make the result reproducible. The plan ID is
the SHA-256 identity of the canonical plan without its own `plan_id` field.

The plan contains:

- `assembly_groups`: selected junction assignments and search evidence;
- `targets`: fragment layouts, strand strings, recovery expectations, and
  exact reconstructions;
- `orders`: complete vendor-neutral oligo and primer rows; and
- `checks`: target- or assembly-group-scoped results, including explicit
  `not_run` status.

Search backends are private composition points. A future optimizer that changes
meaning or scale behavior needs a new algorithm identifier and its own
validation evidence; it cannot silently replace method-v1 semantics.

## Bundle inventory

The manifest schema is `dnadesign.junction.bundle.v2`. It identifies nine
artifacts by portable path, byte length, and SHA-256:

| Artifact | Purpose | Limit |
| --- | --- | ---: |
| `request.json` | Canonical request used for replay. | 16 MiB |
| `plan.json` | Complete plan and evidence. | 256 MiB |
| `checks.json` | Compact scoped results. | 16 MiB |
| `orders/oligos.tsv` | Vendor-neutral order table. | 256 MiB |
| `sequences/targets.fasta` | Normalized submitted targets. | 256 MiB |
| `sequences/oligos.fasta` | Every complete orderable oligo and primer. | 256 MiB |
| `sequences/expected_pcr_products.fasta` | Expected primer-extended top strands. | 256 MiB |
| `views/three_way_junction_review.v1.json` | One neutral review record per target. | 256 MiB |
| `views/junction_sequence_dissimilarity.v1.json` | One compact sequence-comparison record per assembly group. | 256 MiB |

`manifest.json` is canonical JSON and is not listed inside its own inventory.
`checks.json` subjects are exactly `target` or `assembly_group`. The order TSV
includes `assembly_group_id` and complete sequence identities.

The layout stays shallow by role:

```text
<bundle>/
├── manifest.json
├── request.json
├── plan.json
├── checks.json
├── orders/
│   └── oligos.tsv
├── sequences/
│   ├── targets.fasta
│   ├── oligos.fasta
│   └── expected_pcr_products.fasta
└── views/
    ├── three_way_junction_review.v1.json
    └── junction_sequence_dissimilarity.v1.json
```

`junction` does not place rendered figures in this bundle. BaseRender consumes
one view record and writes a separate create-only figure bundle only when a
plot is requested.

The TSV is the stable synthesis handoff. Supplier upload columns and ordering
APIs change independently, so a supplier adapter should consume this table
and own its own versioned contract. Junction does not label a generic CSV as a
vendor-ready order.

## Publication

Publication is create-only, not filesystem immutability. The destination must
not exist, every existing path component must resolve without symlinks, and the
publisher never replaces caller-owned work. It writes into an adjacent private
stage, verifies the staged bytes, atomically installs the directory, and
performs a full post-install replay. A failed post-install verification rolls
back the just-published directory.

New bundle roots use mode `0700`; files use `0600`. These modes reduce
accidental local disclosure but do not replace repository, sync, backup, or
sharing policy.

Each artifact is rendered, written, hashed, and released before the next one is
rendered. Staged verification receives only compact artifact identities. This
avoids retaining all nine potentially large payloads at once. A single
renderer still materializes one complete artifact and serializer temporaries.

## Offline verification

`verify` opens the bundle with descriptor-anchored, no-follow filesystem
operations and requires the exact bundle-v2 inventory. It rejects:

- missing, extra, moved, aliased, non-regular, or symlinked entries;
- declared sizes above the verification limits;
- byte-length or digest mismatches;
- malformed or non-canonical JSON/TSV payloads;
- a request that cannot reproduce a valid plan-v1 artifact; and
- any rendered artifact that differs from replay.

Only request bytes are retained long enough to parse the embedded request.
Descriptor identity records retain byte counts and digests, not file payloads.
After an initial path and inventory pass, verification streams each retained
file once more, compares its SHA-256, and repeats the path and inventory checks
against the post-hash metadata checkpoint. Replayed expected artifacts are also
compared one at a time, then released.

Verification detects later changes; it does not prevent external mutation.

## Scope

The bundle does not own target discovery, study semantics, biological
identity, laboratory protocols, thermodynamic simulation, supplier APIs,
purchasing, or experimental acceptance. Those decisions remain with the
project or process using the design.
