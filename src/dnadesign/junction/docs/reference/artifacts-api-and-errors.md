# Artifacts, API, and errors

**Type:** reference
**Scope:** public operations, plan identity, publication, and verification
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

## Public operations

The CLI and `dnadesign.junction` Python facade expose four task-oriented
operations:

| Operation | Result | Writes? |
| --- | --- | --- |
| `preflight(request)` | Short `PlanSummary` after a complete design. | No. |
| `plan(request)` | Complete immutable `JunctionPlan`. | No. |
| `build(request, destination=...)` | `PublishedJunctionBundle` after create-only publication and replay verification. | Yes. |
| `verify(bundle)` | `BundleVerification` after offline semantic replay. | No. |

The request argument may be a validated `JunctionRequest` or a JSON/YAML path.
`preflight`, `plan`, and `build` each run the complete design independently;
calling one does not cache work for another.

The CLI mirrors the names:

```text
junction preflight REQUEST [--format text|json]
junction plan REQUEST [--format text|json]
junction build REQUEST --output NEW_DIRECTORY [--format text|json]
junction verify BUNDLE [--format text|json]
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

The manifest schema is `dnadesign.junction.bundle.v1`. It identifies five
artifacts by portable path, byte length, and SHA-256:

| Artifact | Purpose | Limit |
| --- | --- | ---: |
| `request.json` | Canonical request used for replay. | 16 MiB |
| `plan.json` | Complete plan and evidence. | 256 MiB |
| `checks.json` | Compact scoped results. | 16 MiB |
| `orders/oligos.tsv` | Vendor-neutral order table. | 256 MiB |
| `views/three_way_junction_review.v1.json` | One neutral review record per target. | 256 MiB |

`manifest.json` is canonical JSON and is not listed inside its own inventory.
`checks.json` subjects are exactly `target` or `assembly_group`. The order TSV
includes `assembly_group_id` and complete sequence identities.

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
avoids retaining all five potentially large payloads at once. A single
renderer still materializes one complete artifact and serializer temporaries.

## Offline verification

`verify` opens the bundle with descriptor-anchored, no-follow filesystem
operations and requires the exact bundle-v1 inventory. It rejects:

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
