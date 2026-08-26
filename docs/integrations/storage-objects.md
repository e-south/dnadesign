---
doc_id: external-storage-objects
surface: integration-contract
owner: dnadesign-maintainers
last_verified: 2026-08-26
---

## External storage objects

Use `storage/` for private or large runtime material that must not live inside a
public Git checkout. The name is deliberate: the root contains tool workspaces,
durable stores, and rebuildable caches, not only datasets.

Each object keeps its producing tool's internal schema. The neutral
`dnadesign.storage-object/v1` manifest adds only the cross-tool facts needed to
locate, retain, inventory, and verify the object without interpreting its
scientific meaning.

### Routed layout

```text
storage/
  workspaces/<owner-tool>/<storage-id>/storage.object.json
  stores/<owner-tool>/<storage-id>/storage.object.json
  tool-cache/<owner-tool>/<storage-id>/storage.object.json
```

The directory route, manifest `object_kind`, `owner_tool`, and `storage_id` must
agree. `AGENTS.md` is the only allowed root-level routing file. Unknown shelves,
other stray files, symlinks, duplicate identities, undeclared files, missing
files, and digest mismatches fail validation.

### Object classes

| Kind | Meaning | Typical storage class |
| --- | --- | --- |
| `workspace` | One tool-owned run or campaign instance | `authoritative`, `reproducible`, or `cold` |
| `store` | A durable tool-owned record or dataset root | `authoritative` |
| `tool-cache` | Rebuildable installations or downloaded model state | `cache` |

`demo: true` is a narrow exception for small tracked examples inside Git. A
demo is capped at 2 MB, and its manifest and every resource must be tracked.
Operational objects must set `demo: false` and live outside a Git checkout.

### Inventory and validation

Inventory an existing object once. Declare input files explicitly; all other
files are assigned the object's default output role.

```bash
uv run dnadesign-storage inventory /absolute/path/to/object \
  --storage-id example-run \
  --owner-repository dnadesign \
  --owner-tool cruncher \
  --object-kind workspace \
  --content-schema cruncher.workspace \
  --content-schema-version 1 \
  --producer-revision <git-revision> \
  --storage-class reproducible \
  --retention-policy review-before-delete \
  --input configs/config.yaml \
  --metadata README.md
```

The command is create-only. It inventories every regular file, writes one
deterministic manifest, verifies exact closure, and refuses to overwrite an
existing manifest. Paths passed with `--input` or `--metadata` receive those
roles; remaining workspace/store files are artifacts, while all tool-cache
files are cache material.

For `--demo`, existing resources must already be small and tracked. Inventory
creates the new manifest with status `created-pending-git-add`; add that
manifest to Git, then run `dnadesign-storage validate` to reach `verified`.
Operational objects never use this two-step exception.

An active workspace may change only through its owning tool. After a successful
run, refresh its receipt with the digest of the receipt that authorized the run:

```bash
uv run dnadesign-storage refresh /absolute/path/to/object \
  --expected-manifest-digest sha256:<digest> \
  --producer-revision <revision-that-produced-the-new-bytes> \
  --json
```

Refresh preserves identity and existing input/metadata roles, records the
revision that produced the refreshed bytes, inventories new artifacts, rejects
missing input or metadata files, and uses the expected digest as a
compare-and-swap guard against concurrent receipt changes. Writers lock
`<object>/.storage-object.lock`, so processes and compute nodes that see the
same POSIX filesystem serialize receipt updates. The lock is contract-owned
coordination state and is excluded from the content manifest. Independently
synced replicas, including separate Dropbox clients, are not one shared
filesystem; keep one writer or provide an external coordination service for
those replicas.

Verify one object before a tool consumes it:

```bash
uv run dnadesign-storage validate /absolute/path/to/object --json
```

Verify the complete routed root before broad maintenance or migration:

```bash
uv run dnadesign-storage validate-root /absolute/path/to/storage --json
```

Storage validation establishes byte identity and retention posture. The owning
tool must still load its own config or store schema and perform its normal
semantic preflight before execution.

### Adoption order

Move only after both checks pass:

1. Copy the complete object to its routed destination.
2. Inventory and validate the external copy.
3. Run the owning tool against the external path.
4. Compare accepted artifacts or state with the source.
5. Update durable callers to use the external path.
6. Retire the embedded copy only when it is reproducible or separately retained.

Caches can move after cache-specific acceptance. Authoritative stores require a
consumer cutover and must not be inferred from file hashes alone. Historical
output-only workspaces should enter `cold` storage without being promoted to
current semantic authority.
