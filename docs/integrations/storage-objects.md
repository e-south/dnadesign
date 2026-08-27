---
doc_id: external-storage-objects
surface: integration-contract
owner: dnadesign-maintainers
last_verified: 2026-08-27
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
demo is capped at 2 MB, and its manifest, empty coordination lock, and every
resource must be tracked and match their stage-0 Git index bytes.
When one routed root contains multiple demos, they must share one resolved Git
checkout. Root validation binds that shared index before revalidation and
rechecks it after every object pass; it rejects demos split across independent
checkouts because Git cannot provide one atomic multi-repository index snapshot.
Operational objects must set `demo: false` and live outside a Git checkout.
Place the external root beneath an account-private filesystem boundary. The v1
manifest verifies routing and bytes; operating-system ACLs remain an explicit
operator responsibility.

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
  --metadata README.md \
  --cache runtime/fontconfig/cache.bin
```

The command is create-only. It inventories every regular file, writes one
deterministic manifest, verifies exact closure, and refuses to overwrite an
existing manifest. Paths passed with `--input`, `--metadata`, or `--cache`
receive those roles; remaining workspace/store files are artifacts, while all
tool-cache files are cache material.

For `--demo`, existing resources must already be small, tracked, and byte-for-byte
identical to their Git index entries. Inventory creates the new manifest with
status `created-pending-git-add`; use the returned command to stage the manifest
and persistent coordination lock, then run `dnadesign-storage validate` to
reach `verified`. Unstaged demo edits fail so a manifest-only commit cannot
describe bytes absent from a clean checkout. Operational objects never use
this two-step exception.

An active workspace or durable store may change only through its owning tool.
After a successful run or store transaction, refresh its receipt with the
digest of the receipt that authorized the write:

```bash
uv run dnadesign-storage refresh /absolute/path/to/object \
  --expected-manifest-digest sha256:<digest> \
  --producer-revision <revision-that-produced-the-new-bytes> \
  --json
```

Refresh preserves identity and existing roles, records the revision that
produced the refreshed bytes, inventories new artifacts, and accepts `--cache`
for newly created cache files. Input and metadata bytes are protected: removing
or changing them fails rather than silently authorizing a new input identity.
For a demo refresh, changed resources must already be staged and match their Git
index entries, while the existing manifest and empty coordination lock must
already match their indexed blobs. The refreshed manifest alone enters
`refreshed-pending-git-add`; the returned command stages the manifest and
re-stages the lock idempotently to restore the fully `verified` state.
If an initial receipt incorrectly classified a mutable operational ledger as
metadata, an operator may name that existing path with `--artifact`. This
one-way, compare-and-swap-protected correction permits only `metadata` to
`artifact`. Similarly, `--cache` may explicitly demote an existing artifact to
rebuildable cache material while continuing to classify newly discovered cache
files. Inputs remain immutable, and metadata cannot be reclassified as cache.
For example, USR `.events.log` is an append-only artifact, not immutable
metadata.
Refresh uses the expected digest as a compare-and-swap guard against concurrent
receipt changes. Writers lock
`<object>/.storage-object.lock`, so processes and compute nodes that see the
same POSIX filesystem serialize receipt updates. The lock is contract-owned
coordination state, is excluded from the content manifest, and must remain at a
stable pathname and inode for the lifetime of an object. Inventory alone may
bootstrap it before the first receipt; refresh and validation reject absence.
Before a writer reports success, it rechecks the lock pathname, mode, and shared
posture against the inode held by its open lock descriptor. If that binding
changes after a receipt commits, the call raises
`StorageObjectPublicationUncertain` and requires explicit revalidation of the
committed object instead of returning a false verified result.
If lock release itself fails after a verified commit, the same typed uncertainty
reports the winning manifest digest and an exact validation command; callers
must not retry with the prior compare-and-swap digest.
Group-writable
object roots must also be group-traversable, set the POSIX setgid bit, and not
set the sticky bit. Other-writable object roots are rejected because unrelated
accounts cannot participate in a trusted shared coordination boundary. These
fail-fast requirements let collaborators reach
coordination files, ensure locks, staging files, and newly inventoried manifests
inherit the shared directory group instead of the writer's primary group, and
allow a group collaborator to atomically replace a receipt owned by another
collaborator. They are created
group-writable so collaborating accounts can participate in the same
coordination boundary. Declared resource files must inherit that group and be
group-readable; their parent directories must inherit the group and be
group-readable and traversable. Manifest staging occurs inside the object so
publication stays on the same filesystem. Create-only inventory commits with
an atomic hard-link-if-absent operation after preflighting the no-replace
primitive required for conditional rollback. Refresh commits with a native atomic
file exchange on Darwin or Linux, validates the displaced receipt against the
exact inode and bytes that authorized the refresh, and swaps back only that
verified prior receipt when publication must be rolled back. If the displaced
pathname changes, the published candidate and changed entry are retained for
explicit recovery. If a last-boundary race exchanges an unverified entry, the
verified candidate is restored before the operation reports uncertainty. A
platform or filesystem without the required primitive fails closed as
`StorageObjectPublicationUnsupported`. If a failed publication also cannot use
the no-replace primitive required for ownership-safe staging cleanup, it raises
`StorageObjectPublicationUncertain` and retains the staging entry for explicit
recovery. If an exchange cannot prove that its
swap-back completed, it raises `StorageObjectPublicationUncertain` and retains
both named files for explicit recovery rather than guessing which receipt may
be deleted. Rollback uses the same conditional primitives: refresh restores the
prior receipt only after atomically displacing the receipt this operation
published, while failed create-only inventory quarantines and identifies the
current receipt before removing it. A competing receipt is restored or retained
for recovery; it is never silently overwritten or deleted. A pre-existing
staging-shaped name fails closed for explicit operator inspection; the tool
never guesses that such bytes are safe to delete. Automatic cleanup first
atomically displaces an entry through open directory descriptors into a
per-OS-owner cleanup directory that only that owner can modify. It verifies the
moved inode and deletes it only inside that protected namespace; a shared-path
replacement is restored or retained instead of being unlinked. Empty cleanup
directories are persistent coordination boundaries. Shared roots normalize
them to mode `0750` even under a restrictive umask so the owning group can
inspect recovery state without gaining write access; private roots use `0700`.
Independently synced replicas,
including separate Dropbox clients, are not one shared
filesystem; keep one writer or provide an external coordination service for
those replicas.

Every successful object inventory, validation, or refresh summary includes
`manifest_digest`. Use that exact value as the next refresh command's
`--expected-manifest-digest`; do not race by hashing the receipt separately.

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
semantic preflight before execution. Validation makes two complete digest and
closure passes and rejects observed drift, but producers must still be
quiescent or participate in their own transaction or locking contract while a
receipt is verified. Root validation also fully revalidates each object before
returning, so receipt or resource drift observed while later objects are
checked fails closed instead of returning a stale root snapshot.

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
