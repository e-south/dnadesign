## Distribution builds and verification

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-06

DNADesign has one distribution version, declared in `pyproject.toml`. Runtime
`dnadesign.__version__` reads installed metadata. Individual tools can version
their API and artifact contracts independently without creating separate release
trains. Downstream studies pin an exact qualified commit or artifact and run their
own numerical comparisons before upgrading.

### Build a reviewable bundle

Use a clean committed checkout and Python 3.12. The destination must be a new
directory outside the checkout. This command does not publish, tag or install a
release into the working environment.

```bash
uv run --locked --no-default-groups python -m dnadesign.devtools.package.build \
  --repo-root . --output /tmp/dnadesign-build
```

The command exports only Git's committed files, resolves and retains hash-locked
build requirements, and builds wheel and source distribution twice in a temporary
environment. It rejects uncommitted source, differing wheel bytes and differing
source-distribution member contents. It records whether the source archives are
also byte-identical; container timestamps can differ despite identical contents.
The temporary source and environment are removed when the command finishes.

`provenance.json` binds the source commit/tree, source timestamp, project/runtime
lock hashes, build tool versions, repeated-build comparison and artifact hashes.
The bundle includes a versioned copy of [CITATION.cff](../../CITATION.cff), build
requirements, wheel, source archive and command logs. Retain the receipt checksum
independently alongside a downstream pin. Logs document the local build; the
receipt and artifact checks alone do not authenticate a publisher.

To repeat the same build-tool resolution:

```bash
uv run --locked --no-default-groups python -m dnadesign.devtools.package.build \
  --repo-root . --output /tmp/dnadesign-build-repeat \
  --build-lock /tmp/dnadesign-build/build-requirements.lock
```

This requires the same source commit, Python/uv toolchain and retained build
requirements. Runtime dependencies remain separately declared in `uv.lock`.
Python and system-tool/platform compatibility require their own execution checks.

### Verify a retained or downloaded bundle

```bash
uv run --locked --no-default-groups python -m dnadesign.devtools.package.provenance \
  /tmp/dnadesign-build --receipt-sha256 <independently-recorded-sha256>
```

The verifier checks every declared file and rejects incomplete bundles, duplicate
paths, changed bytes and paths or symlinks that escape the bundle. Omitting the
receipt checksum checks internal consistency only. It does not install the wheel
or establish numerical compatibility with any study.

### Acceptance and version policy

Run the existing installed-distribution contracts before proposing a release:

```bash
uv run --locked pytest -q src/dnadesign/devtools/tests/package
```

These tests build a wheel and source archive, install outside the source checkout,
exercise NumPy-only OPAL scoring, install the full locked dependency set, check
Construct placement and invoke OPAL, Construct, LatentDNA and DenseGen CLI help.
They also exercise build and download verification. The normal CI test scope
includes this package suite. A successful build is separate from passing these
runtime checks and the relevant tool tests.

Before an authorized public release, check existing tags/assets, choose an unused
version, and update only `project.version`. Use a prerelease suffix for a preview;
never replace files under a published version. A release tag must be
`v<project.version>` at the reviewed source commit; confirm wheel/source metadata
and generated citation match before uploading. Keep build provenance and hashes
with the assets. Publication and downstream pin adoption are separate decisions.
During 0.x development, use a new minor version for incompatible supported API
changes and a patch version for compatible corrections. Describe affected tools
and migration requirements in the release notes; a version increment alone does
not qualify historical replay.

The base install supports NumPy-only multistate scoring. Other current tool CLIs
require `[full]`; the extra pins the qualified Dense Arrays 0.2.0 registry release.
Version-based wheel installation does not provide Git provenance. Downstream
verification must bind retained artifact bytes to installed content rather than
invent a Git commit from `direct_url.json`.

### Current unreleased changes

- OPAL: NumPy-only public scoring installation, lazy runtime imports and moved
  campaign-state rebinding.
- DenseGen: ten sequence-packing examples through the existing playback surface.
- Package: one version authority, explicit full-runtime extra, clean build
  provenance and portable bundle verification.

These are working changes at version 0.2.0, not a declaration of a public release
or historical scientific equivalence. See the owning tools' existing docs and
examples for their contracts.

### Package-page images

The README is also the PyPI description. Keep its links absolute and use a
PNG banner at an immutable source commit or the matching `v<version>` tag.
Never use a relative asset path or a mutable branch for a release image. Keep
published tags and their assets; changing a new banner must not change older
release pages. When incrementing the version, update a version-bound README
image URL in the same change. The package tests enforce this relationship.

Before publishing, fetch the banner URL after the tag exists and compare its
bytes with the source PNG. Check the rendered PyPI page after upload. The PNG is
a package-page export; its editable SVG remains the artwork source.

### Publish a qualified release

The PyPI distribution is `dnadesign-tools`; imports and tool commands keep their
existing names. Merge the reviewed release changes and wait for successful CI on
the exact main commit. Create an annotated `v<version>` tag and a GitHub release.
The `release.yaml` workflow verifies main ancestry, exact main CI, the lock and
the public banner, then calls the same maintained build command above. It tests
the NumPy-only scoring install and ordinary `[full]` installation before the
separate OIDC publishing job runs in the tag-restricted `pypi` environment.

Keep the Trusted Publisher restricted to `e-south/dnadesign`, `release.yaml`, and
`pypi`. Download the workflow's distributions and release-evidence artifacts,
attach the distributions and provenance to the GitHub release, and compare both
artifact hashes with PyPI. Verify an explicit registry installation and inspect
the rendered package page. A successful source build alone is not publication.
