---
doc_id: study-workspace-contract
surface: study-architecture
owner: dnadesign-maintainers
last_verified: 2026-08-08
---

## Study workspace contract

Use `study-catalog/v1` when studies live outside the dnadesign repository. The
contract answers three questions without interpreting the science:

1. Which programs and studies exist?
2. Where does a reader start for each study and tool handoff?
3. Which review artifacts are available, stale, superseded, or blocked?

The catalog does not select a global active study. Callers must name the study
they intend to inspect or run.

### Catalog

Place the catalog at `catalog/studies.yaml`:

```yaml
schema: study-catalog/v1
programs:
  - program_id: stress_response
    title: Stress response
    entrypoint: programs/stress-response/README.md
studies:
  - study_id: promoter_response
    manifest: programs/stress-response/studies/promoter-response/study.yaml
```

`program_id` groups related work for navigation. It does not grant one study
authority over another.

### Study manifest

Each catalog entry points to one `study.yaml`:

```yaml
schema: study/v1
study_id: promoter_response
program_id: stress_response
title: Promoter response
summary: Measures promoter responses across declared conditions.
visibility: private
status: active
owners:
  - stress-study-maintainers
last_verified: 2026-08-08
entrypoint: README.md
operations: operations/ops.study.yaml
evidence_index: evidence/index.yaml
workflows:
  - tool_id: reader
    route: workflows/reader/README.md
    requires: reader-workbench>=0.1
  - tool_id: dnadesign
    route: workflows/dnadesign/README.md
    requires: dnadesign>=0.1
```

The manifest owns identity and routes, not formulas, sample metadata, or tool
configuration. Put those details behind the declared entrypoints.

### Evidence index

`study-evidence-index/v1` records reviewable outputs without putting raw data
or large generated bundles in Git:

```yaml
schema: study-evidence-index/v1
study_id: promoter_response
artifacts:
  - artifact_id: response_review
    artifact_type: review-figure
    status: available
    path: review/response-review.svg
    media_type: image/svg+xml
    content_digest: sha256:<64 lowercase hex characters>
    source_revisions:
      reader: record:response-window@sha256:<digest>
    generated_by:
      - uv
      - run
      - study
      - render
      - promoter_response
```

An available, stale, or superseded artifact must declare exactly one tracked
`path` or external `uri`, a SHA-256 digest, source revisions, and the command
that produced it. A blocked artifact declares a `blocker` and no output
location.

### Python API

```python
from pathlib import Path

from dnadesign.studies.core import load_study_workspace

workspace = load_study_workspace(Path("../research-studies"))
study = workspace.study_index["promoter_response"]
print(study.entrypoint)
```

Loading fails on unknown keys, duplicate identifiers, undeclared programs,
identity drift, missing routes, path traversal, symlink escapes, invalid
artifact states, and tracked-file digest mismatches. External URIs are syntax
checked but not downloaded; the owning study CI verifies remote availability.

### Ownership boundary

- dnadesign owns the loader and validation rules.
- The external repository owns catalog instances, study meaning, and evidence.
- Reader, dnadesign, and OPAL remain owners of their public workflow APIs.
- Workspace routers may point to the catalog but must not copy study facts.
