---
doc_id: junction-docs
title: junction documentation
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-09
---

# `junction` documentation

Start with a complete request: exact linear DNA targets, which targets must be
designed together, recovery primers, bounded search settings, and order
labels. `junction` does not guess those choices from a bare sequence.

## Learn

- [How `junction` works](explanation/how-junction-works.md) explains the
  target-to-oligo process and separates software checks from laboratory
  evidence.
- [Getting started](getting-started.md) builds one synthetic gene-scale example.

## Use

- [Prepare a request](guides/prepare-a-request.md) covers one target, targets
  designed together, independent assembly groups, recovery primers, and order
  labels.
- [Inspect and verify](guides/inspect-and-verify.md) maps review questions to
  bundle files and optional BaseRender images.
- [Scale](guides/scale.md) explains resource limits and the tested software
  scenarios.

## Reference

- [Request contract](reference/request.md) lists fields, validation rules, and
  resource ceilings.
- [Artifacts, API, and errors](reference/artifacts-api-and-errors.md) specifies
  commands, Python calls, bundle contents, publication, and verification.
- [Method v1](reference/method-v1.md) gives the exact geometry, strand formulas,
  search objectives, and deliberate differences from the papers.
- [Sources and scope](reference/sources.md) identifies the primary literature,
  attribution, implementation independence, and unresolved validation gaps.

## Choose one operation

| Need | Command | Work performed |
| --- | --- | --- |
| Full design with a short, no-file summary | `uv run junction preflight <request>` | Runs the complete search; writes nothing. |
| Full design as JSON | `uv run junction plan <request> --format json` | Runs the complete search; writes nothing. |
| Publish a new, verified bundle | `uv run junction build <request> --output <new-directory>` | Runs the design, writes create-only files, then verifies the installed bundle by replay. |
| Check an existing bundle later | `uv run junction verify <bundle>` | Replays the recorded request and checks every file and identity. |

These commands are alternatives, not four stages of one efficient run.
`preflight`, `plan`, and `build` each repeat the complete design search.
