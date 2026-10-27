---
doc_id: junction-docs
title: junction documentation index
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-11
---

# Documentation index

Start with the tutorial for a complete run. Use the guides for individual
tasks and the reference pages for exact fields, limits, and failure rules.

## Use `junction`

| Task | Page |
| --- | --- |
| Build, inspect, and verify the checked-in example | [Run the gene-scale example](getting-started.md) |
| Turn raw, text, or FASTA sequences into a request | [Prepare a request](guides/prepare-a-request.md) |
| Trace an existing bundle and its checks | [Inspect and verify](guides/inspect-and-verify.md) |
| Compare gene fragments with an oligo-pool route | [Choose a synthesis route](guides/choose-a-synthesis-route.md) |
| Bound a larger request | [Scale](guides/scale.md) |

## Understand the method

| Question | Page |
| --- | --- |
| What happens from target sequence to planned oligos? | [Method overview](explanation/how-junction-works.md) |
| Which geometry, scores, and tie rules does v1 use? | [Method v1](reference/method-v1.md) |
| Which papers inspired the tool, and what remains untested? | [Sources and scope](reference/sources.md) |

## Reference

| Need | Page |
| --- | --- |
| Request fields and limits | [Request contract](reference/request.md) |
| CLI, Python API, bundle layout, and errors | [Artifacts, API, and errors](reference/artifacts-api-and-errors.md) |
| BaseRender molecular views | [BaseRender integration](../../baserender/docs/integrations/junction.md) |

## Commands

| Command | Result |
| --- | --- |
| `uv run junction request --base-request <request> --input <file> --primer-binding-length <nt>` | Request JSON; the planner is not run. |
| `uv run junction preflight <request>` | Short summary from the complete design search; no files are written. |
| `uv run junction plan <request> --format json` | Complete plan JSON; no files are written. |
| `uv run junction build <request> --output <new-directory>` | New bundle, followed by installed-bundle replay. |
| `uv run junction verify <bundle>` | Offline replay and exact inventory verification. |

`preflight`, `plan`, and `build` each run the complete search. Choose the one
result you need rather than treating them as sequential stages.
