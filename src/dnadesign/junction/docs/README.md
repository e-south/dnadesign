---
doc_id: junction-docs
title: junction documentation
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Start here

Choose the path that matches what you need to do next.

[![Input target, fragment oligos, pre-ligation junctions, and the expected PCR duplex](assets/assembly-process.svg)](assets/assembly-process.svg)

Give `junction` exact target sequences. It returns vendor-neutral oligo orders,
FASTA files for downstream sequence tools, machine-readable checks, and a
bundle that can be replayed. The figures and sequence exports come from the
same plan.

## Choose a path

| Need | Go to |
| --- | --- |
| Build and verify the example | [Getting started](getting-started.md) |
| Understand the molecular sequence | [How `junction` works](explanation/how-junction-works.md) |
| Prepare one target or a jointly designed set | [Prepare a request](guides/prepare-a-request.md) |
| Trace an existing bundle | [Inspect and verify](guides/inspect-and-verify.md) |
| Plan a larger request safely | [Scale](guides/scale.md) |
| Check every input field and limit | [Request contract](reference/request.md) |
| Use the CLI or Python API | [Artifacts, API, and errors](reference/artifacts-api-and-errors.md) |
| Inspect exact formulas and search rules | [Method v1](reference/method-v1.md) |
| Read the source and validation boundaries | [Sources and scope](reference/sources.md) |

## Commands

| Need | Command | Work performed |
| --- | --- | --- |
| Turn raw, text, or FASTA sequences into request JSON | `uv run junction request --base-request <request> --input <file> --primer-binding-length <nt>` | Replaces the base request's targets while preserving its reviewed design policy; it does not run the design. |
| Full design with a short, no-file summary | `uv run junction preflight <request>` | Runs the complete search; writes nothing. |
| Full design as JSON | `uv run junction plan <request> --format json` | Runs the complete search; writes nothing. |
| Publish a new, verified bundle | `uv run junction build <request> --output <new-directory>` | Runs the design, writes create-only files, then verifies the installed bundle by replay. |
| Check an existing bundle later | `uv run junction verify <bundle>` | Replays the recorded request and checks every file and identity. |

`request` prepares canonical input. The other commands consume that same
request contract. `preflight`, `plan`, and `build` each repeat the complete
design search, so choose the one result you need.

The checked-in [three-fragment request](../examples/three-fragment-review/request.yaml)
and [BaseRender jobs](../examples/three-fragment-review/jobs/) generate the
molecular figures used throughout these docs. Junction's deterministic SVG API
generates the optional sequence-comparison figure.
