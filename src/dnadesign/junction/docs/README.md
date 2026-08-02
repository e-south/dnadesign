# `junction` documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

`junction` converts a complete design request into a deterministic oligo plan.
The request must supply exact linear DNA targets, their joint-design boundary,
recovery primers, search policy, and order metadata. The tool does not infer
those scientific choices from a FASTA file or a bare sequence.

## Learn

- [How `junction` works](explanation/how-junction-works.md) explains the
  target-to-oligo process, maps paper terms to software terms, and separates
  string checks from laboratory evidence.
- [Getting started](getting-started.md) builds one small synthetic example.

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
| Publish a new, internally verified bundle | `uv run junction build <request> --output <new-directory>` | Runs the design, publishes create-only files, then repeats the design while verifying the installed bundle by replay. |
| Check an existing bundle later | `uv run junction verify <bundle>` | Replays the recorded request and checks every file and identity. |

These commands are alternatives, not four stages of one efficient run.
`preflight`, `plan`, and `build` each repeat the complete design search.
