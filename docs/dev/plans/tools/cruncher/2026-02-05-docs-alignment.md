## Cruncher Docs Alignment Design (2026-02-05)

### Contents
- [Intent](#intent)
- [Scope](#scope)
- [Doc structure](#doc-structure)
- [Editing plan](#editing-plan)
- [Verification](#verification)

### Intent

Align Cruncher documentation with current source behavior while keeping the narrative concise and didactic. The docs should describe only active features, emphasize fixed-length PT workflows, and provide predictable, linkable navigation via a consistent Contents section.

### Scope

In scope:
- Cruncher docs under `src/dnadesign/cruncher/docs/` (demos, guides, reference, internals).
- Repo-level docs that reference Cruncher (README and `docs/`).
- Workspace configs under `src/dnadesign/cruncher/workspaces/` for demo accuracy.

Out of scope:
- Core code changes unrelated to documentation.
- Large refactors of analysis or optimizer internals.

### Doc structure

- **Index**: short overview plus links to demos, guides, and references.
- **Demos**: linear, progressive narratives that map directly to commands and artifacts.
- **Guides**: task-focused explanations of sampling, analysis, and ingestion.
- **References**: precise schema and CLI behavior.
- **Internals**: current behavior notes and a trimmed improvement backlog.

### Editing plan

- Add a `## Contents` section to each doc with jump links to headings.
- Remove outdated or deprecated feature callouts and describe only active behavior.
- Update demos to explain the fixed-length requirement and TFBS-core MMR behavior.
- Ensure example configs match the current schema.

### Verification

- Scan docs for deprecated keys or removed feature mentions.
- Run the two-TF demo commands from the docs to ensure they execute as written.
