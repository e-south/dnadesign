---
doc_id: public-private-study-boundary
surface: study-architecture
owner: dnadesign-maintainers
last_verified: 2026-08-08
---

## Public and private study boundary

`dnadesign` is a public software repository. Its generic tools, shared
contracts, test fixtures, and sanitized examples belong here. A live study does
not belong here merely because it uses those tools.

### Place each concern with its owner

| Concern | Owner |
| --- | --- |
| Reusable algorithms, schemas, and public tool APIs | `dnadesign` |
| Neutral study-record loading and validation | `dnadesign.studies.core` |
| Small synthetic or intentionally public examples | this repository |
| Unpublished sequences, measurements, objectives, decisions, and campaign state | a private study workspace |
| Generated model outputs, plots, notebooks, and handoff bundles | the study workspace that produced them |

The private workspace should be a sibling repository or another explicit root,
not a hidden subtree inside the public checkout. It depends on a pinned
dnadesign version and calls public APIs or CLIs. dnadesign must not discover it
through machine-specific paths or import its study modules.

`dnadesign.studies.core` already loads `docs/studies/index.yaml` and
`operations/ops.study.yaml` from an explicit repository root. The external
repository therefore keeps the same record contract without becoming a
dnadesign source subtree.

### External workspace contract

A private study workspace should declare:

1. its own repository and access policy;
2. a pinned dnadesign revision or release;
3. one explicit study root;
4. typed inputs and outputs at each tool handoff; and
5. private CI that runs its routes against that pinned dependency.

Cross-repository bridges may route an agent to the owning workspace. They must
not copy formulas, sequences, measurements, or campaign state into dnadesign.

### What may stay checked in

A concrete study may remain here only when maintainers intend every tracked
record to be public and the study is useful as maintained product evidence.
Keep generated outputs ignored. Prefer a synthetic fixture when the same
contract can be tested without scientific records.

Before adding or refreshing a public study, review at least:

- sequences and sample identifiers;
- filenames, absolute paths, and document metadata;
- unpublished objectives, rankings, and campaign decisions;
- raw or derived measurements; and
- generated notebooks, plots, model artifacts, and order files.

### Existing records

The current checked-in study trees were already published. A later extraction
can improve the package boundary but cannot make those commits private. If an
existing record should not have been public, treat it as disclosed, decide
whether history rewriting is warranted, and account for clones, forks, caches,
and downstream copies separately.

### Migration rule

Do not relocate a live study until its private destination and owner are named.
Then move the study record, implementation, tests, and generated workspace as
one migration; replace internal imports with public dnadesign contracts; and
remove the old public unit without a compatibility shim. Keep only sanitized
fixtures needed to test the public seam.
