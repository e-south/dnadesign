![Studies banner](assets/studies-banner.svg)

`studies` contains lightweight code for living project records that need more
than checked-in Markdown/YAML but should not become generic tool features. The
study records themselves live in [Study records](../../../docs/studies/README.md).

Ops owns the status/preflight API. This package only supplies the study-side
record loading, family adapters, and narrow study-specific helpers that Ops or a
repo-local skill can call explicitly.

Use this package when:
- you are adding or changing family-specific study snapshot or preflight logic
- you need to register a new study-owned status surface without editing OPS core
- a long-running study needs a small parser, linter, or catalog helper that is
  too specific for Cruncher, Construct, Folding, or another generic package

Do not use this package when:
- the change belongs to neutral OPS control-plane or observation-shell code
- the change only touches checked-in study records under `docs/studies/`
- the behavior should be a reusable top-level tool or package feature

Current adapter families:
- `promoter`: promoter-study snapshot and preflight adapters
- `cruncher`: command-centric Cruncher study snapshot and preflight adapters

Current study-specific helpers:
- `retron_hairpin_design`: Retron MSD label lint/compile helpers used through
  the repo-local retron hairpin skill or explicit module invocation, not a
  top-level CLI

See also:
- [Ops README](../ops/README.md)
- [Study records](../../../docs/studies/README.md)
