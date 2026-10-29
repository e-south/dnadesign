<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../index.md)

# Prediction, scoring and selection

Each step in an OPAL campaign has a distinct job. Transforms prepare the data,
a predictive model estimates outcomes, an objective converts those estimates
into scores, and a selection rule chooses candidates. The configuration selects
a registered implementation for each step.

## Choose a component

- [Models](models/README.md): predict outcomes from candidate features.
- [Objectives](objectives/README.md): score predictions against your goal.
- [Selection rules](selection/README.md): rank candidates using scores and, where required, uncertainty.
- [Transforms](transforms/README.md): prepare features and measured outcomes.

Start with the [campaign walkthrough](../workflows/campaign-round.md) for a
complete random-forest example. These references explain how to change its
components and what a new implementation must provide.
