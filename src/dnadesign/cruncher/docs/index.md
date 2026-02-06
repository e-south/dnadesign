# Cruncher docs


## Contents
- [Overview](#overview)
- [Demos (end-to-end)](#demos-end-to-end)
- [Guides (task-focused)](#guides-taskfocused)
- [References (how things work)](#references-how-things-work)
- [For developers](#for-developers)

## Overview

Cruncher designs short, fixed-length DNA sequences that score highly across one or more TF PWMs, using parallel tempering MCMC and TFBS-core MMR to select a diverse elite set. Use the demos for end-to-end workflows and the references for precise schema/CLI behavior.

## Demos (end-to-end)

Each demo maps to a workspace: `demo_basics_two_tf`, `demo_campaigns_multi_tf`.

- [Two-TF demo (end-to-end)](demos/demo_basics_two_tf.md)
- [Campaign demo (multi-TF)](demos/demo_campaigns_multi_tf.md)

## Guides (task-focused)

- [Ingesting and caching data](guides/ingestion.md)
- [MEME Suite setup](guides/meme_suite.md)
- [Intent + lifecycle](guides/intent_and_lifecycle.md)
- [Sampling + analysis](guides/sampling_and_analysis.md)

## References (how things work)

- [CLI reference](reference/cli.md)
- [Config reference](reference/config.md)
- [Architecture and artifacts](reference/architecture.md)

## For developers

- [Package spec](internals/spec.md)
- [Optimizer improvements plan](internals/optimizer_improvements_plan.md)
