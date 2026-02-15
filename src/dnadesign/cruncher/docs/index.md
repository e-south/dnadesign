# Cruncher docs


## Contents
- [Overview](#overview)
- [Common provenance flow](#common-provenance-flow)
- [Demos (end-to-end)](#demos-end-to-end)
- [Guides (task-focused)](#guides-taskfocused)
- [References (how things work)](#references-how-things-work)
- [For developers](#for-developers)

## Overview

Cruncher designs short, fixed-length DNA sequences that score highly across one or more TF PWMs, using Gibbs annealing and TFBS-core MMR to select a diverse elite set. Use the demos for end-to-end workflows and the references for precise schema/CLI behavior.

**Intent (at a glance)**

- **What it is:** an optimization engine for designing **short, fixed-length DNA** sequences that jointly satisfy one or more TF PWMs, then returning a **diverse elite set**.
- **When to use:** multi-TF promoter/operator design under tight length constraints; motif-compatibility tradeoff exploration; producing a small, diverse candidate set for assays; campaign sweeps across many regulator sets + aggregate comparison.
- **What it is not:** a posterior-inference engine (don't interpret traces as posterior samples); a variable-length designer; a motif discovery tool (use MEME/STREME for discovery, then ingest/lock).
- **Mental model:** deterministic data prep (`fetch`/`lock`) + strict Gibbs annealing optimization (`sample`) + artifact-native analytics (`analyze`).

Start with [Intent + lifecycle](guides/intent_and_lifecycle.md) if you're new to Cruncher.

## Common provenance flow

All curated demos follow this contract:

1. fetch TFBS from configured sources per TF
2. merge site sets per TF (`catalog.combine_sites=true`)
3. discover motifs with MEME OOPS into a demo-specific discovered source
4. lock/parse/sample/analyze against discovered motifs (`catalog.pwm_source=matrix`)
5. export sequence tables for downstream wrappers (`cruncher export sequences`)

This keeps optimizer, analysis, baserender showcase, and DenseGen exports on the same motif provenance path.

## Demos (end-to-end)

Each demo maps to a workspace: `demo_basics_two_tf`, `demo_campaigns_multi_tf`,
`densegen_prep_three_tf`.

- [Two-TF demo (end-to-end)](demos/demo_basics_two_tf.md)
- [Campaign demo (multi-TF)](demos/demo_campaigns_multi_tf.md)
- [Densegen prep demo (three-TF)](demos/demo_densegen_prep_three_tf.md)

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
