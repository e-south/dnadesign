---
id: opal-campaign-secg-rmf-greedy
title: SECG RMF greedy campaign
owner: stress_ethanol_cipro_growth
status: inactive
last_verified: 2026-07-13
surface: opal_campaign
campaign_slug: secg_rmf_greedy
campaign_kind: study
runtime_status: blocked_on_label_promotion
---

## SECG RMF Greedy Campaign

**Owner:** stress_ethanol_cipro_growth study
**Lifecycle:** configured, inactive
**Last verified:** 2026-07-13

This study campaign fits one shared eight-output response model and evaluates
three RMF selection views: ethanol, ciprofloxacin, and AND. Each view nominates
six candidates; `selection_batch` is their sequence-deduplicated logical union.

The configuration validates against the study candidate table, but execution
requires the typed Reader response-window sidecar declared in
`configs/campaign.yaml`. Its absence is an activation gate, not an alternate
label-source route. Do not initialize, ingest, or run this campaign until the
[stress-study OPAL route](../../../../../docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md)
records that promotion.

SFXI round-0 runs remain study evidence in their declared y-space. They are not
executable routes into this campaign.
