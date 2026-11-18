---
doc_id: baserender-native-performance
title: Native output performance qualification
owner: dnadesign-maintainers
status: active
last_verified: 2026-11-18
---

# Native output performance

Measure the [native composition path](vector-output.md) before adding preparation
caches or a persistent rasterizer. The qualified batch path resolves Inkscape
once, prepares all records before drawing, and rechecks source identities before
publication. Video holds reuse composed frames. None of these paths needs a
global cache to preserve its current throughput.

## Repeat the measurement

From the repository root, with the existing environment, Arial and Inkscape:

```sh
uv run --locked python -B \
  src/dnadesign/baserender/tests/benchmark_native_output.py \
  --output-dir src/dnadesign/baserender/outputs/native-performance-review \
  --cold-runs 5 --warm-runs 9 --records 4
```

Use a fresh output directory. The benchmark selects the existing neutral
60/100-base job fixture. It compares repeated public API calls with the batch
implementation used by jobs, on the same canvas, fonts, captions and DPI.
It does not measure scientific selection, job input capture or publication I/O.
Runtime/source versions, input/style hashes, resolved font/artwork/backend
identities, measurements and exact output identities are retained in the report.

Cold observations use separate Python processes; the operating-system cache is
not flushed. Cold trial order alternates the two paths. Warm observations exclude
one initial in-process rendering; p95 is the nearest-rank observation, not a
population estimate. Scoped timing wrappers separate preparation, drawing,
serialization, backend resolution and raster conversion. Pixel verification is
outside those timers. Whole-process wall time includes import and verification.
RSS includes verification and reports parent/maximum raster child separately,
rather than claiming an aggregate simultaneous peak.

Every run must retain identical SVG bytes, decoded PNG pixels and diagnostics.
A mismatch fails the benchmark. Missing tools or a failed worker also fail;
there is no raster fallback or silently omitted sample.

## Measured baseline and decision

The retained qualification is `outputs/native-performance-20261118-02/` under
BaseRender, on source revision `1b4427491`, macOS arm64, Python 3.12.11 and
Inkscape 1.4.3. The earlier `-01` run provides an independent order comparison.
Each table entry is seconds per four records, median / observed p95:

| Path | Cold rendering | Warm rendering | Warm preparation | Warm backend resolution |
| --- | --- | --- | --- | --- |
| Repeated public calls | 3.927 / 4.428 | 3.663 / 5.919 | 0.125 / 0.150 | 0.784 / 1.285 |
| Job batch preparation | 3.298 / 5.222 | 3.118 / 3.199 | 0.121 / 0.131 | 0.195 / 0.208 |

Warm batch median improves by about 15%; the earlier run measured about 14%.
Backend reuse accounts for the saving. Preparation is about 4% of batch time;
raster conversion is about 62%. Serialization is about 2%. Parent peak RSS was
132.2 MiB and the largest raster child 146.2 MiB. These are local observations,
not portable performance guarantees. External raster time had cold outliers;
the data do not establish a universal tail-latency improvement.

No further cache is justified by these measurements. Keep hash/font rechecks
and batch preflight. If larger workloads miss the budgets below, profile that
workload before introducing backend lifetime, cache invalidation or concurrency
state. Reusing backend admission is already qualified by the native job tests;
held video frames do not repeat rendering.

## Final source qualification

The repeat on `caa77a11b` is retained at
`outputs/native-performance-final-20261118-01/` under BaseRender, using the same
fixture and benchmark hash. Five cold and nine warm observations per path retain
exactly the earlier SVG bytes, PNG pixels and diagnostics. Warm batch median /
observed p95 is 2.754 / 2.805 s; cold is 2.948 / 2.981 s. Warm preparation is
0.107 / 0.112 s. Parent / largest raster-child peaks are 133.3 / 143.6 MiB. All
local budgets below pass. The comparable individual warm median is 3.262 s,
so batch admission saves about 16% relative to individual calls.
This repeat supports the existing batch admission reuse; it does not justify
additional caches or a portable latency claim. Raw observations and the explicit
budget comparison remain in the generated report and `qualification.json`.

## Local regression budgets

For this exact four-record fixture and host class, review changes against:

| Criterion | Budget |
| --- | --- |
| Warm batch total | median ≤ 4.0 s; observed p95 ≤ 4.5 s |
| Cold batch rendering, excluding import/verification | median ≤ 4.5 s; observed p95 ≤ 7.0 s |
| Warm batch preparation | median ≤ 0.20 s; observed p95 ≤ 0.25 s |
| Parent / largest raster child peak RSS | ≤ 160 / 190 MiB, separately |
| Fidelity | identical SVG bytes, PNG pixels and placement diagnostics |

Compare the generated JSON with these budgets during performance qualification.
They are not wall-clock pytest assertions or a cross-machine CI gate. A changed
host, font, backend, input or physical canvas requires a new comparable baseline;
do not hide an overrun by reducing font size, DPI or verification. Keep failed
and outlying observations when interpreting a repeat.
