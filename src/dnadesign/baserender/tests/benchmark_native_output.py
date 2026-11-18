"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/benchmark_native_output.py

Measure native output stages and batch reuse without weakening fidelity checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

_OWNER = Path(__file__).resolve().parents[1]


def _digest(payload):
    return sha256(payload).hexdigest()


def _summary(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "maximum": max(values),
        "minimum": min(values),
    }


def _worker(mode, *, runs, count):
    imported = time.perf_counter()
    from PIL import Image

    from dnadesign.baserender import adapt_record, rasterize_svg, render_record_svg, validate_job
    from dnadesign.baserender.src.config import resolve_style
    from dnadesign.baserender.src.config.sequence_layout import sequence_layout_from_options
    from dnadesign.baserender.src.outputs import native_batch, native_svg, raster, svg

    import_seconds = time.perf_counter() - imported
    job = validate_job(_OWNER / "docs/examples/native_svg_job.yaml")
    rows = json.loads(job.input.path.read_text())
    records = [
        adapt_record(
            {**rows[index % len(rows)], "id": f"benchmark-{index}"},
            adapter_kind=job.input.adapter.kind,
            adapter_columns=job.input.adapter.columns,
        )
        for index in range(count)
    ]
    config = job.render.native_svg
    style = resolve_style(preset=job.render.style_preset, overrides=job.render.style_overrides)
    layout = sequence_layout_from_options(job.render.options)

    def measure():
        stages = {
            key: 0.0
            for key in ("preparation", "drawing_and_serialization", "serialization", "rasterization", "backend")
        }

        def timed(key, operation):
            def invoke(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return operation(*args, **kwargs)
                finally:
                    stages[key] += time.perf_counter() - start

            return invoke

        compositions = []
        with ExitStack() as stack:
            for name, operation, modules in (
                ("preparation", "prepare_native_svg", (native_svg, native_batch)),
                ("drawing_and_serialization", "draw_native_svg", (native_svg, native_batch)),
                ("serialization", "compose_svg", (svg,)),
                ("rasterization", "rasterize_svg", (raster, native_batch)),
                ("backend", "resolve_rasterizer", (raster,)),
            ):
                wrapped = timed(name, getattr(modules[0], operation))
                for module in modules:
                    stack.enter_context(patch.object(module, operation, wrapped))
            start = time.perf_counter()
            if mode == "individual":
                for record in records:
                    result = render_record_svg(
                        record,
                        style_preset=job.render.style_preset,
                        style_overrides=job.render.style_overrides,
                        layout=layout,
                        captions=config.captions,
                        illustrations=config.illustrations,
                    )
                    png = rasterize_svg(result, **config.rasterization.to_mapping())
                    compositions.append((result, png))
            else:
                backend = raster.resolve_rasterizer(config.rasterization)
                batch = native_batch.PreparedNativeBatch.prepare(records, config=config, style=style, layout=layout)
                compositions = [(result, png) for result, png, _ in batch.draw(rasterizer=backend)]
                batch.verify_sources()
            elapsed = time.perf_counter() - start
        stages["drawing"] = stages.pop("drawing_and_serialization") - stages["serialization"]
        stages["rasterization"] -= stages["backend"] if mode == "individual" else 0
        stages["other"] = elapsed - sum(stages.values())
        stages["total"] = elapsed
        identities = []
        for result, png in compositions:
            with Image.open(io.BytesIO(png.png)) as image:
                pixels = _digest(image.convert("RGBA").tobytes())
            identities.append(
                {
                    "svg_sha256": _digest(result.svg.encode()),
                    "png_pixels_sha256": pixels,
                    "width_px": png.width_px,
                    "height_px": png.height_px,
                    "rasterizer": png.rasterizer.to_mapping(portable=True),
                    "diagnostics": result.diagnostics.to_mapping(portable=True),
                }
            )
        return {"stages_seconds": stages, "outputs": identities}

    samples = [measure() for _ in range(runs)]
    unit = 1 if sys.platform == "darwin" else 1024
    return {
        "mode": mode,
        "records": count,
        "runs": runs,
        "import_seconds": import_seconds,
        "source_data_sha256": _digest(job.input.path.read_bytes()),
        "style_sha256": _digest(json.dumps(asdict(style), sort_keys=True).encode()),
        "native_settings": config.to_mapping(portable=True),
        "layout": layout.to_mapping(),
        "samples": samples,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit,
        "raster_child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cold-runs", type=int, default=5)
    parser.add_argument("--warm-runs", type=int, default=9)
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--worker", choices=("individual", "batch"), help=argparse.SUPPRESS)
    parser.add_argument("--runs", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.cold_runs, args.warm_runs, args.records, args.runs) < 1:
        parser.error("run and record counts must be positive")
    if args.worker:
        print(json.dumps(_worker(args.worker, runs=args.runs, count=args.records)))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    measurements = []
    schedule = [
        (mode, "cold", trial, 1)
        for trial in range(args.cold_runs)
        for mode in (("individual", "batch") if trial % 2 == 0 else ("batch", "individual"))
    ]
    schedule += [(mode, "warm", 0, args.warm_runs + 1) for mode in ("batch", "individual")]
    for mode, kind, trial, runs in schedule:
        started = time.perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--runs",
                str(runs),
                "--records",
                str(args.records),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (args.output_dir / f"{mode}-{kind}-{trial}.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"benchmark worker failed; inspect {mode}-{kind}-{trial}.stderr")
        row = {**json.loads(result.stdout), "kind": kind, "process_seconds": time.perf_counter() - started}
        measurements.append(row)
        with (args.output_dir / "measurements.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"{mode} {kind} trial {trial + 1} finished", file=sys.stderr, flush=True)
    identity = measurements[0]["samples"][0]["outputs"]
    if any(sample["outputs"] != identity for row in measurements for sample in row["samples"]):
        raise RuntimeError("native benchmark changed output pixels, SVG or diagnostics")
    reports = {}
    for mode in ("individual", "batch"):
        for kind in ("cold", "warm"):
            rows = [row for row in measurements if row["mode"] == mode and row["kind"] == kind]
            samples = [sample for row in rows for sample in row["samples"][(1 if kind == "warm" else 0) :]]
            reports[f"{mode}_{kind}"] = {
                "samples": len(samples),
                "stages_seconds": {
                    key: _summary([sample["stages_seconds"][key] for sample in samples])
                    for key in samples[0]["stages_seconds"]
                },
                "process_seconds": _summary([row["process_seconds"] for row in rows]),
                "process_peak_rss_bytes": max(row["process_peak_rss_bytes"] for row in rows),
                "raster_child_peak_rss_bytes": max(row["raster_child_peak_rss_bytes"] for row in rows),
            }
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "distribution_version": version("dnadesign-tools"),
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "records_per_run": args.records,
        "source_data_sha256": measurements[0]["source_data_sha256"],
        "style_sha256": measurements[0]["style_sha256"],
        "native_settings": measurements[0]["native_settings"],
        "layout": measurements[0]["layout"],
        "benchmark_sha256": _digest(Path(__file__).read_bytes()),
        "outputs_identical": True,
        "outputs": identity,
        "results": reports,
        "limits": [
            "Cold means a new Python process, not a cold OS/filesystem cache.",
            "Warm excludes the first in-process sample; p95 is the nearest-rank observation.",
            "Timing includes scoped instrumentation and raster subprocess wait, not pixel-hash verification.",
            "RSS includes identity verification; parent/largest raster child are separate, not aggregate memory.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
