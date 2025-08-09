"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/main.py

Entry-point CLI: `python -m dnadesign.permuter.main --config myfile.yaml`

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dnadesign.permuter.config import _Validator
from dnadesign.permuter.evaluate import evaluate_many
from dnadesign.permuter.logging_utils import init_logger
from dnadesign.permuter.permute_record import permute_record
from dnadesign.permuter.reporter import (
    ensure_manifest_round,
    write_results,
    write_round_files,
)
from dnadesign.permuter.selector import select

_LOG = init_logger("INFO")


def _load_refs(csv_file: Path) -> Dict[str, str]:
    """Load a two-column CSV of reference names and sequences."""
    if not csv_file.is_file():
        fallback = Path(__file__).parent / "input" / csv_file.name
        if fallback.is_file():
            csv_file = fallback
        else:
            raise FileNotFoundError(
                f"Reference CSV not found at {csv_file!r} or {fallback!r}"
            )
    df = pd.read_csv(csv_file, dtype=str)
    if set(df.columns) != {"ref_name", "sequence"}:
        raise ValueError(
            f"Reference CSV must have columns ['ref_name','sequence'], got {list(df.columns)}"
        )
    if df["ref_name"].duplicated().any():
        raise ValueError("Duplicate ref_name entries in reference CSV")
    return dict(zip(df["ref_name"], df["sequence"]))


def _resolve_plot_names(cfg) -> List[str]:
    """Interpret the `report.plots` field in the YAML."""
    if cfg is True:
        return ["scatter_metric_by_position"]
    if cfg in (False, None):
        return []
    if isinstance(cfg, list):
        return cfg
    raise ValueError("report.plots must be true/false or a list of plot names")


def _det_var_id(
    *,
    job_name: str,
    ref_name: str,
    protocol: str,
    round_idx: int,
    sequence: str,
    modifications: List[str],
) -> str:
    """Deterministic, short, URL-safe ID derived from the variant's context."""
    payload = {
        "job": job_name,
        "ref": ref_name,
        "protocol": protocol,
        "round": round_idx,
        "sequence": sequence,
        "modifications": modifications,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=16).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=").upper()[:12]


# ───────────────────────── analysis/plots-only helpers ───────────────────────── #


def _read_jsonl(path: Path) -> List[Dict]:
    recs: List[Dict] = []
    with path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _load_existing(
    output_dir: Path,
) -> Optional[Tuple[List[Dict], List[Dict], Optional[str]]]:
    """
    Returns (all_variants, elites_latest, ref_sequence) if manifest + round files are present,
    otherwise None.
    """
    mpath = output_dir / "MANIFEST.json"
    if not mpath.exists():
        return None
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return None

    rounds = sorted(manifest.get("rounds", []), key=lambda r: r.get("round", 0))
    if not rounds:
        return None

    all_variants: List[Dict] = []
    elites_latest: List[Dict] = []
    for r in rounds:
        vpath = output_dir / r.get("variants", "")
        epath = output_dir / r.get("elites", "")
        if not (vpath.exists() and epath.exists()):
            return None
        all_variants.extend(_read_jsonl(vpath))
        elites_latest = _read_jsonl(epath)

    ref_sequence = manifest.get("ref_sequence")
    return all_variants, elites_latest, ref_sequence


def _strategy_summary(select_cfg: Dict) -> str:
    strat = select_cfg.get("strategy", {})
    stype = str(strat.get("type", ""))
    if stype == "top_k":
        ties = "yes" if strat.get("include_ties", True) else "no"
        return f"top_k(k={strat.get('k')}, ties={ties})"
    if stype == "threshold":
        target = strat.get("target", "objective")
        if "percentile" in strat:
            if target == "objective":
                return f"threshold(objective ≥ p{strat['percentile']})"
            return (
                f"threshold(metric={strat.get('metric_id')}, "
                f"norm={'T' if strat.get('use_normalized', True) else 'F'}, "
                f"≥ p{strat['percentile']})"
            )
        # numeric threshold
        if target == "objective":
            return f"threshold(objective ≥ {strat.get('threshold')})"
        return (
            f"threshold(metric={strat.get('metric_id')}, "
            f"norm={'T' if strat.get('use_normalized', True) else 'F'}, "
            f"≥ {strat.get('threshold')})"
        )
    return stype or "strategy"


# ───────────────────────── job runner ───────────────────────── #


def _run_job(job: Dict, base_dir: Path, cfg_path: Path) -> None:
    """
    Execute a single job block from the config.
    """
    job_name = job["name"]
    permute_cfg = job.get("permute", {})
    protocol = permute_cfg.get("protocol")
    metrics_cfg = job["evaluate"]["metrics"]
    metric_ids = [m["id"] for m in metrics_cfg]
    select_cfg = job["select"]
    run_mode = job.get("run", {}).get("mode", "full")

    # 1) Load & validate references
    refs = _load_refs(Path(job["input_file"]).expanduser())

    _LOG.info(
        f"▶ Job {job_name} | refs: {len(job['references'])} | protocol: {protocol} | mode: {run_mode}"
    )

    # Which plots to generate / CSV?
    plot_names = _resolve_plot_names(job.get("report", {}).get("plots"))
    report_csv = bool(job.get("report", {}).get("csv", False))

    # 2) Loop over each reference sequence
    for ref_name in job["references"]:
        if ref_name not in refs:
            raise ValueError(
                f"[{job_name}] reference '{ref_name}' not found in {job['input_file']}"
            )
        original_seq = refs[ref_name]
        output_dir = base_dir / "batch_results" / f"{job_name}_{ref_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        _LOG.info(f"[{job_name}/{ref_name}] output → {output_dir}")

        # Analysis/auto: plots-only if data exists
        existing = _load_existing(output_dir)
        if run_mode == "analysis" or (run_mode == "auto" and existing):
            if not existing:
                raise RuntimeError(
                    f"[{job_name}/{ref_name}] analysis mode requested but no existing data at {output_dir}"
                )
            all_variants, elites, ref_seq_from_manifest = existing
            _LOG.info(
                f"[{job_name}/{ref_name}] analysis: rendering plots (requested={plot_names or 'none'})"
            )
            write_results(
                variants=all_variants,
                elites=elites,
                output_dir=output_dir,
                job_name=job_name,
                ref_name=ref_name,
                plot_names=plot_names,
                write_csv=report_csv,
                ref_sequence=ref_seq_from_manifest,  # may be None for legacy runs
            )
            _LOG.info(f"[{job_name}/{ref_name}] analysis complete")
            continue

        # a) initialize seeds with unmodified reference
        seeds: List[Dict] = [
            {
                "var_id": _det_var_id(
                    job_name=job_name,
                    ref_name=ref_name,
                    protocol=protocol,
                    round_idx=1,
                    sequence=original_seq,
                    modifications=[],
                ),
                "job_name": job_name,
                "ref_name": ref_name,
                "protocol": protocol,
                "sequence": original_seq,
                "modifications": [],
                "round": 1,
                "parent_var_id": "",
            }
        ]
        all_variants: List[Dict] = []

        # determine rounds
        total_rounds = (
            int(job["iterate"]["total_rounds"])
            if job.get("iterate", {}).get("enabled", False)
            else 1
        )

        # snapshot config once per job/ref
        cfg_snapshot_path = output_dir / "config_snapshot.yaml"
        if not cfg_snapshot_path.exists():
            shutil.copy2(cfg_path, cfg_snapshot_path)

        # b) iterate
        for round_idx in range(1, total_rounds + 1):
            new_variants: List[Dict] = []

            # i) permute seeds
            for seed in seeds:
                raw = permute_record(
                    seed,
                    protocol,
                    params=permute_cfg.get("params", {}),
                    regions=permute_cfg.get("regions", []),
                    lookup_tables=permute_cfg.get("lookup_tables", []),
                )
                for var in raw:
                    modifications = seed["modifications"] + var["modifications"]
                    seq = var["sequence"]
                    new_variants.append(
                        {
                            "job_name": job_name,
                            "ref_name": ref_name,
                            "protocol": protocol,
                            "sequence": seq,
                            "modifications": modifications,
                            "round": round_idx,
                            "parent_var_id": seed["var_id"],
                            "var_id": _det_var_id(
                                job_name=job_name,
                                ref_name=ref_name,
                                protocol=protocol,
                                round_idx=round_idx,
                                sequence=seq,
                                modifications=modifications,
                            ),
                        }
                    )

            _LOG.info(
                f"[{job_name}/{ref_name}] r{round_idx} permute → {len(new_variants)} variants"
            )

            # ii) evaluate metrics
            sequences = [v["sequence"] for v in new_variants]
            metric_to_scores = evaluate_many(
                sequences, metrics_cfg=metrics_cfg, ref_sequence=original_seq
            )
            for v in new_variants:
                v["metrics"] = {}
            for mid, scores in metric_to_scores.items():
                for v, s in zip(new_variants, scores):
                    v["metrics"][mid] = s

            _LOG.info(
                f"[{job_name}/{ref_name}] r{round_idx} evaluate → metrics: {', '.join(metric_ids)}"
            )

            # iii) objective + selection
            job_ctx = {
                "job_name": job_name,
                "ref_name": ref_name,
                "round": round_idx,
                "output_dir": str(output_dir),
            }
            seeds = select(
                new_variants,
                metrics_cfg=metrics_cfg,
                select_cfg=select_cfg,
                job_ctx=job_ctx,
            )
            _LOG.info(
                f"[{job_name}/{ref_name}] r{round_idx} select → elites: {len(seeds)}"
            )

            # Persist round files + manifest (now records ref_sequence)
            ensure_manifest_round(
                output_dir, job_name, ref_name, round_idx, ref_sequence=original_seq
            )
            write_round_files(
                variants=new_variants,
                elites=seeds,
                output_dir=output_dir,
                job_name=job_name,
                ref_name=ref_name,
                round_idx=round_idx,
            )

            all_variants.extend(new_variants)

        # 3) plots & optional CSV
        _LOG.info(
            f"[{job_name}/{ref_name}] finalize → plots: {plot_names or 'none'} | csv: {report_csv}"
        )
        write_results(
            variants=all_variants,
            elites=seeds,
            output_dir=output_dir,
            job_name=job_name,
            ref_name=ref_name,
            plot_names=plot_names,
            write_csv=report_csv,
            ref_sequence=original_seq,
        )
        _LOG.info(f"[{job_name}/{ref_name}] done")

    _LOG.info(f"✔ Job '{job_name}' complete")


def main() -> None:
    """Parse CLI, load & validate config, then run each job in sequence."""
    default_cfg = Path(__file__).parent / "config.yaml"
    parser = argparse.ArgumentParser(description="Run the permuter pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"YAML configuration file (default: {default_cfg.name})",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        _LOG.error(f"Config file not found: {args.config}")
        sys.exit(1)

    validator = _Validator(str(args.config))
    exp_cfg, jobs_cfg = validator.experiment(), validator.jobs()
    _LOG.info(f"Experiment '{exp_cfg['name']}' | jobs: {len(jobs_cfg)}")

    base_dir = Path(__file__).parent
    for job in jobs_cfg:
        _run_job(job, base_dir, args.config)

    _LOG.info("✔ All jobs completed successfully.")


if __name__ == "__main__":
    main()
