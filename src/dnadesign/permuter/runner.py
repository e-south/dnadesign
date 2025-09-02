"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/runner.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from dnadesign.permuter.selector import select  # <-- needed

_LOG = init_logger("INFO")


# ──────────────── small helpers kept here for testability ──────────────── #


def _load_refs(csv_file: Path) -> Dict[str, str]:
    """
    Load a minimal reference CSV with columns: ['ref_name','sequence'].
    """
    df = pd.read_csv(csv_file, dtype=str)
    if set(df.columns) != {"ref_name", "sequence"}:
        raise ValueError(
            f"Reference CSV must have columns ['ref_name','sequence'], got {list(df.columns)}"
        )
    if df["ref_name"].duplicated().any():
        raise ValueError("Duplicate ref_name entries in reference CSV")
    return dict(zip(df["ref_name"], df["sequence"]))


def _resolve_plot_names(cfg) -> List[str]:
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


def _read_jsonl(path: Path) -> List[Dict]:
    recs: List[Dict] = []
    with path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _load_existing(
    output_dir: Path,
) -> Optional[Tuple[List[Dict], List[Dict], Optional[str]]]:
    """
    If a MANIFEST exists, load all previous rounds (variants & elites) and
    return (all_variants, latest_elites, ref_sequence or None).
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


def _resolve_input_file(
    input_file: str | Path, *, config_path: Path, base_output: Path
) -> Path:
    """
    Resolve a job-specified path (refs.csv, lookup tables, etc.) with robust,
    predictable precedence:

      1) absolute path as-is (if exists)
      2) relative to the CONFIG file's directory (experiments/<name>/config.yaml)
      3) relative to base_output (the experiment directory)
      4) relative to the EXPERIMENTS ROOT (parent folder that contains all experiments)
      5) EXPERIMENTS ROOT / input / <basename>
      6) FINAL fallback: dnadesign/permuter/input/<basename>

    Notes
    -----
    - This lets you keep a single global refs.csv or codon table under:
        experiments/refs.csv                 (case 4)
        experiments/input/refs.csv          (case 5)
      while still supporting per-experiment inputs.
    """
    p = Path(input_file)
    exp_root = config_path.parent.parent  # experiments/<name>/ -> experiments/
    cands = []

    if p.is_absolute():
        cands.append(p)
    else:
        # local to the config.yaml
        cands.append(config_path.parent / p)
        # local to the experiment folder (base_output)
        cands.append(base_output / p)
        # at the experiments root
        cands.append(exp_root / p)
        # experiments root / input / <basename>
        cands.append(exp_root / "input" / p.name)
        # module fallback
        cands.append(Path(__file__).parent / "input" / p.name)

    for c in cands:
        c = c.expanduser().resolve()
        if c.exists():
            return c

    # Let caller raise a clean error message
    return p.expanduser().resolve()


def _canonical_region_for_seed(params: Dict[str, Any]) -> Tuple[int | None, int | None]:
    """
    Used only for deterministic seed derivation (not for bounds checking).
    Returns (start,end) where insertion index i is encoded as (i,i).
    """
    region = params.get("region")
    if isinstance(region, int):
        return region, region
    if isinstance(region, (list, tuple)) and len(region) == 2:
        try:
            s, e = int(region[0]), int(region[1])
            return s, e
        except Exception:
            return None, None
    return None, None


def _derive_base_seed(
    *,
    job_name: str,
    ref_name: str,
    protocol: str,
    round_idx: int,
    parent_var_id: str,
    params: Dict[str, Any],
) -> int:
    """
    Deterministically derive a 64-bit seed from stable knobs.
    """
    s, e = _canonical_region_for_seed(params)
    program = params.get("program", {}) or {}
    mismatch = (program.get("mismatch") or {}).get("rate", None)
    indel = (program.get("indel") or {}).get("rate", None)
    payload = {
        "job": job_name,
        "ref": ref_name,
        "protocol": protocol,
        "round": round_idx,
        "parent_var_id": parent_var_id,
        "region": [s, e] if s is not None else None,
        "gc_target": program.get("gc_target", None),
        "mismatch_rate": mismatch,
        "indel_rate": indel,
        "rng_seed": params.get("rng_seed", None),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


# ───────────────────────── public façade ───────────────────────── #


def run_config(config_path: Path, *, base_output: Path) -> None:
    """
    Run a single experiment config with outputs rooted at `base_output`.
    """
    validator = _Validator(str(config_path))
    exp_cfg, jobs_cfg = validator.experiment(), validator.jobs()
    _LOG.info(f"Experiment '{exp_cfg['name']}' | jobs: {len(jobs_cfg)}")

    for job in jobs_cfg:
        _run_job(job, base_output, config_path)

    _LOG.info("✔ Experiment complete")


# ───────────────────────── internal job runner ─────────────────── #


def _run_job(job: Dict, base_dir: Path, cfg_path: Path) -> None:
    job_name = job["name"]
    permute_cfg = job.get("permute", {})
    protocol = permute_cfg.get("protocol")
    metrics_cfg = job["evaluate"]["metrics"]
    metric_ids = [m["id"] for m in metrics_cfg]
    select_cfg = job["select"]
    run_mode = job.get("run", {}).get("mode", "full")

    refs_csv = _resolve_input_file(
        job["input_file"], config_path=cfg_path, base_output=base_dir
    )
    if not refs_csv.exists():
        raise FileNotFoundError(f"[{job_name}] Reference CSV not found at {refs_csv}")
    refs = _load_refs(refs_csv)

    _LOG.info(
        f"▶ Job {job_name} | refs: {len(job['references'])} | protocol: {protocol} | mode: {run_mode}"
    )

    plot_names = _resolve_plot_names(job.get("report", {}).get("plots"))
    report_csv = bool(job.get("report", {}).get("csv", False))

    for ref_name in job["references"]:
        if ref_name not in refs:
            raise ValueError(
                f"[{job_name}] reference '{ref_name}' not found in {job['input_file']}"
            )
        original_seq = refs[ref_name]
        output_dir = base_dir / "batch_results" / f"{job_name}_{ref_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        _LOG.info(f"[{job_name}/{ref_name}] output → {output_dir}")

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
                ref_sequence=ref_seq_from_manifest,
                metric_meta=metrics_cfg,  # for subtitle
            )
            _LOG.info(f"[{job_name}/{ref_name}] analysis complete")
            continue

        # round 1 seed (the original sequence)
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

        total_rounds = (
            int(job["iterate"]["total_rounds"])
            if job.get("iterate", {}).get("enabled", False)
            else 1
        )

        cfg_snapshot_path = output_dir / "config_snapshot.yaml"
        if not cfg_snapshot_path.exists():
            shutil.copy2(cfg_path, cfg_snapshot_path)

        for round_idx in range(1, total_rounds + 1):
            new_variants: List[Dict] = []

            for seed in seeds:
                # params are protocol-specific; inject derived seed deterministically
                base_params = permute_cfg.get("params", {}) or {}
                # deep-ish copy via json roundtrip to avoid mutating job config
                params = json.loads(json.dumps(base_params))
                params["_derived_seed"] = _derive_base_seed(
                    job_name=job_name,
                    ref_name=ref_name,
                    protocol=protocol,
                    round_idx=round_idx,
                    parent_var_id=seed["var_id"],
                    params=params,
                )

                # ─── protocol-specific path fixups / aliases ───────────────
                if protocol == "scan_codon":
                    # Accept multiple config shapes:
                    #   permute.params.codon_table: str
                    #   permute.codon_table(s): str | list[str]
                    #   permute.lookup_table(s): str | list[str]
                    codon_table_value = (
                        params.get("codon_table")
                        or permute_cfg.get("codon_table")
                        or permute_cfg.get("codon_tables")     # <— NEW supported alias
                        or permute_cfg.get("lookup_table")
                        or permute_cfg.get("lookup_tables")
                    )
                    if isinstance(codon_table_value, (list, tuple)):
                        codon_table_value = next(
                            (str(x) for x in codon_table_value if str(x).strip()), ""
                        )
                    if not codon_table_value:
                        raise ValueError(
                            f"[{job_name}] scan_codon requires params.codon_table (path to CSV)."
                        )
                    resolved_ct = _resolve_input_file(
                        codon_table_value, config_path=cfg_path, base_output=base_dir
                    )
                    if not resolved_ct.exists():
                        raise FileNotFoundError(
                            f"[{job_name}] codon_table not found at {resolved_ct}"
                        )
                    params["codon_table"] = str(resolved_ct)

                    # Optional: convert nt-based regions → codon units if clean
                    if "region_codons" not in params and permute_cfg.get("regions"):
                        regs = permute_cfg["regions"]
                        if isinstance(regs, list) and len(regs) in (0, 1):
                            if regs:
                                r = regs[0]
                                if (
                                    isinstance(r, (list, tuple))
                                    and len(r) == 2
                                    and all(isinstance(x, int) for x in r)
                                ):
                                    s, e = int(r[0]), int(r[1])
                                    if s % 3 == 0 and e % 3 == 0:
                                        params["region_codons"] = [s // 3, e // 3]
                                    else:
                                        _LOG.warning(
                                            f"[{job_name}/{ref_name}] regions [{s},{e}) not on codon boundaries; "
                                            f"ignoring region for scan_codon."
                                        )
                        elif isinstance(regs, list) and len(regs) > 1:
                            _LOG.warning(
                                f"[{job_name}/{ref_name}] multiple regions provided; "
                                f"scan_codon currently supports at most one contiguous region."
                            )

                # generate variants
                raw_iter = permute_record(seed, protocol, params=params)

                for var in raw_iter:
                    modifications = seed["modifications"] + var["modifications"]
                    seq = var["sequence"]
                    out = {
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
                    # propagate protocol-specific flat metadata if present
                    for k, v in var.items():
                        if k not in out:
                            out[k] = v
                    new_variants.append(out)

            _LOG.info(
                f"[{job_name}/{ref_name}] r{round_idx} permute → {len(new_variants)} variants"
            )

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
            metric_meta=metrics_cfg,
        )
        _LOG.info(f"[{job_name}/{ref_name}] done")

    _LOG.info(f"✔ Job '{job_name}' complete")
