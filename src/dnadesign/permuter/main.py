"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/main.py

Entry-point CLI: `python -m permuter.main --config myfile.yaml`

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from dnadesign.permuter.config import _Validator
from dnadesign.permuter.evaluators import get_evaluator
from dnadesign.permuter.logging_utils import init_logger
from dnadesign.permuter.permute_record import permute_record
from dnadesign.permuter.reporter import write_results
from dnadesign.permuter.selector import select

_LOG = init_logger("INFO")


def _load_refs(csv_file: Path) -> Dict[str, str]:
    """
    Load a two-column CSV of reference names and sequences.
    """
    if not csv_file.is_file():
        # fallback lookup in package input directory
        fallback = Path(__file__).parent / "input" / csv_file.name
        if fallback.is_file():
            csv_file = fallback
        else:
            raise FileNotFoundError(
                f"Reference CSV not found at {csv_file!r} or {fallback!r}"
            )
    df = pd.read_csv(csv_file, dtype=str)
    # ensure exactly the two expected columns
    if set(df.columns) != {"ref_name", "sequence"}:
        raise ValueError(
            f"Reference CSV must have columns ['ref_name','sequence'], got {list(df.columns)}"
        )
    # no duplicate names allowed
    if df["ref_name"].duplicated().any():
        raise ValueError("Duplicate ref_name entries in reference CSV")
    return dict(zip(df["ref_name"], df["sequence"]))


def _resolve_plot_names(cfg) -> List[str]:
    """
    Interpret the `report.plots` field in the YAML.
    """
    if cfg is True:
        return ["scatter_metric_by_position"]
    if cfg in (False, None):
        return []
    if isinstance(cfg, list):
        return cfg
    raise ValueError("report.plots must be true/false or a list of plot names")


def _run_job(job: Dict, base_dir: Path, cfg_path: Path) -> None:
    """
    Execute a single job block from the config:
      1) Load reference sequences
      2) For each reference:
         a) Generate initial seed
         b) Iterate permute→evaluate→select for N rounds
         c) Collect all variants and final elites
      3) Write CSV outputs and snapshot config
      4) Generate requested plots
    """
    job_name = job["name"]
    _LOG.info(f"▶ Starting job '{job_name}'")

    # 1) Load & validate references
    refs = _load_refs(Path(job["input_file"]).expanduser())
    for r in job["references"]:
        if r not in refs:
            raise ValueError(
                f"[{job_name}] reference '{r}' not found in {job['input_file']}"
            )

    # Which plots to generate?
    plot_names = _resolve_plot_names(job.get("report", {}).get("plots"))
    protocol = job["protocol"]
    permute_cfg = job.get("permute", {})
    evaluate_cfg = job["evaluate"]
    select_cfg = job["select"]

    # 2) Loop over each reference sequence
    for ref_name in job["references"]:
        original_seq = refs[ref_name]

        # a) initialize seeds list with the unmodified reference
        seeds = [
            {
                "id": ref_name,  # unique ID
                "ref_name": ref_name,  # human name
                "protocol": protocol,  # protocol used for permuting
                "sequence": original_seq,
                "modifications": [],  # no edits yet
                "round": 1,  # starting round
            }
        ]
        all_variants: List[Dict] = []  # collect all rounds’ variants

        # determine how many rounds to run
        total_rounds = 1
        if job.get("iterate", {}).get("enabled", False):
            total_rounds = job["iterate"]["total_rounds"]

        # b) iterate through rounds
        for round_idx in range(1, total_rounds + 1):
            new_variants: List[Dict] = []

            # i) permute each current seed
            for seed in seeds:
                raw = permute_record(
                    seed,
                    protocol,
                    params=permute_cfg.get("params", {}),
                    regions=permute_cfg.get("regions", []),
                    lookup_tables=permute_cfg.get("lookup_tables", []),
                )
                for var in tqdm(raw, desc=f"{job_name} perm r{round_idx}", unit="var"):
                    # carry forward modification history
                    var["modifications"] = seed["modifications"] + var["modifications"]
                    var["round"] = round_idx
                    new_variants.append(var)
            _LOG.info(
                f"[{job_name}] round {round_idx}: generated {len(new_variants)} variants"
            )

            # ii) evaluate all new variants
            evaluator = get_evaluator(
                evaluate_cfg["evaluator"], **evaluate_cfg.get("params", {})
            )
            for var in tqdm(
                new_variants, desc=f"{job_name} eval r{round_idx}", unit="seq"
            ):
                score = evaluator.score(
                    [var["sequence"]],
                    metric=evaluate_cfg["metric"],
                    ref_sequence=original_seq,
                )[0]
                var["score"] = score
                var["score_type"] = evaluate_cfg["metric"]
            _LOG.info(f"[{job_name}] round {round_idx}: evaluated scores")

            # iii) select elites for next round
            seeds = select(new_variants, select_cfg)
            _LOG.info(f"[{job_name}] round {round_idx}: selected {len(seeds)} elites")

            # collect for final CSV
            all_variants.extend(new_variants)

        # 3) write out CSVs and snapshot config
        output_dir = base_dir / "batch_results" / job_name / ref_name
        write_results(all_variants, seeds, output_dir, job_name, plot_names)
        shutil.copy2(cfg_path, output_dir / "config_snapshot.yaml")
        _LOG.info(f"[{job_name}] outputs & plots written to {output_dir}")

    _LOG.info(f"✔ Job '{job_name}' complete")


def main() -> None:
    """
    Parse CLI, load & validate config, then run each job in sequence.
    """
    # locate default config next to this script
    default_cfg = Path(__file__).parent / "config.yaml"
    parser = argparse.ArgumentParser(description="Run the permuter pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"YAML configuration file (default: {default_cfg.name})",
    )
    args = parser.parse_args()

    # fail early if config missing
    if not args.config.is_file():
        _LOG.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # load & validate experiment + jobs
    validator = _Validator(str(args.config))
    exp_cfg, jobs_cfg = validator.experiment(), validator.jobs()
    _LOG.info(f"Experiment '{exp_cfg['name']}' with {len(jobs_cfg)} job(s)")

    # run each job
    base_dir = Path(__file__).parent
    for job in jobs_cfg:
        _run_job(job, base_dir, args.config)

    _LOG.info("✔ All jobs completed successfully.")


if __name__ == "__main__":
    main()
