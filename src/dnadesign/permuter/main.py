"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/main.py

Main entry point for the permuter module.

Canonical output-entry schema:
{
  "id": "<uuid>",
  "meta_date_accessed": "2025-04-30T12:34:56",
  "sequence": "ATGCCG…",
  "protocol": "scan_dna",
  "ref_name": "my_ref_sequence_name",
  "modifications": [{"pos":5,"from":"A","to":"G"}],
  "score": 0.456,
  "score_type": "llr",
  "round": 1,
}

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import glob
import logging
import os
import time
import uuid

import pandas as pd
import yaml

from dnadesign.permuter.evaluator import evaluate
from dnadesign.permuter.iterator import iterate
from dnadesign.permuter.permute_record import permute_record
from dnadesign.permuter.reporter import write_results
from dnadesign.permuter.selector import select


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_config(path="config.yaml"):
    cfg = yaml.safe_load(open(path))["permuter"]
    return cfg["experiment"], cfg["jobs"]


def ingest_reference(job):
    logger.info(
        f"[{job['name']}] Ingesting reference from '{job.get('inputs', job.get('input_file', ''))}'"
    )
    inp_dir = os.path.join("batch_inputs", job.get("inputs", ""))
    csvs = glob.glob(os.path.join(inp_dir, "*.csv"))
    assert len(csvs) == 1, f"Expected one CSV in {inp_dir}"
    df = pd.read_csv(csvs[0])
    if "references" in job:
        ref_name = job["references"][0]
        row = df[df["ref_name"] == ref_name].iloc[0]
    else:
        row = df.iloc[0]
        ref_name = row["ref_name"]

    entry = {
        "id": str(uuid.uuid4()),
        "meta_date_accessed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sequence": row["sequence"],
        "protocol": job["protocol"],
        "ref_name": ref_name,
        "modifications": [],
    }
    logger.info(f"[{job['name']}] Loaded reference '{ref_name}' (ID: {entry['id']})")
    return entry


def main(config_path="config.yaml"):
    setup_logging()
    exp, jobs = load_config(config_path)
    logger.info(f"Starting experiment '{exp['name']}' with {len(jobs)} job(s)")

    for job in jobs:
        job_name = job["name"]
        logger.info(f"\n=== Job: {job_name} ===")

        # 1) Ingest reference
        ref_entry = ingest_reference(job)

        # 2) Permute variants
        logger.info(
            f"[{job_name}] Permuting variants using protocol '{job['protocol']}'"
        )
        variants = list(
            permute_record(
                ref_entry,
                job["protocol"],
                job.get("params", {}),
                job.get("permute", {}).get("regions", []),
                job.get("permute", {}).get("lookup_tables", []),
            )
        )
        logger.info(f"[{job_name}] Generated {len(variants)} variant(s)")

        # 3) Evaluate
        metric = job["evaluate"]["metric"]
        logger.info(f"[{job_name}] Evaluating variants with metric '{metric}'")
        scores = evaluate(variants, metric=metric, **job["evaluate"].get("params", {}))

        # map metric to score_type
        score_map = {
            "log_likelihood": "ll",
            "log_likelihood_ratio": "llr",
            "euclidean_distance": "euclid",
        }
        score_type = score_map.get(metric, metric)

        for v, s in zip(variants, scores):
            v["score"] = s
            v["score_type"] = score_type
            v["round"] = 1
        logger.info(f"[{job_name}] Scored all variants (score_type='{score_type}')")

        # 4) Select
        logger.info(
            f"[{job_name}] Selecting elites with strategy '{job['select']['strategy']}'"
        )
        elites = select(
            variants,
            strategy=job["select"]["strategy"],
            **{k: job["select"][k] for k in ["k", "threshold"] if k in job["select"]},
        )
        logger.info(f"[{job_name}] Selected {len(elites)} elite(s)")

        # 5) Optional iterate
        if job.get("iterate", {}).get("enabled", False):
            max_rounds = job["iterate"].get(
                "total_rounds", job["iterate"].get("max_rounds")
            )
            logger.info(f"[{job_name}] Iterating for up to {max_rounds} rounds")
            variants, elites = iterate(
                elites,
                iterator_cfg=job["iterate"],
                protocol=job["protocol"],
                params=job.get("params", {}),
                regions=job.get("permute", {}).get("regions", []),
                lookup_tables=job.get("permute", {}).get("lookup_tables", []),
            )
            logger.info(
                f"[{job_name}] Iteration complete: {len(elites)} elite(s) after {max_rounds} rounds"
            )

        # 6) Report & write
        results_dir = os.path.join("results", job_name)
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"[{job_name}] Writing results to '{results_dir}'")
        write_results(variants, elites, job_name, results_dir)
        logger.info(f"[{job_name}] Job complete")

    logger.info("All jobs finished.")


if __name__ == "__main__":
    main()
