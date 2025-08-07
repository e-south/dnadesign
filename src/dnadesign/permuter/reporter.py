"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/reporter.py

Write variants and elites to JSON and simple plots.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def write_results(
    variants: List[Dict], elites: List[Dict], job_name: str, results_dir: str
):
    # all_variants
    all_path = os.path.join(results_dir, "all_variants.json")
    with open(all_path, "w") as f:
        json.dump(variants, f, indent=2)
    # elites
    elites_path = os.path.join(results_dir, "elites.json")
    with open(elites_path, "w") as f:
        json.dump(elites, f, indent=2)
    # plot
    scores = [e["score"] for e in elites]
    plt.figure()
    plt.bar(range(len(scores)), scores)
    plt.xlabel("Elite Index")
    plt.ylabel("Score")
    plt.title(job_name)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "elites.png"))
    plt.close()
