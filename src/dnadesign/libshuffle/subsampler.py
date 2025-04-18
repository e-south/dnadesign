"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/subsampler.py

Performs iterative random subsampling from a list of sequences.
Implements deduplication (by unique IDs), caching of computed metrics,
and progress tracking via tqdm.
Now, it stores raw Billboard metrics (for composite transformation),
Evo2 metrics, and Needleman–Wunsch alignment metrics per subsample.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import random
from pathlib import Path
import torch
from tqdm import tqdm
from .metrics import compute_billboard_metric, compute_evo2_metric

class Subsampler:
    def __init__(self, sequences, config):
        """
        sequences: list of sequence dictionaries
        config: dictionary loaded from libshuffle configuration
        """
        self.sequences = sequences
        self.config = config
        self.subsample_size = config.get("subsample_size", 100)
        self.num_draws = config.get("num_draws", 200)
        self.with_replacement = config.get("with_replacement", False)
        self.random_seed = config.get("random_seed", 42)
        self.max_attempts_per_draw = config.get("max_attempts_per_draw", 10)
        random.seed(self.random_seed)
        self.cache = {}  # key: frozenset of sequence IDs, value: metrics dict
        self.subsamples = []  # List of subsample result dictionaries

    def _validate_entries(self):
        for entry in self.sequences:
            if "id" not in entry:
                raise ValueError("Every sequence entry must contain a unique 'id' field.")

    def _draw_subsample(self):
        if self.with_replacement:
            return [random.randrange(len(self.sequences)) for _ in range(self.subsample_size)]
        else:
            return random.sample(range(len(self.sequences)), self.subsample_size)

    def run(self):
        self._validate_entries()
        total_draws = self.num_draws
        progress_bar = tqdm(total=total_draws, desc="Subsampling")
        draws = 0

        while draws < total_draws:
            attempts = 0
            while attempts < self.max_attempts_per_draw:
                # pick a unique set of indices
                indices = self._draw_subsample()
                subsample_ids = frozenset(self.sequences[i]["id"] for i in indices)
                if subsample_ids in self.cache:
                    attempts += 1
                    continue

                # build the actual subsample
                subsample = [self.sequences[i] for i in indices]
                unique_clusters = {
                    seq.get("meta_cluster_count")
                    for seq in subsample
                    if "meta_cluster_count" in seq
                }
                unique_cluster_count = len(unique_clusters)

                # compute all metrics
                raw_billboard = compute_billboard_metric(subsample, self.config)
                evo2_metric = compute_evo2_metric(subsample, self.config)
                # also compute L2 version for plotting
                l2_config = self.config.copy()
                l2_config["evo2_metric"] = {"type": "l2"}
                evo2_metric_l2 = compute_evo2_metric(subsample, l2_config)

                # cache the raw results
                self.cache[subsample_ids] = {
                    "raw_billboard": raw_billboard,
                    "evo2_metric": evo2_metric,
                    "evo2_metric_l2": evo2_metric_l2,
                    "indices": indices,
                    "selected_ids": [self.sequences[i]["id"] for i in indices]
                }

                # extract NW metrics if present
                nw_sim, nw_diss = None, None
                if isinstance(raw_billboard, dict):
                    nw_sim  = raw_billboard.get("nw_similarity")
                    nw_diss = raw_billboard.get("nw_dissimilarity")

                # assemble the entry
                composite_enabled = self.config.get("billboard_metric", {}).get("composite_score", False)
                entry = {
                    "subsample_id": f"sublibrary_{draws+1:03d}",
                    "evo2_metric": evo2_metric,
                    "evo2_metric_l2": evo2_metric_l2,
                    "indices": indices,
                    "selected_ids": [self.sequences[i]["id"] for i in indices],
                    "unique_cluster_count": unique_cluster_count
                }

                if composite_enabled:
                    entry["raw_billboard_vector"] = raw_billboard
                    entry["billboard_metric"] = None  # placeholder until composite transform
                else:
                    entry["billboard_metric"] = raw_billboard

                # attach the NW scores for easy plotting/filtering downstream
                entry["nw_similarity"]    = nw_sim
                entry["nw_dissimilarity"] = nw_diss

                # record and advance
                self.subsamples.append(entry)
                draws += 1
                progress_bar.update(1)
                break
            else:
                progress_bar.close()
                raise RuntimeError("Maximum attempts exceeded while deduplicating subsamples.")

        progress_bar.close()
        return self.subsamples