"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/subsampler.py

Performs iterative random subsampling from a list of sequences.
Implements deduplication (by unique IDs), caching of computed metrics,
and progress tracking via tqdm.

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
                indices = self._draw_subsample()
                subsample_ids = frozenset(self.sequences[i]["id"] for i in indices)
                if subsample_ids in self.cache:
                    attempts += 1
                    continue
                subsample = [self.sequences[i] for i in indices]
                billboard_metric = compute_billboard_metric(subsample, self.config)
                evo2_metric = compute_evo2_metric(subsample, self.config)
                self.cache[subsample_ids] = {
                    "billboard_metric": billboard_metric,
                    "evo2_metric": evo2_metric,
                    "indices": indices,
                    "selected_ids": [self.sequences[i]["id"] for i in indices]
                }
                self.subsamples.append({
                    "subsample_id": f"sublibrary_{draws+1:03d}",
                    "billboard_metric": billboard_metric,
                    "evo2_metric": evo2_metric,
                    "indices": indices,
                    "selected_ids": [self.sequences[i]["id"] for i in indices]
                })
                draws += 1
                progress_bar.update(1)
                break
            else:
                progress_bar.close()
                raise RuntimeError("Maximum attempts exceeded while deduplicating subsamples.")
        progress_bar.close()
        return self.subsamples
