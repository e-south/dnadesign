"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/subsampler.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import random
import logging
from tqdm import tqdm
from dnadesign.libshuffle.metrics import compute_pairwise_stats, compute_billboard_metric

class Subsampler:
    """Draw random subsamples, compute metrics, cache duplicates."""
    def __init__(self, sequences, cfg):
        self.seq = sequences
        self.cfg = cfg
        random.seed(cfg.random_seed)
        self.log = logging.getLogger('libshuffle.subsampler')

    def run(self):
        n, draws = self.cfg.subsample_size, self.cfg.num_draws
        seen, out = set(), []
        self.log.info(f"Drawing {draws} subsamples of size {n}")
        pbar = tqdm(total=draws, desc="Subsampling")
        count = attempts = 0
        while count < draws:
            idxs = ([random.randrange(len(self.seq)) for _ in range(n)]
                    if self.cfg.with_replacement else
                    random.sample(range(len(self.seq)), n))
            key = frozenset(self.seq[i]['id'] for i in idxs)
            if key in seen:
                attempts += 1
                if attempts >= self.cfg.max_attempts_per_draw:
                    raise RuntimeError("Too many duplicates.")
                continue
            seen.add(key)
            subset = [self.seq[i] for i in idxs]
            stats = compute_pairwise_stats(subset)
            bb = compute_billboard_metric(subset, self.cfg)
            uniq = len({s.get('meta_cluster_count') for s in subset})
            out.append({
                'subsample_id': f"s_{count+1:04d}",
                'indices': idxs,
                **stats,
                'raw_billboard': bb,
                'unique_cluster_count': uniq,
            })
            count += 1
            pbar.update(1)
        pbar.close()
        return out
