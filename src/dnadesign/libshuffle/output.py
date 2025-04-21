"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/output.py

Provides helper functions to generate global summary and sublibrary YAML files,
and optionally save selected subsample .pt files.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import yaml, torch
from pathlib import Path

def _dump(path, data): Path(path).write_text(yaml.dump(data, sort_keys=False))

class OutputWriter:
    def __init__(self, outdir: Path, cfg):
        self.outdir = outdir
        self.raw = getattr(cfg, '_raw', {})

    def save_summary(self, pt, total):
        summary = {'config': self.raw, 'input_pt': pt.name, 'total_sequences': total}
        _dump(self.outdir/'global_summary.yaml', summary)

    def save_sublibraries(self, subs):
        out = {s['subsample_id']: {'indices': s['indices'],
                    'mean_cosine': s['mean_cosine'],
                    'mean_euclidean': s['mean_euclidean'],
                    'passed': s.get('passed_selection', False)}
               for s in subs}
        _dump(self.outdir/'sublibraries.yaml', out)

    def save_selected(self, seqs, winner):
        sel = [seqs[i] for i in winner['indices']]
        d = self.outdir/'selected'; d.mkdir(exist_ok=True)
        torch.save(sel, d/f"{winner['subsample_id']}.pt")

    def save_sublibraries_by_id(self, seqs, subs, ids):
        outdir = self.outdir/'saved_subs'; outdir.mkdir(exist_ok=True)
        for s in subs:
            if s['subsample_id'] in ids:
                sel = [seqs[i] for i in s['indices']]
                torch.save(sel, outdir/f"{s['subsample_id']}.pt")
