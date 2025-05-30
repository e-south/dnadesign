"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import datetime
import logging
import sys
from pathlib import Path

import torch

from dnadesign.libshuffle.config import LibShuffleConfig
from dnadesign.libshuffle.output import OutputWriter
from dnadesign.libshuffle.selection import select_best_subsample
from dnadesign.libshuffle.subsampler import Subsampler
from dnadesign.libshuffle.visualization import Plotter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("libshuffle")


def main():
    cfg = LibShuffleConfig.load("configs/example.yaml")

    project_root = Path(__file__).resolve().parent.parent
    pt_dir = project_root / cfg.input_pt_path
    if not pt_dir.exists():
        alt = project_root / "sequences" / cfg.input_pt_path
        if alt.exists():
            pt_dir = alt
        else:
            logger.error(f"Input directory not found: {pt_dir}")
            sys.exit(1)

    pts = list(pt_dir.glob("*.pt"))
    if len(pts) != 1:
        logger.error(f"Expect exactly one .pt file in {pt_dir}, found {len(pts)}.")
        sys.exit(1)
    pt_file = pts[0]

    ts = datetime.datetime.now().strftime("%Y%m%d")
    outdir = project_root / "libshuffle" / "batch_results" / f"{cfg.output_dir_prefix}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {outdir.resolve()}")

    sequences = torch.load(pt_file, weights_only=True)
    if not isinstance(sequences, list):
        logger.error(".pt must contain a list of sequence dicts.")
        sys.exit(1)

    subsamples = Subsampler(sequences, cfg).run()
    winner = select_best_subsample(subsamples, cfg, sequences)
    logger.info(f"Selected subsample: {winner['subsample_id']}")

    plotter = Plotter(cfg)
    plotter.plot_scatter(subsamples, winner, outdir)
    plotter.plot_flag_composition(subsamples, outdir)
    plotter.plot_kde(subsamples, outdir)
    plotter.plot_pairplot(subsamples, outdir)
    plotter.plot_hitzone(subsamples, winner, outdir)

    writer = OutputWriter(outdir, cfg)
    writer.save_summary(pt_file, len(sequences))
    writer.save_sublibraries(subsamples)
    if cfg.save_selected:
        writer.save_selected(sequences, winner)
    if cfg.save_sublibraries:
        writer.save_sublibraries_by_id(sequences, subsamples, cfg.save_sublibraries)

    logger.info("Libshuffle complete.")


if __name__ == "__main__":
    main()
