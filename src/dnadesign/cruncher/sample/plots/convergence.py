"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/plots/convergence.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
import arviz as az
import json

def report_convergence(idata: az.InferenceData, out_dir: Path) -> None:
    """
    Compute R̂ and ESS for the “score” variable,
    save a JSON or TXT summary → convergence.txt
    """
    rhat = az.rhat(idata, var_names=["score"])["score"].item()
    ess  = az.ess(idata, var_names=["score"])["score"].item()
    out = out_dir / "convergence.txt"
    out_dir.mkdir(exist_ok=True, parents=True)
    with out.open("w") as fh:
        fh.write(f"rhat: {rhat:.3f}\n")
        fh.write(f"ess:  {ess:.1f}\n")