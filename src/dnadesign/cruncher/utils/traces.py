"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/utils/traces.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
import arviz as az
from arviz.data.inference_data import InferenceData

def save_trace(idata: InferenceData, path: Path) -> None:
    """
    Save an ArviZ InferenceData object to NetCDF.

    Ensures the parent directory exists before writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(path)