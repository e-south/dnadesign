"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/artifacts/traces.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from arviz.data.inference_data import InferenceData


def save_trace(idata: InferenceData, path: Path) -> None:
    """
    Save an ArviZ InferenceData object to NetCDF.

    Ensures the parent directory exists before writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        idata.to_netcdf(path)
    except Exception as exc:  # pragma: no cover - backend availability varies by env
        raise RuntimeError(
            "NetCDF backend missing; install netCDF4 or h5netcdf, or set sample.output.trace.save=false in the config."
        ) from exc
