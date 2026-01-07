"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/parquet_io.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..core.stderr_filter import maybe_install_pyarrow_sysctl_filter
from ..core.utils import OpalError, ensure_dir


@lru_cache(maxsize=1)
def _pyarrow():
    maybe_install_pyarrow_sysctl_filter()
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    return pa, pc, ds, pq


def pyarrow_compute():
    return _pyarrow()[1]


def dataset_from_dir(path: Path):
    _, _, ds, _ = _pyarrow()
    return ds.dataset(str(path))


def table_from_pandas(df: pd.DataFrame, *, schema=None):
    pa = _pyarrow()[0]
    return pa.Table.from_pandas(df, preserve_index=False, schema=schema)


def write_parquet_table(path: Path, table) -> None:
    pq = _pyarrow()[3]
    pq.write_table(table, path)


def read_parquet_df(
    path: Path,
    *,
    columns: Optional[Sequence[str]] = None,
    dtype_backend: str | None = None,
) -> pd.DataFrame:
    """
    Read Parquet via the PyArrow engine with explicit dtype_backend.
    Default keeps Python lists for list/vector columns; set dtype_backend="pyarrow"
    when Arrow-backed dtypes are desired.
    """
    try:
        if dtype_backend is not None:
            return pd.read_parquet(
                path,
                columns=list(columns) if columns else None,
                engine="pyarrow",
                dtype_backend=dtype_backend,
            )
        return pd.read_parquet(path, columns=list(columns) if columns else None, engine="pyarrow")
    except TypeError as exc:
        # dtype_backend not supported → require newer pandas instead of falling back.
        raise OpalError(
            "pandas.read_parquet does not support dtype_backend; upgrade pandas to use OPAL's Parquet dtype handling."
        ) from exc


def write_parquet_df(
    path: Path,
    df: pd.DataFrame,
    *,
    index: bool = False,
    ensure_parent: bool = True,
) -> None:
    """
    Write Parquet via PyArrow. Use this for all OPAL parquet writes.
    """
    if ensure_parent:
        ensure_dir(Path(path).parent)
    try:
        df.to_parquet(path, index=index, engine="pyarrow")
    except TypeError as exc:
        raise OpalError("pandas.to_parquet with engine='pyarrow' failed.") from exc


def schema_signature(schema) -> list[tuple[str, str]]:
    """
    Stable, human-readable schema signature for error messages/comparisons.
    """
    return [(field.name, str(field.type)) for field in schema]
