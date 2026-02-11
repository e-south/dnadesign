"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/pretty.py

Pretty-printing utilities and column profiling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import pyarrow as pa

# -------------------------
# Pretty-printing primitives
# -------------------------


@dataclass(frozen=True)
class PrettyOpts:
    max_colwidth: int = 80  # final printed cell width cap
    max_list_items: int = 6  # items shown from list/array
    max_dict_items: int = 6  # items shown from dict
    precision: int = 4  # numeric precision
    null_token: str = "∅"  # printed representation for nulls
    depth: int = 2  # max nested rendering depth


def _ellipsize(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    head = max_len // 2 - 1
    tail = max_len - head - 1
    return s[:head] + "…" + s[-tail:]


def _fmt_num(x: Any, prec: int) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf == 0:
        return "0"
    ax = abs(xf)
    if ax >= 1e6 or ax < 1e-4:
        return f"{xf:.{prec}g}"
    return f"{xf:.{prec}f}".rstrip("0").rstrip(".")


def _shape_of(obj: Any) -> Tuple[int, ...] | None:
    try:
        import numpy as np  # optional

        if isinstance(obj, np.ndarray):
            return tuple(int(d) for d in obj.shape)
    except ImportError:
        pass
    # best-effort for nested lists
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        if n == 0 or not isinstance(obj[0], (list, tuple)):
            return (n,)
        # rectangular only
        sub = _shape_of(obj[0])
        if sub:
            return (n,) + sub
        return (n,)
    return None


def _fmt_list(xs: Iterable[Any], opts: PrettyOpts, depth: int) -> str:
    # summarize numerics compactly
    xs = list(xs)
    n = len(xs)
    # numeric?
    numeric = all(isinstance(v, (int, float)) or (v is None) for v in xs)
    shown = xs[: opts.max_list_items]
    if numeric:
        body = ", ".join(_fmt_num(v, opts.precision) if v is not None else opts.null_token for v in shown)
    else:
        body = ", ".join(fmt_value(v, opts, depth + 1) for v in shown)
    suffix = ", …" if n > opts.max_list_items else ""
    shp = _shape_of(xs)
    if shp and len(shp) > 1:
        hdr = "tensor[" + "×".join(str(d) for d in shp) + "]"
    else:
        hdr = f"vec[{n}]"
    out = f"{hdr} [{body}{suffix}]"
    return _ellipsize(out, opts.max_colwidth)


def _fmt_dict(d: dict, opts: PrettyOpts, depth: int) -> str:
    items = list(d.items())
    shown = items[: opts.max_dict_items]
    parts = []
    for k, v in shown:
        ks = str(k)
        vs = fmt_value(v, opts, depth + 1)
        parts.append(f"{ks}: {vs}")
    suffix = ", …" if len(items) > opts.max_dict_items else ""
    out = "{ " + ", ".join(parts) + suffix + " }"
    return _ellipsize(out, opts.max_colwidth)


def fmt_value(v: Any, opts: PrettyOpts = PrettyOpts(), depth: int = 0) -> str:
    if v is None:
        return opts.null_token
    if depth >= opts.depth:
        # cap expansion of nested structures
        if isinstance(v, (list, tuple)):
            n = len(v)
            return f"vec[{n}] […]"
        if isinstance(v, dict):
            return "{…}"
        return _ellipsize(str(v), opts.max_colwidth)
    # specific types
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _fmt_num(v, opts.precision)
    # numpy ndarray
    try:
        import numpy as np  # optional

        if isinstance(v, np.ndarray):
            return _fmt_list(v.tolist(), opts, depth)
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        return _fmt_list(v, opts, depth)
    if isinstance(v, dict):
        return _fmt_dict(v, opts, depth)
    # string / fallback
    s = str(v)
    return _ellipsize(s, opts.max_colwidth)


# -------------------------
# Arrow schema pretty tree
# -------------------------


def _render_field(f: pa.Field, indent: int = 0) -> List[str]:
    pad = "  " * indent
    nullable = " (nullable)" if f.nullable else ""
    t = f.type
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        child = t.value_field
        line = f"{pad}{f.name}: list<{child.type}>{nullable}"
        rest = _render_field(pa.field("element", child.type, child.nullable), indent + 1)
        return [line] + rest
    if pa.types.is_struct(t):
        lines = [f"{pad}{f.name}: struct{nullable}"]
        for sf in t:
            lines += _render_field(sf, indent + 1)
        return lines
    return [f"{pad}{f.name}: {t}{nullable}"]


def render_schema_tree(schema: pa.Schema) -> str:
    out: List[str] = []
    for f in schema:
        out += _render_field(f, 0)
    return "\n".join(out)


# -------------------------
# Column profiling (succinct)
# -------------------------


def _list_len_stats(vals: List[Any]) -> Tuple[int | None, int | None, float | None]:
    lens = []
    for v in vals:
        if isinstance(v, (list, tuple)):
            lens.append(len(v))
    if not lens:
        return None, None, None
    return min(lens), max(lens), sum(lens) / len(lens)


def profile_table(
    tbl: pa.Table,
    opts: PrettyOpts,
    columns: List[str] | None = None,
    sample: int = 1024,
) -> List[dict]:
    cols = columns or list(tbl.schema.names)
    n = tbl.num_rows
    out = []
    for name in cols:
        arr = tbl.column(name)
        nn = n - (arr.null_count or 0)
        example = None
        ex_slice = arr.slice(0, min(sample, n)).to_pylist()
        for v in ex_slice:
            if v is not None:
                example = v
                break
        example_s = fmt_value(example, opts) if example is not None else opts.null_token
        minL = maxL = avgL = None
        # soft stats for list-like cols
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            minL, maxL, avgL = _list_len_stats(ex_slice)
        out.append(
            {
                "column": name,
                "type": str(arr.type),
                "non_null": nn,
                "nulls": (arr.null_count or 0),
                "null_pct": (100.0 * (arr.null_count or 0) / n) if n else 0.0,
                "example": example_s,
                "list_min": minL,
                "list_max": maxL,
                "list_avg": round(avgL, 2) if isinstance(avgL, (int, float)) else None,
            }
        )
    return out


def profile_batches(
    batches: Iterable[pa.RecordBatch],
    schema: pa.Schema,
    opts: PrettyOpts,
    columns: List[str] | None = None,
    sample: int = 1024,
    total_rows: int | None = None,
) -> List[dict]:
    cols = columns or list(schema.names)
    for name in cols:
        if schema.get_field_index(name) < 0:
            raise KeyError(f"Column not found in schema: {name}")
    null_counts = {name: 0 for name in cols}
    samples = {name: [] for name in cols}
    rows_seen = 0
    for batch in batches:
        rows_seen += batch.num_rows
        for idx, name in enumerate(cols):
            arr = batch.column(idx)
            null_counts[name] += arr.null_count or 0
            if sample > 0 and len(samples[name]) < sample:
                take = sample - len(samples[name])
                samples[name].extend(arr.slice(0, take).to_pylist())
    n = total_rows if total_rows is not None else rows_seen
    out = []
    for name in cols:
        arr_type = schema.field(name).type
        vals = samples[name]
        example = next((v for v in vals if v is not None), None)
        example_s = fmt_value(example, opts) if example is not None else opts.null_token
        minL = maxL = avgL = None
        if pa.types.is_list(arr_type) or pa.types.is_large_list(arr_type):
            minL, maxL, avgL = _list_len_stats(vals)
        nulls = null_counts[name]
        out.append(
            {
                "column": name,
                "type": str(arr_type),
                "non_null": (n - nulls) if n else 0,
                "nulls": nulls,
                "null_pct": (100.0 * nulls / n) if n else 0.0,
                "example": example_s,
                "list_min": minL,
                "list_max": maxL,
                "list_avg": round(avgL, 2) if isinstance(avgL, (int, float)) else None,
            }
        )
    return out
