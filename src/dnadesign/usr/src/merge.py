"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/sync.py

USR Merge (PT → Parquet)

Append ONLY new rows from one or more archived .pt files into an existing USR
dataset (records.parquet). Validates & canonicalizes rows, computes ids with
USR defaults, enforces DNA alphabet/length screens, performs destination-aware
type/shape validation for mapped columns, prints a dry-run plan, and (optionally)
executes the merge atomically with an events.log entry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import torch

from .dataset import Dataset
from .errors import NamespaceError, SchemaError, ValidationError
from .io import append_event, read_parquet, write_parquet_atomic
from .normalize import compute_id, normalize_sequence
from .schema import REQUIRED_COLUMNS

# ------------------------------- dataclasses -------------------------------


@dataclass(frozen=True)
class MapOptions:
    dtype: Optional[str] = None  # "float32" (default), "float64", "int64", "string"
    flatten_if_leading_dim_is_1: bool = False


@dataclass(frozen=True)
class MapSpec:
    src: str
    dst: str
    options: MapOptions = MapOptions()


@dataclass(frozen=True)
class CarrySpec:
    src: str
    dst: str


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    source_label: str
    defaults_bio_type: str
    defaults_alphabet: str
    mappings: List[MapSpec] = field(default_factory=list)
    carry: List[CarrySpec] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class DestinationSpec:
    dataset: str
    root: Optional[Path]
    defaults_bio_type: str
    defaults_alphabet: str
    expected_length: Optional[int]
    allow_non_acgt: bool


@dataclass(frozen=True)
class MergeConfig:
    version: int
    dest: DestinationSpec
    sources: List[SourceSpec]


@dataclass
class ColumnRef:
    name: str
    type: pa.DataType
    # For list types, we capture a reference "shape" (length for 1-D, tuple for nested)
    ref_shape: Optional[Tuple[int, ...]] = None


@dataclass
class MergePlan:
    cfg_path: Path
    dataset_path: Path
    records_path: Path
    sources_brief: List[Tuple[Path, int, str]]  # (path, entries, label)

    # screening counts
    screen_non_acgt: int = 0
    screen_len_mismatch: int = 0
    incoming_dupes: int = 0

    # id reconciliation
    proposed: int = 0
    skipped_existing: int = 0
    skipped_id_conflict: int = 0

    # mapped column validation
    mapped_ok: bool = True
    mapped_mismatch_messages: List[str] = field(default_factory=list)

    # destination preview
    will_add_rows: int = 0
    dest_row_count_before: int = 0
    dest_row_count_after: int = 0

    # column list (new + existing)
    column_targets: List[str] = field(default_factory=list)

    # The batch that would be appended if executed (essential + mapped/carry)
    batch_df: Optional[pd.DataFrame] = None

    # human strings for UX
    column_mapping_hints: List[str] = field(default_factory=list)


@dataclass
class MergeOutcome:
    added: int
    total_after: int


# ------------------------------- config load -------------------------------

_NS_RE = re.compile(r"^[a-z][a-z0-9_]*__")  # <tool>__<field>
_ESSENTIAL = {k for k, _ in REQUIRED_COLUMNS}


def _ensure_abs(p: Optional[str], keyname: str) -> Optional[Path]:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        raise SchemaError(f"{keyname} must be an ABSOLUTE path (got '{p}').")
    return path


def load_merge_config(path: Path) -> MergeConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SchemaError("Bad YAML: root must be a mapping.")
    if raw.get("version") != 1:
        raise SchemaError("merge.yaml version must be 1.")

    dest = raw.get("destination") or {}
    dataset = dest.get("dataset")
    if not dataset or not isinstance(dataset, str):
        raise SchemaError("destination.dataset (dataset folder name) is required.")

    root = _ensure_abs(dest.get("root"), "destination.root")

    dfl = dest.get("defaults") or {}
    d_bio = dfl.get("bio_type", "dna")
    d_alph = dfl.get("alphabet", "dna_4")

    enf = dest.get("enforce") or {}
    expected_length = enf.get("expected_length", None)
    if expected_length is not None:
        if not isinstance(expected_length, int) or expected_length <= 0:
            raise SchemaError(
                "destination.enforce.expected_length must be a positive int."
            )
    allow_non_acgt = bool(enf.get("allow_non_acgt", False))

    srcs = []
    for s in raw.get("sources") or []:
        p = _ensure_abs(s.get("path"), "sources[].path")
        if p is None:
            raise SchemaError("sources[].path is required.")
        slabel = s.get("source_label") or str(p)
        s_dfl = s.get("defaults") or {}
        s_bio = s_dfl.get("bio_type", d_bio)
        s_alph = s_dfl.get("alphabet", d_alph)

        maps = []
        for m in s.get("map") or []:
            frm, to = m.get("from"), m.get("to")
            if not frm or not to:
                raise SchemaError("Every map entry needs 'from' and 'to'.")
            if to in _ESSENTIAL:
                raise NamespaceError(f"Refusing to map into essential column '{to}'.")
            if not _NS_RE.match(to):
                raise NamespaceError(
                    f"Destination column '{to}' must be namespaced like '<tool>__<field>'."
                )
            opt = m.get("options") or {}
            maps.append(
                MapSpec(
                    src=str(frm),
                    dst=str(to),
                    options=MapOptions(
                        dtype=opt.get("dtype"),
                        flatten_if_leading_dim_is_1=bool(
                            opt.get("flatten_if_leading_dim_is_1", False)
                        ),
                    ),
                )
            )

        carry = []
        for c in s.get("carry_over") or []:
            frm, to = c.get("from"), c.get("to")
            if not frm or not to:
                raise SchemaError("Every carry_over entry needs 'from' and 'to'.")
            if to in _ESSENTIAL:
                raise NamespaceError(f"Refusing to carry into essential column '{to}'.")
            if not _NS_RE.match(to):
                raise NamespaceError(
                    f"Carry destination '{to}' must be namespaced like '<tool>__<field>'."
                )
            carry.append(CarrySpec(src=str(frm), dst=str(to)))

        note = s.get("note") or ""
        srcs.append(
            SourceSpec(
                path=p,
                source_label=str(slabel),
                defaults_bio_type=s_bio,
                defaults_alphabet=s_alph,
                mappings=maps,
                carry=carry,
                note=note,
            )
        )

    return MergeConfig(
        version=1,
        dest=DestinationSpec(
            dataset=str(dataset),
            root=root,
            defaults_bio_type=d_bio,
            defaults_alphabet=d_alph,
            expected_length=expected_length,
            allow_non_acgt=allow_non_acgt,
        ),
        sources=srcs,
    )


# ------------------------------- helpers -------------------------------


def _to_list_numeric(
    v: Any, *, flatten_leading_one: bool, dtype: Optional[str]
) -> Optional[List]:
    """
    Coerce tensors/arrays into (possibly nested) Python lists of numeric scalars.
    Applies [1, N, ...] -> [N, ...] if flatten_leading_one=True.
    Returns None for "missing" values.
    """
    if v is None:
        return None
    # Torch tensor
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu()
        if flatten_leading_one and t.ndim >= 1 and t.shape[0] == 1:
            t = t.reshape(*t.shape[1:])
        # dtype coercions (default float32 unless explicitly given)
        target = (dtype or "float32").lower()
        if target in ("float32", "fp32"):
            t = t.to(dtype=torch.float32)
        elif target in ("float64", "double"):
            t = t.to(dtype=torch.float64)
        elif target in ("int64", "long"):
            t = t.to(dtype=torch.int64)
        # bfloat16/float16 → float32 default path already handled above
        return t.tolist()

    # Numpy array?
    try:
        import numpy as np  # local import to keep module lean at import time

        if isinstance(v, np.ndarray):
            a = v
            if flatten_leading_one and a.ndim >= 1 and a.shape[0] == 1:
                a = a.reshape(*a.shape[1:])
            return a.tolist()
    except Exception:
        pass

    # Python lists/tuples (verify shape later)
    if isinstance(v, (list, tuple)):
        out = list(v)
        if (
            flatten_leading_one
            and isinstance(out, list)
            and out
            and isinstance(out[0], list)
            and len(out) == 1
        ):
            out = out[0]
        return out

    # Scalars -> respect dtype rule: if numeric requested, wrap scalar if and only if destination ref expects length 1
    if isinstance(v, (int, float)):
        return [float(v)] if dtype is None or "float" in dtype else [int(v)]
    # Strings that look like JSON arrays
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


def _shape_of_list(v: Any) -> Optional[Tuple[int, ...]]:
    """
    Return uniform nested list shape (e.g., [512] or [16,512]).
    If ragged/mixed or not a list, return None.
    """
    if not isinstance(v, list):
        return None

    def rec(x) -> Tuple[int, ...] | None:
        if not isinstance(x, list):
            return ()
        if not x:
            return (0,)
        # all same type?
        if any(isinstance(xi, list) != isinstance(x[0], list) for xi in x):
            return None
        if isinstance(x[0], list):
            sub = [rec(xi) for xi in x]
            if any(s is None for s in sub):
                return None
            if len(set(sub)) != 1:
                return None
            return (len(x),) + sub[0]
        else:
            return (len(x),)

    return rec(v)


def _first_non_null(values: Sequence[Any]) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _arrow_type_for_new_numeric_list(dtype_opt: Optional[str]) -> pa.DataType:
    t = (dtype_opt or "float32").lower()
    if t in ("float32", "fp32"):
        return pa.list_(pa.float32())
    if t in ("float64", "double"):
        return pa.list_(pa.float64())
    if t in ("int64", "long"):
        return pa.list_(pa.int64())
    # default
    return pa.list_(pa.float32())


def _coerce_scalar_to_string(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        # pretty but compact for dict/list
        if isinstance(v, (list, dict)):
            return json.dumps(v, separators=(",", ":"))
        return str(v)
    except Exception:
        return None


def _get_existing_column_refs(tbl: pa.Table) -> Dict[str, ColumnRef]:
    refs: Dict[str, ColumnRef] = {}
    for f in tbl.schema:
        name = f.name
        if name in _ESSENTIAL:
            continue
        t = f.type
        ref_shape: Optional[Tuple[int, ...]] = None
        if pa.types.is_list(t):
            # probe first non-null for shape
            vals = tbl.column(name).to_pylist()
            v0 = _first_non_null(vals)
            if isinstance(v0, list):
                ref_shape = _shape_of_list(v0)
        refs[name] = ColumnRef(name=name, type=t, ref_shape=ref_shape)
    return refs


def _assert_namespaced_safe(name: str) -> None:
    if name in _ESSENTIAL:
        raise NamespaceError(f"Refusing to write essential column '{name}'.")
    if not _NS_RE.match(name):
        raise NamespaceError(
            f"Column '{name}' must be namespaced as '<tool>__<field>'."
        )


# ------------------------------- plan & execute -------------------------------


def plan_merge(ds: Dataset, cfg: MergeConfig) -> MergePlan:
    ds_dir = ds.dir
    if not ds.records_path.exists():
        raise SchemaError(f"Dataset not found: {ds.records_path}")

    t0 = read_parquet(ds.records_path)
    existing_ids = set(t0.column("id").to_pylist())
    id_to_seq: Dict[str, str] = {}
    # Build id→sequence for conflict detection
    for rid, seq in zip(t0.column("id").to_pylist(), t0.column("sequence").to_pylist()):
        id_to_seq[rid] = seq

    existing_cols = set(t0.schema.names)

    # Reference typing for destination mapped columns
    refs = _get_existing_column_refs(t0)

    # Containers
    out_rows: List[Dict[str, Any]] = []
    counts = {
        "screen_non_acgt": 0,
        "screen_len_mismatch": 0,
        "incoming_dupes": 0,
        "proposed": 0,
        "skipped_existing": 0,
        "skipped_id_conflict": 0,
    }
    col_targets: List[str] = []
    mapping_hints: List[str] = []
    sources_brief: List[Tuple[Path, int, str]] = []

    # For within-incoming dedup: keep first id seen per source+global
    seen_ids_global: set[str] = set()

    # For mapped columns staging: collect column->list of values aligned to accepted rows (we’ll build DataFrame after)
    mapped_names: List[str] = []
    # We’ll populate per-row dicts directly (essential + mapped + carry)

    # Iterate sources
    for s in cfg.sources:
        entries = torch.load(s.path, map_location="cpu")
        if not (isinstance(entries, list) and entries and isinstance(entries[0], dict)):
            raise SchemaError(f"{s.path} must be a non-empty list of dicts.")
        sources_brief.append((s.path, len(entries), s.source_label))

        # record mapping hints
        for m in s.mappings:
            opts = []
            if m.options.dtype:
                opts.append(f"list<{m.options.dtype}>")
            if m.options.flatten_if_leading_dim_is_1:
                opts.append("flatten leading dim if 1")
            opt_s = f" ({', '.join(opts)})" if opts else ""
            mapping_hints.append(f"{m.src:30s} -> {m.dst}{opt_s}")
            _assert_namespaced_safe(m.dst)
            if m.dst not in mapped_names:
                mapped_names.append(m.dst)

        for c in s.carry:
            _assert_namespaced_safe(c.dst)
            if c.dst not in mapped_names:
                mapped_names.append(c.dst)

        # process rows
        for e in entries:
            # 1) Extract & normalize sequence
            if "sequence" not in e:
                # Skip bad row (treat as length mismatch bucket for reporting simplicity)
                counts["screen_len_mismatch"] += 1
                continue
            seq_raw = str(e["sequence"])
            bio = s.defaults_bio_type or cfg.dest.defaults_bio_type
            alph = s.defaults_alphabet or cfg.dest.defaults_alphabet

            # enforce dna_4 alphabet if configured; normalize_sequence raises on non-ACGT
            try:
                seq_norm = normalize_sequence(seq_raw, bio, alph)
            except Exception:
                if bio == "dna" and alph == "dna_4" and not cfg.dest.allow_non_acgt:
                    counts["screen_non_acgt"] += 1
                    continue
                # If allowed, accept original trimmed value
                seq_norm = (seq_raw or "").strip()

            length = len(seq_norm)
            if (
                cfg.dest.expected_length is not None
                and length != cfg.dest.expected_length
            ):
                counts["screen_len_mismatch"] += 1
                continue

            rid = compute_id(bio, seq_norm)

            # 2) de-dup within incoming (keep first)
            if rid in seen_ids_global:
                counts["incoming_dupes"] += 1
                continue

            # 3) reconcile with destination
            if rid in existing_ids:
                if id_to_seq.get(rid) == seq_norm:
                    counts["skipped_existing"] += 1
                    continue
                else:
                    counts["skipped_id_conflict"] += 1
                    continue

            # eligible
            counts["proposed"] += 1
            seen_ids_global.add(rid)

            row: Dict[str, Any] = {
                "id": rid,
                "bio_type": bio,
                "sequence": seq_norm,  # preserve case (already trimmed)
                "alphabet": alph,
                "length": length,
                "source": s.source_label if s.source_label else str(s.path),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }

            # 4) mapped columns
            for m in s.mappings:
                val = e.get(m.src, None)
                # Coerce vectors/tensors → list[*] if they look numeric
                list_val = _to_list_numeric(
                    val,
                    flatten_leading_one=m.options.flatten_if_leading_dim_is_1,
                    dtype=m.options.dtype,
                )
                if list_val is not None:
                    row[m.dst] = list_val
                else:
                    # Fall back to string if destination is string ref OR value is non-numeric structure
                    # If destination exists and is list/number, coercion failure will be caught in validation below.
                    row[m.dst] = _coerce_scalar_to_string(val)

            # 5) carry_over (verbatim best-effort → string)
            for c in s.carry:
                row[c.dst] = _coerce_scalar_to_string(e.get(c.src, None))

            out_rows.append(row)

    # Prepare DataFrame for proposed rows
    if out_rows:
        batch_df = pd.DataFrame(out_rows)
    else:
        batch_df = pd.DataFrame(columns=[k for k, _ in REQUIRED_COLUMNS])

    # Validate mapped columns against destination references
    mapped_cols_in_batch = [c for c in batch_df.columns if c not in _ESSENTIAL]
    column_targets = sorted(set(mapped_cols_in_batch))
    mismatches: List[str] = []

    # Build per-column reference/type inference
    # 1) Existing destination columns: enforce exact Arrow type + (if list) shape
    for col in column_targets:
        if col in refs:
            ref = refs[col]
            vals = batch_df[col].tolist() if col in batch_df.columns else []
            # If destination is list<T>, enforce uniform list shape == ref.ref_shape
            if pa.types.is_list(ref.type) and ref.ref_shape is not None:
                # Extract shapes from non-null values
                shapes = []
                for v in vals:
                    if v is None:
                        continue
                    sh = _shape_of_list(v) if isinstance(v, list) else None
                    if sh is None:
                        mismatches.append(
                            f"{col}: expected uniform list with shape {ref.ref_shape}, "
                            f"got non-list/scalar for a new row."
                        )
                        break
                    shapes.append(sh)
                if shapes and any(s != ref.ref_shape for s in shapes):
                    mismatches.append(
                        f"{col}: list shape mismatch (incoming {set(shapes)} vs reference {ref.ref_shape})."
                    )
        else:
            # New column: if appears list-like in batch, enforce uniform shape within batch
            if col in batch_df.columns:
                vals = batch_df[col].tolist()
                example = _first_non_null(vals)
                if isinstance(example, list):
                    sh = _shape_of_list(example)
                    if sh is None:
                        mismatches.append(
                            f"{col}: ragged/mixed nested list shapes are not supported."
                        )
                    else:
                        # Ensure every non-null matches this shape
                        bad = []
                        for v in vals:
                            if v is None:
                                continue
                            if _shape_of_list(v) != sh:
                                bad.append(v)
                                break
                        if bad:
                            mismatches.append(
                                f"{col}: inconsistent list shapes in batch (expected shape {sh})."
                            )

    plan = MergePlan(
        cfg_path=Path(""),  # filled by caller
        dataset_path=ds_dir,
        records_path=ds.records_path,
        sources_brief=sources_brief,
        screen_non_acgt=counts["screen_non_acgt"],
        screen_len_mismatch=counts["screen_len_mismatch"],
        incoming_dupes=counts["incoming_dupes"],
        proposed=counts["proposed"],
        skipped_existing=counts["skipped_existing"],
        skipped_id_conflict=counts["skipped_id_conflict"],
        mapped_ok=(len(mismatches) == 0),
        mapped_mismatch_messages=mismatches,
        will_add_rows=max(0, counts["proposed"]),
        dest_row_count_before=t0.num_rows,
        dest_row_count_after=t0.num_rows + max(0, counts["proposed"]),
        column_targets=column_targets,
        batch_df=batch_df,
        column_mapping_hints=mapping_hints,
    )
    return plan


# ------------------------------- UX helpers -------------------------------


def format_dry_run(plan: MergePlan, cfg: MergeConfig, cfg_path: Path) -> str:
    def fmt_int(n):
        return f"{n:,}"

    def fmt_pct(n, d):
        return f"{(100.0*n/d):.1f}%" if d > 0 else "0.0%"

    src_lines = []
    for p, n, lbl in plan.sources_brief:
        src_lines.append(f"  • {str(p)}  entries={fmt_int(n):>7}  label={lbl}")

    map_lines = []
    for hint in plan.column_mapping_hints:
        map_lines.append(f"  {hint}")

    # Row screening pre-id
    pre_screen = []
    if cfg.dest.defaults_bio_type == "dna" and cfg.dest.defaults_alphabet == "dna_4":
        pre_screen.append(
            f"  invalid alphabet (non-ACGT, dna_4): {fmt_int(plan.screen_non_acgt)}"
        )
    if cfg.dest.expected_length is not None:
        pre_screen.append(
            f"  expected_length mismatch ({cfg.dest.expected_length}):      {fmt_int(plan.screen_len_mismatch)}"
        )
    if plan.incoming_dupes:
        pre_screen.append(
            f"  incoming duplicates (same id within batch): {fmt_int(plan.incoming_dupes)}"
        )

    id_lines = [
        f"  proposed (after screening):         {fmt_int(plan.proposed)}",
        f"  already present (same id+seq):      {fmt_int(plan.skipped_existing)}   (skipped)",
        f"  id conflict (same id, diff seq):    {fmt_int(plan.skipped_id_conflict)}   (skipped + WARN)",
    ]

    type_lines = (
        ["  mapped columns OK                   (0 mismatches)"]
        if plan.mapped_ok
        else ["  MISMATCHES:"] + [f"   - {m}" for m in plan.mapped_mismatch_messages]
    )

    pct = fmt_pct(plan.will_add_rows, max(1, plan.proposed))
    body = [
        "MERGE PLAN (dry-run)",
        f"Dataset        : {cfg.dest.dataset}",
        f"Records path   : {str(plan.records_path)}",
        f"Config         : {str(cfg_path)}",
        "",
        f"Sources ({len(plan.sources_brief)}):",
        *src_lines,
        "",
        "Column mappings:" if map_lines else "Column mappings: (none)",
        *(map_lines if map_lines else []),
        "",
        "Row screening (pre-id):",
        *(pre_screen if pre_screen else ["  (none)"]),
        "",
        "ID results:",
        *id_lines,
        "",
        "Type/shape validation:",
        *type_lines,
        "",
        "Will ADD:",
        f"  {fmt_int(plan.will_add_rows)} rows ({pct} of screened), to dataset with {fmt_int(plan.dest_row_count_before)} → {fmt_int(plan.dest_row_count_after)} rows",
    ]
    return "\n".join(body)


# ------------------------------- execution -------------------------------


def _to_arrow_array_for_column(
    name: str,
    values: List[Any],
    t0: pa.Table,
    refs: Dict[str, ColumnRef],
    dtype_opt: Optional[str],
) -> pa.Array:
    """
    Build an Arrow array for the mapped/carry column `name`, casting to destination
    type if exists; otherwise infer a sensible type (lists default to float32).
    """
    if name in refs:
        ref = refs[name]
        # Cast to exact destination type
        t = ref.type
        # If list type, ensure values are lists or None; Arrow will validate element types
        if pa.types.is_list(t):
            # Normalize python Nones / lists
            cleaned = []
            for v in values:
                if v is None:
                    cleaned.append(None)
                elif isinstance(v, list):
                    cleaned.append(v)
                else:
                    # last attempt: parse JSON list
                    if isinstance(v, str) and v.strip().startswith("["):
                        try:
                            cleaned.append(json.loads(v))
                        except Exception:
                            cleaned.append(None)
                    else:
                        cleaned.append(None)
            return pa.array(cleaned, type=t)
        # Scalar: try direct cast
        if pa.types.is_string(t):
            return pa.array([None if v is None else str(v) for v in values], type=t)
        if pa.types.is_floating(t):
            return pa.array([None if v is None else float(v) for v in values], type=t)
        if pa.types.is_integer(t):
            return pa.array([None if v is None else int(v) for v in values], type=t)
        if pa.types.is_boolean(t):
            return pa.array([None if v is None else bool(v) for v in values], type=t)
        # Fallback
        return pa.array(values, type=t)

    # New column: infer
    ex = _first_non_null(values)
    if isinstance(ex, list):
        # numeric lists default to float32 unless dtype_opt points elsewhere
        t = _arrow_type_for_new_numeric_list(dtype_opt)
        return pa.array(values, type=t)
    # strings or scalars → string
    return pa.array(
        [None if v is None else _coerce_scalar_to_string(v) for v in values],
        type=pa.string(),
    )


def execute_merge(
    ds: Dataset, plan: MergePlan, cfg: MergeConfig, assume_yes: bool = False
) -> MergeOutcome:
    if not plan.mapped_ok:
        raise ValidationError(
            "Type/shape validation failed:\n  - "
            + "\n  - ".join(plan.mapped_mismatch_messages)
        )

    # Prompt
    if not assume_yes:
        print(
            "\n" + "Proceed to write? Type 'yes' to continue, anything else to abort: ",
            end="",
            flush=True,
        )
        ans = input().strip()
        if ans != "yes":
            raise ValidationError("User aborted (expected literal 'yes').")

    # Build Arrow table to append: essential columns + mapped/carry present in plan.batch_df
    batch_df = plan.batch_df if plan.batch_df is not None else pd.DataFrame()
    if batch_df.empty or plan.will_add_rows == 0:
        # No new rows to add; nothing to do
        print("Nothing to add.")
        return MergeOutcome(added=0, total_after=plan.dest_row_count_before)

    # Ensure essential columns exist
    for col, _ in REQUIRED_COLUMNS:
        if col not in batch_df.columns:
            batch_df[col] = []

    # Read destination once more
    t0 = read_parquet(ds.records_path)
    refs = _get_existing_column_refs(t0)

    # Compose arrow table for batch
    arrays: List[pa.Array] = []
    fields: List[pa.Field] = []
    for name, typ in REQUIRED_COLUMNS:
        if name == "created_at":
            # ensure tz-aware ns (arrow timestamp(us, UTC))
            arr = pa.array(
                [pd.Timestamp(v).to_pydatetime() for v in batch_df[name].tolist()],
                type=pa.timestamp("us", tz="UTC"),
            )
        elif name == "length":
            arr = pa.array([int(v) for v in batch_df[name].tolist()], type=pa.int32())
        else:
            arr = pa.array(batch_df[name].tolist(), type=typ)
        arrays.append(arr)
        fields.append(pa.field(name, arr.type, nullable=False))

    # mapped & carry columns
    for name in plan.column_targets:
        vals = (
            batch_df[name].tolist()
            if name in batch_df.columns
            else [None] * len(batch_df)
        )
        arr = _to_arrow_array_for_column(name, vals, t0, refs, dtype_opt=None)
        arrays.append(arr)
        fields.append(pa.field(name, arr.type, nullable=True))

    batch = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

    # Append (schema-aware) and write atomically
    combined = pa.concat_tables([t0, batch], promote_options="default")
    write_parquet_atomic(
        combined, ds.records_path, ds.snapshot_dir, preserve_metadata_from=t0
    )

    # Log event (succinct JSON line)
    append_event(
        ds.events_path,
        {
            "action": "merge_from_pt",
            "dataset": ds.name,
            "config": str(plan.cfg_path),
            "sources": [
                {"path": str(p), "entries": n} for (p, n, _lbl) in plan.sources_brief
            ],
            "proposed": plan.proposed,
            "added": plan.will_add_rows,
            "skipped_existing": plan.skipped_existing,
            "skipped_id_conflict": plan.skipped_id_conflict,
            "screen_non_acgt": plan.screen_non_acgt,
            "screen_len_mismatch": plan.screen_len_mismatch,
            "columns": plan.column_targets,
        },
    )

    return MergeOutcome(added=plan.will_add_rows, total_after=combined.num_rows)
