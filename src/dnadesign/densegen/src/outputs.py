"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/densegen/src/outputs.py

Pluggable output sinks for DenseGen:
- USRSink: writes to USR via USRWriter (namespacing handled by Dataset.attach)
- JSONLSink: writes newline-delimited JSON with essential + namespaced derived fields

Both sinks implement: add(sequence: str, meta: dict, source_label: str) and flush().

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import pyarrow.parquet as pq

from .usr_adapter import USRWriter

log = logging.getLogger(__name__)


def _sha256_id(bio_type: str, alphabet: str, sequence: str) -> str:
    # Fallback ID; not guaranteed to match USR's compute_id, but deterministic.
    h = hashlib.sha256()
    h.update(f"{bio_type}|{alphabet}|{sequence}".encode("utf-8"))
    return h.hexdigest()


def _namespace_meta(meta: Dict, namespace: str) -> Dict:
    # Flatten meta into namespaced keys like USR.attach would create (densegen__*)
    out = {}
    for k, v in meta.items():
        nk = f"{namespace}__{k}" if not k.startswith(f"{namespace}__") else k
        out[nk] = v
    return out


class SinkBase:
    def add(self, sequence: str, meta: Dict, source_label: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class USRSink(SinkBase):
    def __init__(self, writer: USRWriter):
        self.writer = writer

    def add(self, sequence: str, meta: Dict, source_label: str) -> None:
        # meta stays un-namespaced; USRWriter/Dataset.attach applies the namespace
        self.writer.add(sequence, meta, source_label)

    def flush(self) -> None:
        self.writer.flush()


class JSONLSink(SinkBase):
    def __init__(
        self,
        path: str,
        *,
        namespace: str = "densegen",
        bio_type: str = "dna",
        alphabet: str = "dna_4",
        deduplicate: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.bio_type = bio_type
        self.alphabet = alphabet
        self.deduplicate = deduplicate
        self._seen_ids = set()
        # Preload existing ids if the file exists (for resume-safety)
        if self.path.exists() and self.deduplicate:
            try:
                with self.path.open("r") as f:
                    for line in f:
                        j = json.loads(line)
                        if "id" in j:
                            self._seen_ids.add(j["id"])
            except Exception:
                # If unreadable, continue from scratch (best-effort)
                pass
        self._buf: list[dict] = []

    def add(self, sequence: str, meta: Dict, source_label: str) -> None:
        rid = _sha256_id(self.bio_type, self.alphabet, sequence)
        if self.deduplicate and rid in self._seen_ids:
            return
        self._seen_ids.add(rid)

        derived_ns = _namespace_meta(meta, self.namespace)
        row = {
            "id": rid,
            "sequence": sequence,
            "bio_type": self.bio_type,
            "alphabet": self.alphabet,
            "source": source_label,
            **derived_ns,
        }
        self._buf.append(row)

        if len(self._buf) >= 128:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        with self.path.open("a") as f:
            for r in self._buf:
                f.write(json.dumps(r) + "\n")
        self._buf.clear()


def build_sinks(cfg: dict) -> Iterable[SinkBase]:
    """
    Create one or more sinks based on config:
      output.kind: usr | jsonl | both
      output.jsonl.path: outputs/densegen.jsonl
      usr: {dataset, root, namespace, ...}
    """
    out_cfg = cfg.get("output", {})
    if not out_cfg:
        raise ValueError("`output` section is required (usr | jsonl | both).")
    kind = (out_cfg.get("kind") or "").lower()
    if kind not in {"usr", "jsonl", "both"}:
        raise ValueError("`output.kind` must be one of: usr | jsonl | both.")
    sinks: list[SinkBase] = []

    if kind in {"usr", "both"}:
        usr_cfg = cfg.get("usr", {}) or out_cfg.get("usr", {})
        if not usr_cfg or "dataset" not in usr_cfg:
            raise ValueError("USR sink requested but `usr.dataset` not provided.")
        writer = USRWriter(
            dataset=usr_cfg.get("dataset", "densegen_sequences"),
            root=Path(usr_cfg["root"]).resolve() if usr_cfg.get("root") else None,
            namespace=usr_cfg.get("namespace", "densegen"),
            chunk_size=int(usr_cfg.get("chunk_size", 128)),
            allow_overwrite=bool(usr_cfg.get("allow_overwrite", True)),
        )
        sinks.append(USRSink(writer))

    if kind in {"jsonl", "both"}:
        jl_cfg = out_cfg.get("jsonl", {})
        if not jl_cfg or "path" not in jl_cfg:
            raise ValueError("JSONL sink requested but `output.jsonl.path` not provided.")
        path = jl_cfg.get("path", "outputs/densegen.jsonl")
        ns = (cfg.get("usr", {}) or {}).get("namespace", "densegen")
        sinks.append(JSONLSink(path=path, namespace=ns))

    if not sinks:
        raise AssertionError("No output sinks created; check `output.kind` and related settings.")

    return sinks


def _maybe_json_load(val):
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val


def load_records_from_config(cfg: dict) -> Tuple[pd.DataFrame, str]:
    """
    Load output records from either JSONL (preferred if present) or USR dataset.
    Returns (df, source_label), where source_label is 'jsonl:<path>' or 'usr:<dataset>'.
    No silent fallbacks: if a configured sink is missing on disk, raises with specifics.
    """
    out_cfg = cfg.get("output", {})
    kind = (out_cfg.get("kind") or "").lower()
    if kind not in {"usr", "jsonl", "both"}:
        raise ValueError("`output.kind` must be one of: usr | jsonl | both.")

    dfs: list[pd.DataFrame] = []
    source_label = ""

    if kind in {"jsonl", "both"}:
        jl_path = out_cfg.get("jsonl", {}).get("path")
        if jl_path:
            p = Path(jl_path).resolve()
            if not p.exists():
                raise FileNotFoundError(f"JSONL path does not exist: {p}")
            rows = []
            with p.open("r") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            if rows:
                df = pd.DataFrame(rows)
                # Coerce likely JSON-encoded columns back to Python objects
                for col in list(df.columns):
                    df[col] = df[col].map(_maybe_json_load)
                dfs.append(df)
                source_label = f"jsonl:{p}"

    if kind in {"usr", "both"} and not dfs:
        usr_cfg = cfg.get("usr", {}) or out_cfg.get("usr", {})
        if not usr_cfg or "dataset" not in usr_cfg:
            raise ValueError("USR output requested but `usr.dataset` not set.")
        from .usr_adapter import Dataset  # type: ignore[attr-defined]

        # Access underlying dataset (same logic as USRWriter)
        ds = Dataset(
            Path(usr_cfg.get("root") or Path(__file__).resolve().parents[2] / "usr" / "datasets"),
            usr_cfg["dataset"],
        )
        rp = ds.records_path
        if not rp.exists():
            raise FileNotFoundError(f"USR records not found at: {rp}")
        tbl = pq.read_table(rp)
        df = tbl.to_pandas()
        # best-effort JSON coercion for namespaced columns
        for col in [c for c in df.columns if "__" in c]:
            df[col] = df[col].map(_maybe_json_load)
        dfs.append(df)
        source_label = f"usr:{usr_cfg['dataset']}"

    if not dfs:
        raise RuntimeError("Could not load any outputs; ensure either JSONL or USR sink produced data.")
    return dfs[0], source_label
