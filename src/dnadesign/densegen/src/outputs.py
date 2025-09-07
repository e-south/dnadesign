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
from pathlib import Path
from typing import Dict, Iterable

from .usr_adapter import USRWriter


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
    def add(
        self, sequence: str, meta: Dict, source_label: str
    ) -> None:  # pragma: no cover - abstract
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
    kind = (cfg.get("output", {}).get("kind") or "usr").lower()
    sinks: list[SinkBase] = []

    if kind in {"usr", "both"}:
        usr_cfg = cfg.get("usr", {}) or cfg.get("output", {}).get("usr", {})
        writer = USRWriter(
            dataset=usr_cfg.get("dataset", "densegen_sequences"),
            root=Path(usr_cfg["root"]).resolve() if usr_cfg.get("root") else None,
            namespace=usr_cfg.get("namespace", "densegen"),
            chunk_size=int(usr_cfg.get("chunk_size", 128)),
            allow_overwrite=bool(usr_cfg.get("allow_overwrite", True)),
        )
        sinks.append(USRSink(writer))

    if kind in {"jsonl", "both"}:
        jl_cfg = cfg.get("output", {}).get("jsonl", {})
        path = jl_cfg.get("path", "outputs/densegen.jsonl")
        ns = (cfg.get("usr", {}) or {}).get("namespace", "densegen")
        sinks.append(JSONLSink(path=path, namespace=ns))

    if not sinks:
        # default to USR if misconfigured
        usr_cfg = cfg.get("usr", {}) or {}
        writer = USRWriter(
            dataset=usr_cfg.get("dataset", "densegen_sequences"),
            root=Path(usr_cfg["root"]).resolve() if usr_cfg.get("root") else None,
            namespace=usr_cfg.get("namespace", "densegen"),
            chunk_size=int(usr_cfg.get("chunk_size", 128)),
            allow_overwrite=bool(usr_cfg.get("allow_overwrite", True)),
        )
        sinks.append(USRSink(writer))

    return sinks
