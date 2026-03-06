"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/cli_ingest.py

CLI ingest request builders for extract and generate command inputs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from .config import IngestConfig
from .errors import ConfigError
from .input_parsing import load_nonempty_lines, read_ids_arg


@dataclass(frozen=True)
class CliIngestRequest:
    ingest: IngestConfig
    inputs: Any


def build_extract_ingest(
    *,
    seq: Optional[List[str]],
    seq_file: Optional[Path],
    usr: Optional[str],
    field: str,
    ids: Optional[str],
    usr_root: Optional[Path],
    pt: Optional[Path],
    records_jsonl: Optional[Path],
    i_know_this_is_pickle: bool,
    guard_pickle: Callable[[bool], None],
) -> CliIngestRequest:
    if usr:
        return CliIngestRequest(
            ingest=IngestConfig(
                source="usr",
                dataset=usr,
                field=field,
                root=(usr_root.as_posix() if usr_root else None),
                ids=read_ids_arg(ids),
            ),
            inputs=None,
        )

    if pt:
        guard_pickle(i_know_this_is_pickle)
        return CliIngestRequest(
            ingest=IngestConfig(source="pt_file", field=field),
            inputs=pt.as_posix(),
        )

    if records_jsonl:
        records = [json.loads(ln) for ln in load_nonempty_lines(records_jsonl)]
        return CliIngestRequest(
            ingest=IngestConfig(source="records", field=field),
            inputs=records,
        )

    if seq_file:
        return CliIngestRequest(
            ingest=IngestConfig(source="sequences"),
            inputs=load_nonempty_lines(seq_file),
        )

    if seq:
        return CliIngestRequest(
            ingest=IngestConfig(source="sequences"),
            inputs=seq,
        )

    raise ConfigError("Provide one of --seq/--seq-file/--usr/--pt/--records-jsonl")


def build_generate_ingest(
    *,
    prompt: Optional[List[str]],
    prompt_file: Optional[Path],
    usr: Optional[str],
    field: str,
    ids: Optional[str],
    usr_root: Optional[Path],
) -> CliIngestRequest:
    if usr:
        return CliIngestRequest(
            ingest=IngestConfig(
                source="usr",
                dataset=usr,
                field=field,
                root=(usr_root.as_posix() if usr_root else None),
                ids=read_ids_arg(ids),
            ),
            inputs=None,
        )

    if prompt_file:
        return CliIngestRequest(
            ingest=IngestConfig(source="sequences"),
            inputs=load_nonempty_lines(prompt_file),
        )

    if prompt:
        return CliIngestRequest(
            ingest=IngestConfig(source="sequences"),
            inputs=prompt,
        )

    raise ConfigError("Provide prompts via --prompt/--prompt-file or use --usr.")
