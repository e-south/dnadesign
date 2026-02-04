"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/io/parsers/meme.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import numpy as np
from Bio import motifs

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.io.parsers.backend import register

_META_KV_RE = re.compile(r"(\w+)\s*=\s*([0-9.eE+-]+)")
_BLOCK_SITE_RE = re.compile(r"^(?P<name>\S+)\s*\(\s*(?P<start>-?\d+)\s*\)\s+(?P<seq>[A-Za-z]+)")
_PVAL_RE = re.compile(r"^(?P<name>\S+)\s+(?P<start>-?\d+)\s+(?P<pval>[0-9.eE+-]+)")
_ALPHABET_RE = re.compile(r"^ALPHABET\s*=\s*(.+)$", re.IGNORECASE)
_VERSION_RE = re.compile(r"^MEME\s+version\s+(?P<version>.+)$", re.IGNORECASE)
_TRAINING_RE = re.compile(r"^Training set(?:\s+file)?\s*:\s*(?P<path>.+)$", re.IGNORECASE)
_CMD_RE = re.compile(r"^command line\s*:\s*(?P<cmd>.+)$", re.IGNORECASE)
_BG_FREQ_RE = re.compile(
    r"A\s+([0-9.eE+-]+)\s+C\s+([0-9.eE+-]+)\s+G\s+([0-9.eE+-]+)\s+T\s+([0-9.eE+-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MemeFileMeta:
    version: Optional[str] = None
    command_line: Optional[str] = None
    training_set: Optional[str] = None
    alphabet: Optional[str] = None
    background_freqs: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True, slots=True)
class BlockSite:
    sequence_name: str
    start: int
    sequence: str
    weight: Optional[float] = None
    strand: Optional[str] = None
    pvalue: Optional[float] = None


@dataclass(frozen=True, slots=True)
class MemeMotif:
    motif_id: str
    motif_name: str
    motif_label: str
    ordinal: Optional[int]
    width: Optional[int]
    nsites: Optional[int]
    evalue: Optional[float]
    llr: Optional[float]
    prob_matrix: list[list[float]]
    log_odds_matrix: Optional[list[list[float]]]
    block_sites: list[BlockSite]


@dataclass(frozen=True, slots=True)
class MemeFileParseResult:
    motifs: list[MemeMotif]
    meta: MemeFileMeta


def parse_meme_file(path: Path) -> MemeFileParseResult:
    return parse_meme_content(path.read_text(), path)


def parse_meme_content(text: str, path: Path) -> MemeFileParseResult:
    if _looks_like_xml(text):
        return _parse_meme_xml_text(text, path)
    return parse_meme_text(text, path)


def select_meme_motif(
    result: MemeFileParseResult,
    *,
    file_stem: str,
    selector: Optional[object] = None,
    path: Optional[Path] = None,
) -> MemeMotif:
    if not result.motifs:
        raise ValueError("No MEME motifs found in file.")

    if selector is not None:
        if isinstance(selector, str) and selector.strip().lower() == "name_match":
            selector = None
        else:
            return _select_by_selector(result.motifs, selector, path=path)

    stem = file_stem.strip().lower()
    if stem:
        matches = [m for m in result.motifs if _motif_name_matches(m, stem)]
        if len(matches) > 1:
            raise ValueError(_render_selection_error("Ambiguous MEME motif selection", matches, path=path))
        if len(matches) == 1:
            return matches[0]

    meme1 = [m for m in result.motifs if m.motif_id.strip().lower() == "meme-1"]
    if len(meme1) == 1:
        return meme1[0]
    if len(result.motifs) == 1:
        return result.motifs[0]
    return result.motifs[0]


@register("MEME")
def parse_meme(path: Path) -> PWM:
    """
    Parse a MEME-format PWM (XML or text) and always
    use the file's own stem (preserving case) as PWM.name.
    """
    result = parse_meme_file(path)
    motif = select_meme_motif(result, file_stem=path.stem, path=path)
    prob_mat = np.array(motif.prob_matrix, dtype="float32")
    logodds_mat = np.array(motif.log_odds_matrix, dtype="float32") if motif.log_odds_matrix else None
    return PWM(
        name=path.stem,
        matrix=prob_mat,
        nsites=motif.nsites,
        evalue=motif.evalue,
        log_odds_matrix=logodds_mat,
    )


def _looks_like_xml(text: str) -> bool:
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        return raw.startswith("<")
    return False


def _parse_meme_xml_text(text: str, path: Path) -> MemeFileParseResult:
    try:
        handle = StringIO(text)
        mlist = motifs.parse(handle, "MEME")
    except (ValueError, ElementTree.ParseError) as exc:
        raise ValueError(f"Failed to parse MEME XML '{path}': {exc}") from exc
    if not mlist:
        raise ValueError(f"No motifs parsed from MEME XML '{path}'.")

    motifs_out: list[MemeMotif] = []
    for idx, motif in enumerate(mlist, start=1):
        prob_mat = np.array(motif.pwm).T.astype(float).tolist()
        name = getattr(motif, "name", None) or getattr(motif, "id", None) or f"MEME-{idx}"
        motif_id = str(name)
        motif_name = str(name)
        motifs_out.append(
            MemeMotif(
                motif_id=motif_id,
                motif_name=motif_name,
                motif_label=motif_name,
                ordinal=idx,
                width=len(prob_mat),
                nsites=getattr(motif, "num_occurrences", None),
                evalue=getattr(motif, "evalue", None),
                llr=None,
                prob_matrix=prob_mat,
                log_odds_matrix=None,
                block_sites=[],
            )
        )

    return MemeFileParseResult(motifs=motifs_out, meta=MemeFileMeta())


def parse_meme_text(text: str, path: Path) -> MemeFileParseResult:
    meta = MemeFileMeta()
    motifs_out: list[MemeMotif] = []
    current: dict[str, object] | None = None
    state: Optional[str] = None
    bg_next = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()

        if not stripped:
            if state in {"prob_matrix", "log_odds", "blocks", "pvalues"}:
                state = None
            if bg_next:
                bg_next = False
            continue
        if stripped == "//":
            if state in {"prob_matrix", "log_odds", "blocks", "pvalues"}:
                state = None
            continue

        version_match = _VERSION_RE.match(stripped)
        if version_match:
            meta = _replace_meta(meta, version=version_match.group("version").strip())
            continue

        cmd_match = _CMD_RE.match(stripped)
        if cmd_match:
            meta = _replace_meta(meta, command_line=cmd_match.group("cmd").strip())
            continue

        train_match = _TRAINING_RE.match(stripped)
        if train_match:
            meta = _replace_meta(meta, training_set=train_match.group("path").strip())
            continue

        alpha_match = _ALPHABET_RE.match(stripped)
        if alpha_match:
            alphabet = alpha_match.group(1).strip().replace(" ", "")
            if alphabet.upper() != "ACGT":
                raise ValueError(f"Unsupported MEME alphabet '{alphabet}' in {path}. Expected ACGT.")
            meta = _replace_meta(meta, alphabet=alphabet.upper())
            continue

        if "background letter frequencies" in lower:
            bg_next = True
            continue

        if bg_next:
            freq_match = _BG_FREQ_RE.search(stripped)
            if freq_match:
                freqs = tuple(float(val) for val in freq_match.groups())
                meta = _replace_meta(meta, background_freqs=freqs)
            bg_next = False
            continue

        if stripped.startswith("MOTIF "):
            if current is not None:
                motifs_out.append(_finalize_motif(current, path))
            current = _init_motif(stripped)
            state = None
            continue

        if lower.startswith("letter-probability matrix"):
            if current is None:
                raise ValueError(f"Found letter-probability matrix before MOTIF in {path}.")
            _update_matrix_meta(current, stripped, path)
            state = "prob_matrix"
            continue

        if lower.startswith("log-odds matrix"):
            if current is None:
                raise ValueError(f"Found log-odds matrix before MOTIF in {path}.")
            _update_matrix_meta(current, stripped, path, log_odds=True)
            state = "log_odds"
            continue

        if "sites sorted by position" in lower and "p-value" in lower:
            if current is None:
                raise ValueError(f"Found sites table before MOTIF in {path}.")
            state = "pvalues"
            continue

        if "blocks format" in lower:
            if current is None:
                raise ValueError(f"Found BLOCKS section before MOTIF in {path}.")
            state = "blocks"
            continue

        if state == "prob_matrix":
            if stripped.startswith("---"):
                continue
            current["prob_rows"].append(_parse_matrix_row(stripped, path))
            continue

        if state == "log_odds":
            if stripped.startswith("---"):
                continue
            current["logodds_rows"].append(_parse_matrix_row(stripped, path))
            continue

        if state == "pvalues":
            if stripped.startswith("---") or lower.startswith("sequence"):
                continue
            match = _PVAL_RE.match(stripped)
            if match:
                key = (match.group("name"), int(match.group("start")))
                pval = float(match.group("pval"))
                _store_pvalue(current, key, pval)
            continue

        if state == "blocks":
            if stripped.startswith("---") or lower.startswith("sequence"):
                continue
            if lower.startswith("bl") and "motif" in lower:
                continue
            match = _BLOCK_SITE_RE.match(stripped)
            if not match:
                raise ValueError(f"Unrecognized BLOCKS line in {path}: '{stripped}'")
            _add_block_site(
                current,
                match.group("name"),
                int(match.group("start")),
                match.group("seq"),
                path,
            )
            continue

    if current is not None:
        motifs_out.append(_finalize_motif(current, path))

    if not motifs_out:
        raise ValueError(f"No motifs parsed from MEME file {path}.")

    return MemeFileParseResult(motifs=motifs_out, meta=meta)


def _replace_meta(meta: MemeFileMeta, **kwargs) -> MemeFileMeta:
    return MemeFileMeta(
        version=kwargs.get("version", meta.version),
        command_line=kwargs.get("command_line", meta.command_line),
        training_set=kwargs.get("training_set", meta.training_set),
        alphabet=kwargs.get("alphabet", meta.alphabet),
        background_freqs=kwargs.get("background_freqs", meta.background_freqs),
    )


def _init_motif(line: str) -> dict[str, object]:
    label = line.split(None, 1)[1].strip() if " " in line else line
    parts = label.split()
    motif_id = parts[0] if parts else label
    motif_name = " ".join(parts[1:]).strip() if len(parts) > 1 else motif_id
    ordinal = int(parts[0]) if parts and parts[0].isdigit() else None
    if ordinal is not None and len(parts) > 1:
        motif_id = parts[1]
        motif_name = " ".join(parts[2:]).strip() if len(parts) > 2 else motif_id
    return {
        "motif_id": motif_id,
        "motif_name": motif_name or motif_id,
        "motif_label": label,
        "ordinal": ordinal,
        "width": None,
        "nsites": None,
        "evalue": None,
        "llr": None,
        "prob_rows": [],
        "logodds_rows": [],
        "block_sites": [],
        "pvalues": {},
    }


def _update_matrix_meta(current: dict[str, object], line: str, path: Path, log_odds: bool = False) -> None:
    meta = {k.lower(): v for k, v in _META_KV_RE.findall(line)}
    if "alength" in meta:
        if int(float(meta["alength"])) != 4:
            raise ValueError(f"Unsupported MEME alength={meta['alength']} in {path}. Expected 4.")
    if "w" in meta:
        width = int(float(meta["w"]))
        existing = current.get("width")
        if existing is not None and int(existing) != width:
            raise ValueError(f"Conflicting MEME motif widths in {path}: {existing} vs {width}.")
        current["width"] = width
    if "nsites" in meta:
        current["nsites"] = int(float(meta["nsites"]))
    if "e" in meta:
        current["evalue"] = float(meta["e"])
    if "llr" in meta and not log_odds:
        current["llr"] = float(meta["llr"])


def _parse_matrix_row(line: str, path: Path) -> list[float]:
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid MEME matrix row in {path}: '{line}'")
    return [float(val) for val in parts[-4:]]


def _store_pvalue(current: dict[str, object], key: tuple[str, int], pval: float) -> None:
    pvalues: dict[tuple[str, int], Optional[float]] = current["pvalues"]  # type: ignore[assignment]
    if key in pvalues and pvalues[key] != pval:
        pvalues[key] = None
        return
    pvalues[key] = pval


def _add_block_site(
    current: dict[str, object],
    name: str,
    start: int,
    seq: str,
    path: Path,
) -> None:
    seq_norm = seq.strip().upper()
    if any(ch not in "ACGT" for ch in seq_norm):
        raise ValueError(f"Non-ACGT characters found in MEME BLOCKS site '{seq}' in {path}.")
    block_sites: list[BlockSite] = current["block_sites"]  # type: ignore[assignment]
    block_sites.append(BlockSite(sequence_name=name, start=start, sequence=seq_norm))


def _finalize_motif(current: dict[str, object], path: Path) -> MemeMotif:
    prob_rows: list[list[float]] = current["prob_rows"]  # type: ignore[assignment]
    if not prob_rows:
        raise ValueError(f"Missing letter-probability matrix in {path}.")

    width = current.get("width") or len(prob_rows)
    if len(prob_rows) != width:
        raise ValueError(f"Parsed {len(prob_rows)} rows for width {width} in {path}.")

    logodds_rows: list[list[float]] = current["logodds_rows"]  # type: ignore[assignment]
    if logodds_rows and len(logodds_rows) != width:
        raise ValueError(f"Log-odds rows length mismatch in {path}: {len(logodds_rows)} vs {width}.")

    block_sites: list[BlockSite] = current["block_sites"]  # type: ignore[assignment]
    if block_sites:
        for site in block_sites:
            if len(site.sequence) != width:
                raise ValueError(
                    f"BLOCKS site length {len(site.sequence)} does not match motif width {width} in {path}."
                )

    pvalues: dict[tuple[str, int], Optional[float]] = current["pvalues"]  # type: ignore[assignment]
    if pvalues and block_sites:
        enriched: list[BlockSite] = []
        for site in block_sites:
            key = (site.sequence_name, site.start)
            enriched.append(
                BlockSite(
                    sequence_name=site.sequence_name,
                    start=site.start,
                    sequence=site.sequence,
                    weight=site.weight,
                    strand=site.strand,
                    pvalue=pvalues.get(key),
                )
            )
        block_sites = enriched

    return MemeMotif(
        motif_id=str(current["motif_id"]),
        motif_name=str(current["motif_name"]),
        motif_label=str(current["motif_label"]),
        ordinal=current.get("ordinal"),
        width=int(width),
        nsites=current.get("nsites"),
        evalue=current.get("evalue"),
        llr=current.get("llr"),
        prob_matrix=prob_rows,
        log_odds_matrix=logodds_rows or None,
        block_sites=block_sites,
    )


def _motif_name_matches(motif: MemeMotif, stem: str) -> bool:
    return motif.motif_name.lower() == stem or motif.motif_id.lower() == stem


def _select_by_selector(motifs: list[MemeMotif], selector: object, *, path: Optional[Path]) -> MemeMotif:
    if isinstance(selector, int):
        return _select_by_index(motifs, selector, path)
    selector_str = str(selector).strip()
    if not selector_str:
        raise ValueError("meme_motif_selector must be non-empty when provided.")
    if selector_str.lower() == "name_match":
        raise ValueError("meme_motif_selector='name_match' requires file-stem selection context.")
    if selector_str.isdigit():
        return _select_by_index(motifs, int(selector_str), path)
    matches = [m for m in motifs if _selector_matches(m, selector_str)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(_render_selection_error("Ambiguous MEME motif selector", matches, path=path))
    raise ValueError(_render_selection_error(f"No MEME motif matches selector '{selector_str}'", motifs, path=path))


def _selector_matches(motif: MemeMotif, selector: str) -> bool:
    needle = selector.lower()
    return motif.motif_id.lower() == needle or motif.motif_name.lower() == needle or motif.motif_label.lower() == needle


def _select_by_index(motifs: list[MemeMotif], index: int, path: Optional[Path]) -> MemeMotif:
    if index < 1 or index > len(motifs):
        raise ValueError(_render_selection_error(f"MEME motif index {index} out of range", motifs, path=path))
    return motifs[index - 1]


def _render_selection_error(message: str, motifs: list[MemeMotif], *, path: Optional[Path]) -> str:
    header = f"{message}"
    if path is not None:
        header = f"{message} ({path})."
    lines = [
        header,
        "Available motifs:",
        "Hint: select a motif by name, label, or index.",
    ]
    for idx, motif in enumerate(motifs, start=1):
        label = motif.motif_label
        lines.append(f"  [{idx}] {label}")
    return "\n".join(lines)
