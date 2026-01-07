"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/ingest/adapters/local.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Union

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.ingest.models import (
    DatasetDescriptor,
    DatasetQuery,
    MotifDescriptor,
    MotifQuery,
    MotifRecord,
    OrganismRef,
    Provenance,
    SiteInstance,
    SiteQuery,
)
from dnadesign.cruncher.ingest.normalize import build_motif_record, normalize_site_sequence
from dnadesign.cruncher.io.parsers.backend import load_pwm
from dnadesign.cruncher.io.parsers.meme import (
    MemeFileParseResult,
    MemeMotif,
    parse_meme_content,
    select_meme_motif,
)


@dataclass(frozen=True, slots=True)
class LocalMotifAdapterConfig:
    source_id: str
    root: Path
    patterns: Sequence[str]
    recursive: bool = False
    format_map: dict[str, str] = field(default_factory=dict)
    default_format: Optional[str] = None
    tf_name_strategy: str = "stem"
    matrix_semantics: str = "probabilities"
    organism: Optional[OrganismRef] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    extra_parser_modules: Sequence[str] = ()
    extract_sites: bool = False
    meme_motif_selector: Optional[Union[str, int]] = None


class LocalMotifAdapter:
    """Ingest motif matrices from a local directory of motif files."""

    def __init__(self, config: LocalMotifAdapterConfig) -> None:
        self.source_id = config.source_id
        self._config = config
        self._root = config.root
        if not self._root.exists():
            raise FileNotFoundError(f"Local motif root does not exist: {self._root}")
        if not self._root.is_dir():
            raise NotADirectoryError(f"Local motif root is not a directory: {self._root}")
        self._format_map = {k.lower(): v.upper() for k, v in config.format_map.items()}
        self._default_format = config.default_format.upper() if config.default_format else None
        self._paths = self._discover_paths()
        self._index = self._index_paths(self._paths)
        self._extract_sites = bool(config.extract_sites)
        self._meme_motif_selector = config.meme_motif_selector
        self._meme_cache: dict[Path, tuple[MemeFileParseResult, str]] = {}

    def capabilities(self) -> Set[str]:
        caps = {"motifs:list", "motifs:get", "motifs:iter"}
        if self._extract_sites:
            caps.add("sites:list")
        return caps

    def list_motifs(self, query: MotifQuery) -> list[MotifDescriptor]:
        return list(self._iter_descriptors(query))

    def iter_motifs(self, query: MotifQuery, *, page_size: int = 200) -> Iterable[MotifDescriptor]:
        del page_size
        return self._iter_descriptors(query)

    def get_motif(self, motif_id: str) -> MotifRecord:
        if not motif_id or not motif_id.strip():
            raise ValueError("motif_id must be non-empty for local motif retrieval")
        key = motif_id.strip().lower()
        path = self._index.get(key)
        if path is None:
            options = ", ".join(sorted({p.stem for p in self._paths}))
            raise FileNotFoundError(
                f"Motif id '{motif_id}' not found under {self._root}. Available: {options or 'none'}."
            )
        tf_name = self._tf_name_for_path(path)
        raw_paths = [self._relativize_path(path)]
        if self._format_for_path(path) == "MEME":
            self._ensure_meme_probabilities(path)
            result, raw_payload = self._load_meme(path)
            motif = self._select_meme_motif(result, path)
            tags = self._meme_tags(result, motif)
            return build_motif_record(
                source=self.source_id,
                motif_id=path.stem,
                tf_name=tf_name,
                matrix=motif.prob_matrix,
                log_odds_matrix=motif.log_odds_matrix,
                matrix_semantics=self._config.matrix_semantics,
                organism=self._config.organism,
                raw_payload=raw_payload,
                source_version=self._config.source_version,
                source_url=self._config.source_url,
                license=self._config.license,
                citation=self._config.citation,
                raw_artifact_paths=raw_paths,
                tags=tags,
                background=result.meta.background_freqs,
            )
        pwm = self._load_pwm(path)
        raw_payload = path.read_text()
        tags = {"matrix_source": "file"}
        tags.update(self._config.tags)
        return build_motif_record(
            source=self.source_id,
            motif_id=path.stem,
            tf_name=tf_name,
            matrix=pwm.matrix,
            matrix_semantics=self._config.matrix_semantics,
            organism=self._config.organism,
            raw_payload=raw_payload,
            source_version=self._config.source_version,
            source_url=self._config.source_url,
            license=self._config.license,
            citation=self._config.citation,
            raw_artifact_paths=raw_paths,
            tags=tags,
        )

    def list_sites(self, query: SiteQuery) -> Iterable[SiteInstance]:
        if not self._extract_sites:
            raise ValueError(
                "Local motif sources do not provide binding sites. "
                "Set ingest.local_sources[].extract_sites=true to enable MEME BLOCKS extraction."
            )
        if not query.motif_id and not query.tf_name:
            raise ValueError("site queries require motif_id or tf_name for local sources")
        paths = self._resolve_site_paths(query)
        return self._iter_sites(paths, query)

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery) -> Iterable[SiteInstance]:
        if not self._extract_sites:
            raise ValueError(
                "Local motif sources do not provide binding sites. "
                "Set ingest.local_sources[].extract_sites=true to enable MEME BLOCKS extraction."
            )
        site_query = SiteQuery(
            organism=query.organism,
            motif_id=motif_id,
            tf_name=query.tf_name,
            dataset_id=query.dataset_id,
            region=query.region,
            limit=query.limit,
        )
        paths = self._resolve_site_paths(site_query)
        return self._iter_sites(paths, site_query)

    def list_datasets(self, query: DatasetQuery) -> list[DatasetDescriptor]:
        del query
        return []

    def _iter_descriptors(self, query: MotifQuery) -> Iterable[MotifDescriptor]:
        paths = self._paths
        if query.tf_name:
            tf_query = query.tf_name.strip().lower()
            if not tf_query:
                raise ValueError("tf_name must be non-empty for motif search")
            paths = [path for path in paths if self._tf_name_for_path(path).lower() == tf_query]
        if query.regex:
            regex = query.regex.strip()
            if not regex:
                raise ValueError("regex must be non-empty for motif search")
            try:
                pattern = re.compile(regex)
            except re.error as exc:
                raise ValueError(f"Invalid regex '{query.regex}': {exc}") from exc
            paths = [path for path in paths if pattern.search(self._tf_name_for_path(path))]
        if query.limit is not None:
            paths = paths[: max(int(query.limit), 0)]
        for path in paths:
            tf_name = self._tf_name_for_path(path)
            fmt = self._format_for_path(path)
            if fmt == "MEME":
                self._ensure_meme_probabilities(path)
                result, _ = self._load_meme(path)
                motif = self._select_meme_motif(result, path)
                pwm = PWM(name=path.stem, matrix=motif.prob_matrix)
            else:
                pwm = self._load_pwm(path)
            tags = {"matrix_source": "file"}
            tags.update(self._config.tags)
            yield MotifDescriptor(
                source=self.source_id,
                motif_id=path.stem,
                tf_name=tf_name,
                organism=self._config.organism,
                length=pwm.length,
                kind="PFM" if self._config.matrix_semantics == "probabilities" else "PWM",
                tags=tags,
            )

    def _discover_paths(self) -> List[Path]:
        patterns = [pat for pat in self._config.patterns if pat]
        if not patterns:
            raise ValueError("local motif source patterns must be non-empty")
        files: dict[Path, Path] = {}
        for pattern in patterns:
            if self._config.recursive:
                matches = self._root.rglob(pattern)
            else:
                matches = self._root.glob(pattern)
            for path in matches:
                if not path.is_file():
                    continue
                files[path.resolve()] = path
        paths = sorted(files.values(), key=lambda p: p.as_posix())
        if not paths:
            raise FileNotFoundError(f"No motif files matched patterns {patterns} under {self._root}.")
        return paths

    def _index_paths(self, paths: Iterable[Path]) -> dict[str, Path]:
        index: dict[str, Path] = {}
        collisions: dict[str, list[Path]] = {}
        for path in paths:
            key = path.stem.lower()
            if key in index and index[key] != path:
                collisions.setdefault(key, [index[key]]).append(path)
            else:
                index[key] = path
        if collisions:
            rendered = "; ".join(
                f"{stem}: {', '.join(p.name for p in sorted(paths))}" for stem, paths in collisions.items()
            )
            raise ValueError(f"Duplicate motif ids detected under {self._root}: {rendered}")
        return index

    def _tf_name_for_path(self, path: Path) -> str:
        strategy = self._config.tf_name_strategy
        if strategy == "stem":
            return path.stem
        if strategy == "filename":
            return path.name
        raise ValueError(f"Unknown tf_name_strategy '{strategy}'")

    def _format_for_path(self, path: Path) -> str:
        ext = path.suffix.lower()
        fmt = self._format_map.get(ext)
        if fmt:
            return fmt
        if self._default_format:
            return self._default_format
        raise ValueError(
            f"No parser mapping for extension '{ext}' (file: {path}). "
            "Set ingest.local_sources[].format_map or default_format."
        )

    def _load_meme(self, path: Path) -> tuple[MemeFileParseResult, str]:
        cached = self._meme_cache.get(path)
        if cached is not None:
            return cached
        raw_payload = path.read_text()
        result = parse_meme_content(raw_payload, path)
        self._meme_cache[path] = (result, raw_payload)
        return result, raw_payload

    def _select_meme_motif(self, result: MemeFileParseResult, path: Path) -> MemeMotif:
        try:
            return select_meme_motif(
                result,
                file_stem=path.stem,
                selector=self._meme_motif_selector,
                path=path,
            )
        except ValueError as exc:
            raise ValueError(
                f"Unable to select MEME motif for '{path}': {exc}. "
                "Set ingest.local_sources[].meme_motif_selector to an index, motif ID, or label."
            ) from exc

    def _ensure_meme_probabilities(self, path: Path) -> None:
        if self._config.matrix_semantics != "probabilities":
            raise ValueError(
                f"MEME parser emits probability matrices; set matrix_semantics=probabilities for '{path}'."
            )

    def _meme_tags(self, result: MemeFileParseResult, motif: MemeMotif) -> dict[str, str]:
        tags = {"matrix_source": "file"}
        tags.update(self._config.tags)
        if result.meta.version:
            tags["meme_version"] = result.meta.version
        if result.meta.command_line:
            tags["meme_command"] = result.meta.command_line
        if result.meta.training_set:
            tags["meme_training_set"] = result.meta.training_set
        tags["meme_motif_id"] = motif.motif_id
        tags["meme_motif_name"] = motif.motif_name
        tags["meme_motif_label"] = motif.motif_label
        if motif.width is not None:
            tags["meme_width"] = str(motif.width)
        if motif.nsites is not None:
            tags["meme_nsites"] = str(motif.nsites)
        if motif.evalue is not None:
            tags["meme_evalue"] = str(motif.evalue)
        if motif.llr is not None:
            tags["meme_llr"] = str(motif.llr)
        return tags

    def _resolve_site_paths(self, query: SiteQuery) -> list[Path]:
        paths: list[Path] = []
        if query.motif_id:
            key = query.motif_id.strip().lower()
            if not key:
                raise ValueError("motif_id must be non-empty for local site retrieval")
            path = self._index.get(key)
            if path is None:
                options = ", ".join(sorted({p.stem for p in self._paths}))
                raise FileNotFoundError(
                    f"Motif id '{query.motif_id}' not found under {self._root}. Available: {options or 'none'}."
                )
            return [path]
        tf_query = query.tf_name.strip().lower() if query.tf_name else ""
        if not tf_query:
            raise ValueError("tf_name must be non-empty for local site retrieval")
        for path in self._paths:
            if self._tf_name_for_path(path).lower() == tf_query:
                paths.append(path)
        if not paths:
            raise FileNotFoundError(f"No motif files matched tf_name '{query.tf_name}' under {self._root}.")
        return paths

    def _iter_sites(self, paths: Iterable[Path], query: SiteQuery) -> Iterable[SiteInstance]:
        remaining = int(query.limit) if query.limit is not None else None
        for path in paths:
            fmt = self._format_for_path(path)
            if fmt != "MEME":
                raise ValueError(f"Binding-site extraction only supports MEME files; '{path}' uses format '{fmt}'.")
            result, _ = self._load_meme(path)
            motif = self._select_meme_motif(result, path)
            if not motif.block_sites:
                raise ValueError(
                    f"No MEME BLOCKS sites found in '{path}'. "
                    "Ensure the MEME output includes a BLOCKS section or disable extract_sites."
                )
            motif_ref = f"{self.source_id}:{path.stem}"
            provenance = Provenance(
                retrieved_at=datetime.now(timezone.utc),
                source_version=self._config.source_version,
                source_url=self._config.source_url,
                license=self._config.license,
                citation=self._config.citation,
                raw_artifact_paths=(self._relativize_path(path),),
                tags=self._site_tags(),
            )
            for site in motif.block_sites:
                seq = normalize_site_sequence(site.sequence, False)
                if motif.width is not None and len(seq) != motif.width:
                    raise ValueError(
                        f"MEME BLOCKS site length {len(seq)} does not match motif width {motif.width} in '{path}'."
                    )
                evidence = {
                    "sequence_name": site.sequence_name,
                    "start": site.start,
                    "meme_motif": motif.motif_label,
                }
                if site.pvalue is not None:
                    evidence["pvalue"] = site.pvalue
                yield SiteInstance(
                    source=self.source_id,
                    site_id=self._site_id(motif_ref, site.sequence_name, site.start, seq),
                    motif_ref=motif_ref,
                    organism=self._config.organism,
                    coordinate=None,
                    sequence=seq,
                    strand=None,
                    score=None,
                    evidence=evidence,
                    provenance=provenance,
                )
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

    def _site_id(self, motif_ref: str, sequence_name: str, start: int, seq: str) -> str:
        raw = f"{motif_ref}|{sequence_name}|{start}|{seq}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _site_tags(self) -> dict[str, object]:
        tags: dict[str, object] = {"record_kind": "meme_blocks", "matrix_source": "file"}
        tags.update(self._config.tags)
        return tags

    def _load_pwm(self, path: Path):
        fmt = self._format_for_path(path)
        try:
            return load_pwm(path, fmt=fmt, extra_modules=self._config.extra_parser_modules)
        except ValueError as exc:
            if "No parser registered for format" in str(exc):
                raise ValueError(
                    f"Parser '{fmt}' is not registered for file '{path}'. "
                    "Configure io.parsers.extra_modules or update format_map."
                ) from exc
            raise ValueError(f"Failed to parse motif file '{path}': {exc}") from exc

    def _relativize_path(self, path: Path) -> str:
        try:
            rel = path.relative_to(self._root)
            return rel.as_posix()
        except ValueError:
            return str(path)
