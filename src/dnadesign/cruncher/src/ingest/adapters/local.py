"""Local filesystem motif adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from dnadesign.cruncher.ingest.models import (
    DatasetDescriptor,
    DatasetQuery,
    MotifDescriptor,
    MotifQuery,
    MotifRecord,
    OrganismRef,
    SiteInstance,
    SiteQuery,
)
from dnadesign.cruncher.ingest.normalize import build_motif_record
from dnadesign.cruncher.io.parsers.backend import load_pwm


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

    def capabilities(self) -> Set[str]:
        return {"motifs:list", "motifs:get", "motifs:iter"}

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
        pwm = self._load_pwm(path)
        tf_name = self._tf_name_for_path(path)
        raw_payload = path.read_text()
        raw_paths = [self._relativize_path(path)]
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
        del query
        raise ValueError("Local motif sources do not provide binding sites.")

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery) -> Iterable[SiteInstance]:
        del motif_id, query
        raise ValueError("Local motif sources do not provide binding sites.")

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
            pwm = self._load_pwm(path)
            tf_name = self._tf_name_for_path(path)
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
