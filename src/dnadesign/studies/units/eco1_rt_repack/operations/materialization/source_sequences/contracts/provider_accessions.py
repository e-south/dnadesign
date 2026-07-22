"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts/provider_accessions.py

Provider accession policy from the Eco1 conservation source contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from re import Pattern

from .conservation_sources import ConservationSourceContract


@dataclass(frozen=True)
class ProviderAccessionPolicy:
    """Compiled provider accession patterns declared by ``conservation-sources.yaml``."""

    provider_ids: tuple[str, ...]
    patterns_by_provider: Mapping[str, tuple[Pattern[str], ...]]

    @classmethod
    def from_contract(cls, contract: ConservationSourceContract) -> ProviderAccessionPolicy:
        """Compile provider accession patterns from a parsed source contract."""

        return cls(
            provider_ids=contract.provider_ids,
            patterns_by_provider={
                provider_id: tuple(_compile_patterns(provider_id, patterns))
                for provider_id, patterns in contract.provider_accession_patterns.items()
            },
        )

    def provider_for_accession(self, accession: str) -> str:
        """Resolve one accession to a declared provider id."""

        for provider_id in self.provider_ids:
            if self.valid_provider_accession(provider_id, accession):
                return provider_id
        raise ValueError(f"unsupported accession provider for {accession!r}")

    def valid_provider_accession(self, provider_id: str, accession: str) -> bool:
        """Return whether an accession has the declared provider's expected shape."""

        patterns = self.patterns_by_provider.get(provider_id, ())
        return any(pattern.fullmatch(accession) is not None for pattern in patterns)


def _compile_patterns(provider_id: str, patterns: tuple[str, ...]) -> tuple[Pattern[str], ...]:
    compiled: list[Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"invalid accession pattern for provider {provider_id!r}: {pattern!r}") from exc
    return tuple(compiled)
