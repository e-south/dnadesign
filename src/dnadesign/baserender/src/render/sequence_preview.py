"""Bounded, explicit sequence previews for compact render surfaces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_DEFAULT_VISIBLE_BASES = 12
_DIGEST_PREFIX_LENGTH = 12
_HASH_CHUNK_CHARS = 4_096


def _sha256_prefix(text: str, *, encoding: str, errors: str = "strict") -> str:
    digest = hashlib.sha256()
    for start in range(0, len(text), _HASH_CHUNK_CHARS):
        digest.update(text[start : start + _HASH_CHUNK_CHARS].encode(encoding, errors=errors))
    return digest.hexdigest()[:_DIGEST_PREFIX_LENGTH]


def _escaped_edge(text: str, *, visible_chars: int, from_end: bool) -> str:
    characters = reversed(text[-visible_chars:]) if from_end else iter(text[:visible_chars])
    tokens: list[str] = []
    used = 0
    for character in characters:
        token = character.encode("unicode_escape").decode("ascii")
        if tokens and used + len(token) > visible_chars:
            break
        tokens.append(token)
        used += len(token)
        if used >= visible_chars:
            break
    if from_end:
        tokens.reverse()
    return "".join(tokens)


@dataclass(frozen=True, slots=True)
class SequencePreview:
    """Presentation-only summary; the owning typed contract retains exact sequence data."""

    preview: str
    length_nt: int
    sha256_prefix: str

    def label(self, component: str) -> str:
        return f"{component} · {self.length_nt} nt · {self.sha256_prefix} · {self.preview}"


@dataclass(frozen=True, slots=True)
class TextPreview:
    preview: str
    length_chars: int
    sha256_prefix: str
    abbreviated: bool


def bounded_text_preview(text: str, *, visible_chars: int = 12, exact_limit: int = 16) -> TextPreview:
    """Keep ordinary labels exact and summarize oversized labels deterministically."""

    digest = _sha256_prefix(text, encoding="utf-8", errors="surrogatepass")
    if len(text) <= exact_limit:
        display_text = text.encode("unicode_escape").decode("ascii")
        if len(display_text) <= exact_limit:
            return TextPreview(preview=display_text, length_chars=len(text), sha256_prefix=digest, abbreviated=False)
    left_length = (visible_chars + 1) // 2
    right_length = visible_chars // 2
    left = _escaped_edge(text, visible_chars=left_length, from_end=False)
    right = _escaped_edge(text, visible_chars=right_length, from_end=True)
    preview = f"{left}…{right}"
    return TextPreview(preview=preview, length_chars=len(text), sha256_prefix=digest, abbreviated=True)


def bounded_sequence_preview(
    sequence: str,
    *,
    visible_bases: int = _DEFAULT_VISIBLE_BASES,
) -> SequencePreview:
    """Return a deterministic preview whose sequence text has a fixed display budget."""

    if visible_bases < 2:
        raise ValueError("visible_bases must be at least 2")
    digest = _sha256_prefix(sequence, encoding="ascii")
    if not sequence:
        preview = "none"
    else:
        shown_bases = min(visible_bases, len(sequence) - 1)
        left_length = (shown_bases + 1) // 2
        right_length = shown_bases // 2
        left = sequence[:left_length]
        right = sequence[-right_length:] if right_length else ""
        preview = f"{left}…{right}"
    return SequencePreview(preview=preview, length_nt=len(sequence), sha256_prefix=digest)


__all__ = ["SequencePreview", "TextPreview", "bounded_sequence_preview", "bounded_text_preview"]
