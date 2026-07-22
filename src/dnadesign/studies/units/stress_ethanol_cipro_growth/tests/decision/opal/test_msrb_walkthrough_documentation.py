"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_msrb_walkthrough_documentation.py

Documentation contracts for the stress-study MSRB walkthrough.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("dnadesign repository root not found")


REPO_ROOT = _repo_root()
WALKTHROUGH = (
    REPO_ROOT / "docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior-walkthrough.html"
)
ROUTE_INDEX = REPO_ROOT / "docs/studies/stress_ethanol_cipro_growth/routes/README.md"
OPAL_CONTEXT_INDEX = REPO_ROOT / "docs/studies/stress_ethanol_cipro_growth/contexts/opal/README.md"


class _WalkthroughParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang: str | None = None
        self.metadata: dict[str, str] = {}
        self.references: list[str] = []
        self.text_parts: list[str] = []
        self._ignored_text_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.html_lang = attributes.get("lang")
        if tag in {"script", "style"}:
            self._ignored_text_depth += 1
        if tag == "meta" and attributes.get("name") and attributes.get("content"):
            self.metadata[str(attributes["name"])] = str(attributes["content"])
        for key in ("href", "src"):
            value = attributes.get(key)
            if value:
                self.references.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._ignored_text_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_text_depth == 0:
            self.text_parts.append(data)


def _parsed_walkthrough() -> tuple[str, _WalkthroughParser]:
    source = WALKTHROUGH.read_text(encoding="utf-8")
    parser = _WalkthroughParser()
    parser.feed(source)
    return source, parser


def test_msrb_walkthrough_is_linked_from_both_study_navigation_hubs() -> None:
    route_text = ROUTE_INDEX.read_text(encoding="utf-8")
    context_text = OPAL_CONTEXT_INDEX.read_text(encoding="utf-8")

    assert "[MSRB symbol walkthrough](../contexts/opal/multistate-response-behavior-walkthrough.html)" in route_text
    assert "[MSRB symbol walkthrough](multistate-response-behavior-walkthrough.html)" in context_text


def test_msrb_walkthrough_preserves_the_stable_objective_semantics() -> None:
    source, parser = _parsed_walkthrough()
    visible_text = " ".join(" ".join(parser.text_parts).split())

    assert parser.html_lang == "en"
    assert parser.metadata["artifact-role"] == "didactic-companion"
    assert parser.metadata["msrb-objective-id"] == "multistate_response_behavior_v1"
    for required_text in (
        "binary target program",
        "Response ordering",
        "Intended-ON signal",
        "Intended-OFF suppression",
        "soft minimum",
        "τ ≈ 0.31",
        "not a pass threshold",
        "one wet-lab change may move several coordinates at once",
    ):
        assert required_text in visible_text

    assert "Transient explanatory artifact" not in visible_text
    assert "/Users/" not in source
    assert "file://" not in source
    assert "https://" not in source
    assert "http://" not in source
    assert "document.addEventListener('keydown'" not in source
    assert "tablist.addEventListener('keydown'" in source
    assert "event.preventDefault();" in source
    assert "tabs[nextIndex].focus();" in source


def test_msrb_walkthrough_local_references_resolve_inside_the_repository() -> None:
    _, parser = _parsed_walkthrough()
    assert {
        "../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md",
        "multistate-response-behavior.md",
        "multistate-response-behavior.md#why-the-study-uses-approximately-031-log2",
    } <= set(parser.references)

    for reference in parser.references:
        parsed = urlsplit(reference)
        assert parsed.scheme == ""
        assert parsed.netloc == ""
        target = (WALKTHROUGH.parent / unquote(parsed.path)).resolve()
        assert target.is_relative_to(REPO_ROOT)
        assert target.is_file()

    study_binding = (WALKTHROUGH.parent / "multistate-response-behavior.md").read_text(encoding="utf-8")
    assert "#### Why the study uses approximately 0.31 log2" in study_binding
