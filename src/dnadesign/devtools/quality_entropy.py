"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/quality_entropy.py

Builds a quality entropy report for stale SOR metadata, unresolved gaps, and evidence links.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from .docs_checks import LAST_VERIFIED_PATTERN, OWNER_PATTERN, SOR_MARKDOWN_FILES

_SECTION_HEADER_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_TABLE_ROW_PATTERN = re.compile(r"^\|(.+)\|\s*$")
_CODE_SPAN_PATTERN = re.compile(r"`([^`]+)`")
_URL_PATTERN = re.compile(r"https?://[^\s|)]+")


@dataclass(frozen=True)
class EntropyReport:
    stale_sor_docs: list[str]
    unresolved_gaps: list[str]
    scorecard_contract_issues: list[str]
    broken_evidence_links: list[str]

    @property
    def has_critical_findings(self) -> bool:
        return bool(self.stale_sor_docs or self.scorecard_contract_issues or self.broken_evidence_links)


def _extract_metadata_field(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_section_lines(text: str, section_title: str) -> list[str]:
    lines = text.splitlines()
    result: list[str] = []
    in_section = False
    target_header = f"## {section_title}"
    for line in lines:
        if line.strip() == target_header:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            result.append(line.rstrip("\n"))
    return result


def _required_section_lines(text: str, section_title: str) -> list[str]:
    target_header = f"## {section_title}"
    if target_header not in {line.strip() for line in text.splitlines()}:
        raise ValueError(f"QUALITY_SCORE.md missing required section: {target_header}")
    return _extract_section_lines(text, section_title)


def _extract_table_rows(section_lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if set(stripped.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(
            cell.lower()
            in {
                "area",
                "axis",
                "score (0-4)",
                "trend",
                "gate",
                "evidence",
                "owner",
                "last verified",
                "next action",
                "gap",
                "impact",
                "tracking artifact",
                "exit criteria",
            }
            for cell in cells
        ):
            continue
        rows.append(cells)
    return rows


def _quality_scorecard_rows(quality_score_text: str) -> list[list[str]]:
    score_lines = _required_section_lines(quality_score_text, "Quality scorecard")
    rows = _extract_table_rows(score_lines)
    if not rows:
        raise ValueError("QUALITY_SCORE.md section '## Quality scorecard' must include at least one data row.")
    return rows


def _find_stale_sor_docs(repo_root: Path, *, max_sor_age_days: int) -> list[str]:
    issues: list[str] = []
    today = dt.date.today()
    for doc_name in SOR_MARKDOWN_FILES:
        path = repo_root / doc_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")

        owner = _extract_metadata_field(text, OWNER_PATTERN)
        last_verified_raw = _extract_metadata_field(text, LAST_VERIFIED_PATTERN)
        if owner is None or not owner or last_verified_raw is None or not last_verified_raw:
            issues.append(f"{path}: missing required Owner/Last verified metadata.")
            continue

        try:
            last_verified = dt.date.fromisoformat(last_verified_raw)
        except ValueError:
            issues.append(f"{path}: Last verified must use YYYY-MM-DD.")
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified is in the future ({last_verified.isoformat()}).")
            continue

        age_days = (today - last_verified).days
        if age_days > max_sor_age_days:
            issues.append(f"{path}: stale ({age_days} days old; max {max_sor_age_days}).")
    return issues


def _find_unresolved_gaps(quality_score_text: str) -> list[str]:
    gap_lines = _required_section_lines(quality_score_text, "Gap tracker")
    rows = _extract_table_rows(gap_lines)
    gaps: list[str] = []
    for row in rows:
        if not row:
            continue
        gap_name = row[0].strip()
        if gap_name:
            gaps.append(gap_name)
    return gaps


def _scorecard_area(row: list[str]) -> str:
    if not row:
        return "<unknown>"
    area = row[0].strip()
    return area or "<unknown>"


def _extract_evidence_tokens(cell: str) -> tuple[list[str], list[str]]:
    code_tokens = [item.strip() for item in _CODE_SPAN_PATTERN.findall(cell) if item.strip()]
    url_tokens = [item.strip() for item in _URL_PATTERN.findall(cell) if item.strip()]
    return code_tokens, url_tokens


def _find_scorecard_contract_issues(*, rows: list[list[str]], max_sor_age_days: int) -> list[str]:
    issues: list[str] = []
    today = dt.date.today()
    for index, row in enumerate(rows, start=1):
        area = _scorecard_area(row)
        if len(row) != 9:
            issues.append(f"row {index} ({area}): expected 9 columns, found {len(row)}.")
            continue

        score = row[2].strip()
        gate = row[4].strip()
        evidence = row[5].strip()
        owner = row[6].strip()
        last_verified = row[7].strip()
        next_action = row[8].strip()

        if not score:
            issues.append(f"row {index} ({area}): score must not be empty.")
        else:
            try:
                score_value = float(score)
            except ValueError:
                issues.append(f"row {index} ({area}): score must be numeric in range 0-4.")
            else:
                if score_value < 0.0 or score_value > 4.0:
                    issues.append(f"row {index} ({area}): score must be in range 0-4.")

        if not gate:
            issues.append(f"row {index} ({area}): gate must not be empty.")

        if not owner:
            issues.append(f"row {index} ({area}): owner must not be empty.")

        if not last_verified:
            issues.append(f"row {index} ({area}): last verified must not be empty.")
        else:
            try:
                row_date = dt.date.fromisoformat(last_verified)
            except ValueError:
                issues.append(f"row {index} ({area}): last verified must use YYYY-MM-DD.")
            else:
                if row_date > today:
                    issues.append(f"row {index} ({area}): last verified cannot be in the future.")
                age_days = (today - row_date).days
                if age_days > max_sor_age_days:
                    issues.append(
                        f"row {index} ({area}): last verified is stale by {age_days} days (max {max_sor_age_days})."
                    )

        if not evidence:
            issues.append(f"row {index} ({area}): evidence must not be empty.")
        else:
            code_tokens, url_tokens = _extract_evidence_tokens(evidence)
            if not code_tokens and not url_tokens:
                issues.append(f"row {index} ({area}): evidence must include at least one backticked path/ref or URL.")

        if not next_action:
            issues.append(f"row {index} ({area}): next action must not be empty.")

    return issues


def _find_broken_evidence_links(repo_root: Path, rows: list[list[str]]) -> list[str]:
    broken: list[str] = []
    for row in rows:
        if len(row) < 6:
            continue
        area = _scorecard_area(row)
        evidence_cell = row[5]
        code_tokens, _url_tokens = _extract_evidence_tokens(evidence_cell)
        for token in code_tokens:
            if token.startswith(("http://", "https://", "mailto:")):
                continue
            target = (repo_root / token).resolve()
            if not target.exists():
                broken.append(f"{area}: {token}")
    return broken


def build_entropy_report(*, repo_root: Path, max_sor_age_days: int) -> EntropyReport:
    quality_score_path = repo_root / "QUALITY_SCORE.md"
    if not quality_score_path.exists():
        raise FileNotFoundError(f"Expected quality score document at {quality_score_path}")

    quality_score_text = quality_score_path.read_text(encoding="utf-8")
    scorecard_rows = _quality_scorecard_rows(quality_score_text)
    return EntropyReport(
        stale_sor_docs=_find_stale_sor_docs(repo_root, max_sor_age_days=max_sor_age_days),
        unresolved_gaps=_find_unresolved_gaps(quality_score_text),
        scorecard_contract_issues=_find_scorecard_contract_issues(
            rows=scorecard_rows, max_sor_age_days=max_sor_age_days
        ),
        broken_evidence_links=_find_broken_evidence_links(repo_root, scorecard_rows),
    )


def _render_report_markdown(report: EntropyReport) -> str:
    lines = [
        "# Quality entropy report",
        "",
        f"- Generated at (UTC): `{dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}`",
        f"- Stale SOR docs: {len(report.stale_sor_docs)}",
        f"- Unresolved gaps: {len(report.unresolved_gaps)}",
        f"- Scorecard contract issues: {len(report.scorecard_contract_issues)}",
        f"- Broken evidence links: {len(report.broken_evidence_links)}",
        "",
    ]

    lines.append("## Stale SOR docs")
    if report.stale_sor_docs:
        lines.extend(f"- {item}" for item in report.stale_sor_docs)
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Unresolved gaps")
    if report.unresolved_gaps:
        lines.extend(f"- {item}" for item in report.unresolved_gaps)
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Scorecard contract issues")
    if report.scorecard_contract_issues:
        lines.extend(f"- {item}" for item in report.scorecard_contract_issues)
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Broken evidence links")
    if report.broken_evidence_links:
        lines.extend(f"- {item}" for item in report.broken_evidence_links)
    else:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate quality entropy report.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--max-sor-age-days", type=int, default=90)
    parser.add_argument("--report-file", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build_entropy_report(repo_root=args.repo_root, max_sor_age_days=args.max_sor_age_days)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(_render_report_markdown(report), encoding="utf-8")
    print(f"Quality entropy report written to {args.report_file}")
    return 1 if report.has_critical_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
