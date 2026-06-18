#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
STATUS_SRC="$REPO_ROOT/src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status"
PROBE_SRC="$REPO_ROOT/src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe"
failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

require_file() {
  local path="$1"
  [[ -f "$path" ]] && pass "found ${path#$REPO_ROOT/}" || fail "missing file $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] && pass "found ${path#$REPO_ROOT/}" || fail "missing directory $path"
}

require_section() {
  local section="$1"
  grep -Fxq "$section" "$SKILL_FILE" && pass "section present: $section" || fail "section missing: $section"
}

require_pattern() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  grep -Eq "$pattern" "$path" && pass "$label" || fail "$label"
}

require_absent() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  if grep -Eq "$pattern" "$path"; then fail "$label"; else pass "$label"; fi
}

require_tree_absent() {
  local pattern="$1"
  local label="$2"
  local path="$3"
  if rg -n "$pattern" "$path" -g '*.py' >/dev/null; then fail "$label"; else pass "$label"; fi
}

effective_line_count() {
  local path="$1"
  local line_count
  line_count="$(wc -l < "$path" | tr -d ' ')"
  if [[ "$path" == *.py ]] \
    && [[ "$(sed -n '1p' "$path")" == '"""' ]] \
    && [[ "$(sed -n '2p' "$path")" == "--------------------------------------------------------------------------------" ]] \
    && [[ "$(sed -n '9p' "$path")" == "--------------------------------------------------------------------------------" ]] \
    && [[ "$(sed -n '10p' "$path")" == '"""' ]]; then
    line_count=$((line_count - 10))
  fi
  printf '%s\n' "$line_count"
}

require_max_lines() {
  local path="$1"
  local max_lines="$2"
  local label="$3"
  local line_count
  line_count="$(effective_line_count "$path")"
  if [[ "$line_count" -le "$max_lines" ]]; then
    pass "$label (${line_count}/${max_lines})"
  else
    fail "$label (${line_count}/${max_lines})"
  fi
}

require_frontmatter_yaml() {
  if uv run python - "$SKILL_FILE" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if not text.startswith("---\n"):
    raise SystemExit("missing opening frontmatter delimiter")
try:
    frontmatter = text.split("---", 2)[1]
except IndexError as exc:
    raise SystemExit("missing closing frontmatter delimiter") from exc
payload = yaml.safe_load(frontmatter)
if not isinstance(payload, dict):
    raise SystemExit("frontmatter is not a mapping")
if payload.get("name") != path.parent.name:
    raise SystemExit("frontmatter name does not match skill folder")
description = payload.get("description")
if not isinstance(description, str) or not description.strip():
    raise SystemExit("frontmatter description is missing")
if len(description) > 260:
    raise SystemExit(f"frontmatter description exceeds hard budget: {len(description)} > 260")
if "Do not use for another study or for family-level routing" not in description:
    raise SystemExit("frontmatter must reject other studies and family-level routing")
metadata = payload.get("metadata")
if not isinstance(metadata, dict):
    raise SystemExit("frontmatter metadata is missing")
version = metadata.get("version")
if not isinstance(version, str) or version.count(".") != 2:
    raise SystemExit("frontmatter metadata.version must be semver-shaped")
PY
  then
    pass "frontmatter parses as YAML and stays within discovery budget"
  else
    fail "frontmatter parses as YAML and stays within discovery budget"
  fi
}

require_bounded_preflight_commands() {
  if uv run python - "$ROOT_DIR" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth" <<'PY'
from pathlib import Path
import sys

roots = [Path(value) for value in sys.argv[1:]]
bad: list[str] = []
for root in roots:
    for path in sorted(root.rglob("*.md")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if (
                "studies.stress-ethanol-cipro-growth.preflight" in line
                and "--scope next" in line
                and "--command-timeout-seconds 30" not in line
            ):
                bad.append(f"{path}:{line_number}: {line.strip()}")
if bad:
    raise SystemExit("\n".join(bad))
PY
  then
    pass "documented stress preflight commands keep the 30s timeout"
  else
    fail "documented stress preflight commands keep the 30s timeout"
  fi
}

require_file "$SKILL_FILE"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/ops.study.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/notify/profile-doctor.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/notify/resolve-events.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/opal.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/sequence-views.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/completion.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/notify.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/latentdna_reference_normalization_audit.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/opal_candidate_table_pre_assay.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/README.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/densegen.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/compute/infer.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/construct.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/cluster.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/campaign-commands.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/latentdna.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/candidate-table.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/README.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/densegen.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/infer.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/latentdna.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/cluster.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/opal.yaml"
require_file "$STATUS_SRC/ops/status.registry.yaml"
require_file "$STATUS_SRC/service.py"
require_file "$STATUS_SRC/snapshot.py"
require_file "$STATUS_SRC/preflight.py"
require_file "$STATUS_SRC/probes/runtime_dependencies.py"
require_file "$STATUS_SRC/probes/semantic_completeness.py"
require_file "$STATUS_SRC/probes/sequence_view_contracts.py"
require_file "$PROBE_SRC/README.md"
require_file "$PROBE_SRC/__init__.py"
require_file "$PROBE_SRC/__main__.py"
require_file "$PROBE_SRC/plan_logic/axis_oracle.py"
require_file "$PROBE_SRC/evaluation/decision.py"
require_file "$PROBE_SRC/evaluation/prediction_ledger.py"
require_file "$PROBE_SRC/reporting/progress.py"
require_dir "$PROBE_SRC/reporting/review"
require_dir "$PROBE_SRC/reporting/review/aggregate_plots"
require_file "$PROBE_SRC/reporting/review/__init__.py"
require_file "$PROBE_SRC/reporting/review/builder.py"
require_file "$PROBE_SRC/reporting/review/aggregate_plots/contracts.py"
require_file "$PROBE_SRC/reporting/review/aggregate_plots/renderers.py"
require_file "$PROBE_SRC/reporting/review/aggregate_plots/writer.py"
require_file "$PROBE_SRC/reporting/status.py"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/external-sources.md"

require_frontmatter_yaml
require_bounded_preflight_commands
require_section "## Required Deliverables"
require_section "## Trigger Tests"
require_pattern '^name: stress-ethanol-cipro-growth-status$' "skill name is study-specific"
require_pattern 'studies\.stress-ethanol-cipro-growth\.status' "skill names study status command"
require_pattern 'studies\.stress-ethanol-cipro-growth\.preflight' "skill names study preflight command"
require_pattern 'stress_ethanol_cipro_growth' "skill anchors the concrete study id"
require_pattern 'Do not use for another study or for family-level routing' "skill rejects family-level routing"
require_pattern 'Do not generalize it to another study' "skill guardrail rejects cross-study reuse"
require_pattern 'OPAL candidate-table details are meaningful only in this study' "skill keeps OPAL table study-owned"
require_pattern 'operations/status/probes/' "skill reference exposes probe subpackage" "$REFERENCE_DIR/study-surfaces.md"
require_pattern '^## References$' "skill exposes progressive-disclosure references"
require_pattern 'routes/decision/opal/README\.md' "skill routes OPAL detail after one-hop map"
require_pattern 'routes/analysis/latentdna\.md' "skill routes LatentDNA detail after one-hop map"
require_pattern 'command-groups/README\.md' "skill routes runtime command groups through progressive map"
require_pattern 'opal_candidate_table_pre_assay' "preflight contract documents current OPAL main-path gate" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md"
require_pattern 'opal\.candidate_table\.contract' "OPAL preflight checks validate candidate-table contract" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/opal_candidate_table_pre_assay.yaml"
require_pattern 'appendix_source_datasets' "LatentDNA binding separates appendix sources" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/latentdna/binding.yaml"
require_pattern 'latentdna\.readiness\.semantic' "preflight contract documents LatentDNA semantic readiness" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md"
require_pattern 'missing_source_datasets' "preflight contract exposes LatentDNA missing-source details" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md"
require_pattern 'latentdna\.readiness\.semantic' "preflight provider emits LatentDNA semantic readiness check" "$STATUS_SRC/preflight.py"
require_pattern 'missing_source_datasets' "preflight provider emits LatentDNA missing-source details" "$STATUS_SRC/preflight.py"
require_pattern 'LatentDNA primary readiness attention' "LatentDNA readiness summary reports primary attention" "$STATUS_SRC/latentdna_readiness.py"
require_pattern 'missing_appendix_source_datasets' "LatentDNA readiness exposes appendix drift separately" "$STATUS_SRC/latentdna_readiness.py"
require_pattern 'densegen-axis-probe-v0\.md' "OPAL route links the DenseGen probe context" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md"
require_pattern 'densegen-axis-probe-v0\.md' "OPAL context index links the DenseGen probe context" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/README.md"
require_pattern '__all__: list\[str\] = \[\]' "DenseGen probe package root exports no flat aggregate API" "$PROBE_SRC/__init__.py"
require_pattern 'python -m dnadesign\.studies\.units\.stress_ethanol_cipro_growth\.decision\.opal\.densegen_axis_probe report --run-root' "DenseGen probe context documents runnable report command" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md"
require_pattern 'python -m dnadesign\.studies\.units\.stress_ethanol_cipro_growth\.decision\.opal\.densegen_axis_probe progress --run-root' "DenseGen probe context documents runnable progress command" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md"
require_absent 'uv run opal_densegen_axis_probe' "DenseGen probe context avoids non-existent project script" "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md"
require_tree_absent 'dnadesign\.opal\.src' "DenseGen probe uses public OPAL API only" "$PROBE_SRC"

require_absent 'promoter-study-status' "skill has no old status kind"
require_absent 'promoter-study-preflight' "skill has no old preflight kind"
require_absent 'usr\.data-plane\.promoter-study' "skill has no old registry id"
require_absent 'status_adapters/promoter_status' "skill has no old adapter path"
require_absent 'generic promoter' "skill avoids generic promoter routing language"

if [[ -e "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/notify.yaml" ]]; then
  fail "stress Notify commands are split by subcommand family"
else
  pass "stress Notify commands are split by subcommand family"
fi

if [[ -e "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation.yaml" ]]; then
  fail "stress Infer readiness checks are split by owner/action lane"
else
  pass "stress Infer readiness checks are split by owner/action lane"
fi

if [[ -e "$PROBE_SRC/review.py" ]]; then
  fail "DenseGen probe review is a semantic package"
else
  pass "DenseGen probe review is a semantic package"
fi

if [[ -e "$PROBE_SRC/reporting/review/probe_plots.py" ]]; then
  fail "DenseGen probe aggregate plots are a semantic package"
else
  pass "DenseGen probe aggregate plots are a semantic package"
fi

require_max_lines "$STATUS_SRC/service.py" 320 "status service stays orchestration-sized"
require_max_lines "$STATUS_SRC/probes/runtime_dependencies.py" 140 "runtime probe module stays bounded"
require_max_lines "$STATUS_SRC/probes/semantic_completeness.py" 200 "semantic-completeness probe module stays bounded"
require_max_lines "$STATUS_SRC/probes/sequence_view_contracts.py" 240 "sequence-view probe module stays bounded"
require_max_lines "$PROBE_SRC/plan_logic/axis_oracle.py" 450 "DenseGen axis-oracle module stays bounded"
require_max_lines "$PROBE_SRC/evaluation/decision.py" 450 "DenseGen probe decision module stays bounded"
require_max_lines "$PROBE_SRC/evaluation/prediction_ledger.py" 120 "DenseGen probe prediction-ledger module stays bounded"
require_max_lines "$PROBE_SRC/cli.py" 280 "DenseGen probe CLI module stays bounded"
require_max_lines "$PROBE_SRC/reporting/status.py" 220 "DenseGen probe status module stays bounded"
require_max_lines "$PROBE_SRC/reporting/review/builder.py" 180 "DenseGen probe review builder stays bounded"
require_max_lines "$PROBE_SRC/reporting/review/configured_plots.py" 260 "DenseGen probe review configured-plot module stays bounded"
require_max_lines "$PROBE_SRC/reporting/review/aggregate_plots/contracts.py" 140 "DenseGen probe aggregate plot contracts stay bounded"
require_max_lines "$PROBE_SRC/reporting/review/aggregate_plots/renderers.py" 260 "DenseGen probe aggregate plot renderers stay bounded"
require_max_lines "$PROBE_SRC/reporting/review/aggregate_plots/writer.py" 120 "DenseGen probe aggregate plot writer stays bounded"
require_max_lines "$PROBE_SRC/reporting/review/rendering/html.py" 320 "DenseGen probe review HTML renderer stays bounded"
require_max_lines "$PROBE_SRC/reporting/review/rendering/markdown.py" 240 "DenseGen probe review Markdown renderer stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/README.md" 140 "stress study route map stays one-hop"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/densegen.md" 80 "DenseGen route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/compute/infer.md" 80 "Infer route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/construct.md" 80 "Construct route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/cluster.md" 80 "Cluster route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md" 100 "OPAL route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/campaign-commands.md" 80 "OPAL command detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/latentdna.md" 120 "LatentDNA route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-axis-probe-v0.md" 180 "DenseGen probe context stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/sequence-views.yaml" 120 "Infer sequence-view readiness fragment stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/notify.yaml" 100 "Infer Notify readiness fragment stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/README.md" 70 "runtime command-group map stays one-hop"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/densegen.yaml" 40 "DenseGen runtime lane stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/infer.yaml" 45 "Infer runtime lane stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/latentdna.yaml" 40 "LatentDNA runtime lane stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/cluster.yaml" 40 "Cluster runtime lane stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/runtime/command-groups/lanes/opal.yaml" 40 "OPAL runtime lane stays bounded"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
