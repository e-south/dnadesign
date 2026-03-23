#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${CONSTRUCT_RUNBOOK_MODE:-dry-run-all}"
MANIFEST="$WORKSPACE_DIR/inputs/seed_manifest.yaml"
USR_ROOT="${CONSTRUCT_RUNBOOK_USR_ROOT:-$WORKSPACE_DIR/outputs/usr_datasets}"
PROJECT_ROOT="${CONSTRUCT_RUNBOOK_PROJECT_ROOT:-__CONSTRUCT_PROJECT_ROOT__}"
PROJECTS=(slot_a_window slot_b_window)

construct_cmd() {
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    uv run --project "$PROJECT_ROOT" construct "$@"
  else
    uv run construct "$@"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./runbook.sh --mode <seed|validate-all|dry-run-all|run-all>

Modes:
  seed          Bootstrap the curated demo USR datasets only.
  validate-all  Seed demo datasets, then runtime-validate both packaged projects.
  dry-run-all   Seed demo datasets, runtime-validate both packaged projects, then dry-run both.
  run-all       Seed demo datasets, runtime-validate both packaged projects, then materialize both.

Environment overrides:
  CONSTRUCT_RUNBOOK_MODE
  CONSTRUCT_RUNBOOK_USR_ROOT
  CONSTRUCT_RUNBOOK_PROJECT_ROOT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

seed_demo() {
  construct_cmd seed anchor-template-demo \
    --root "$USR_ROOT" \
    --manifest "$MANIFEST"
}

validate_all() {
  local project
  for project in "${PROJECTS[@]}"; do
    construct_cmd workspace validate-project \
      --workspace "$WORKSPACE_DIR" \
      --project "$project" \
      --runtime
  done
}

dry_run_all() {
  local project
  for project in "${PROJECTS[@]}"; do
    construct_cmd workspace run-project \
      --workspace "$WORKSPACE_DIR" \
      --project "$project" \
      --dry-run
  done
}

run_all() {
  local project
  for project in "${PROJECTS[@]}"; do
    construct_cmd workspace run-project \
      --workspace "$WORKSPACE_DIR" \
      --project "$project"
  done
}

seed_demo

case "$MODE" in
  seed)
    ;;
  validate-all)
    validate_all
    ;;
  dry-run-all)
    validate_all
    dry_run_all
    ;;
  run-all)
    validate_all
    run_all
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
