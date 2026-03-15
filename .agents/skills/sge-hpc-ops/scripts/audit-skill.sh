#!/usr/bin/env bash
set -euo pipefail

skill_dir="${1:-.}"
skill_file="$skill_dir/SKILL.md"
folder_name="$(basename "$skill_dir")"
failures=0

pass() {
  printf 'PASS: %s\n' "$1"
}

fail() {
  printf 'FAIL: %s\n' "$1"
  failures=$((failures + 1))
}

require_heading() {
  local heading="$1"
  if grep -Fxq "$heading" "$skill_file"; then
    pass "Heading present: $heading"
  else
    fail "Heading missing: $heading"
  fi
}

if [[ -f "$skill_file" ]]; then
  pass "SKILL.md exists"
else
  fail "SKILL.md is missing"
fi

if [[ "$folder_name" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  pass "Folder name is kebab-case"
else
  fail "Folder name must be kebab-case"
fi

if [[ -f "$skill_dir/README.md" ]]; then
  fail "README.md should not be inside skill folder"
else
  pass "No README.md in skill folder"
fi

if [[ ! -f "$skill_file" ]]; then
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi

frontmatter="$(awk '
  NR==1 && $0=="---" {in_fm=1; next}
  in_fm && $0=="---" {exit}
  in_fm {print}
' "$skill_file")"

if [[ -n "$frontmatter" ]]; then
  pass "Frontmatter block found"
else
  fail "Frontmatter block missing or malformed"
fi

name_field="$(printf '%s\n' "$frontmatter" | awk -F': *' '$1=="name" {print $2; exit}')"
description_field="$(printf '%s\n' "$frontmatter" | awk -F': *' '$1=="description" {sub(/^description:[ ]*/, "", $0); print $0; exit}')"
metadata_block="$(printf '%s\n' "$frontmatter" | awk '
  /^metadata:[[:space:]]*$/ {in_meta=1; next}
  in_meta && $0 ~ /^[^[:space:]]/ {exit}
  in_meta {print}
')"
version_field="$(printf '%s\n' "$metadata_block" | sed -n 's/^[[:space:]]*version:[[:space:]]*//p' | head -n 1)"
category_field="$(printf '%s\n' "$metadata_block" | sed -n 's/^[[:space:]]*category:[[:space:]]*//p' | head -n 1)"
tags_field="$(printf '%s\n' "$metadata_block" | sed -n 's/^[[:space:]]*tags:[[:space:]]*//p' | head -n 1)"

if [[ -n "$name_field" ]]; then
  pass "Frontmatter name field present"
else
  fail "Frontmatter name field missing"
fi

if [[ -n "$description_field" ]]; then
  pass "Frontmatter description field present"
else
  fail "Frontmatter description field missing"
fi

if [[ -n "$metadata_block" ]]; then
  pass "Frontmatter metadata block present"
else
  fail "Frontmatter metadata block missing"
fi

if [[ -n "$version_field" ]]; then
  pass "Frontmatter metadata.version field present"
else
  fail "Frontmatter metadata.version field missing"
fi

if [[ "$version_field" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  pass "Frontmatter metadata.version is semver-shaped"
else
  fail "Frontmatter metadata.version must be semver-shaped"
fi

if [[ -n "$category_field" ]]; then
  pass "Frontmatter metadata.category field present"
else
  fail "Frontmatter metadata.category field missing"
fi

if [[ -n "$tags_field" ]]; then
  pass "Frontmatter metadata.tags field present"
else
  fail "Frontmatter metadata.tags field missing"
fi

if [[ "$name_field" == "$folder_name" ]]; then
  pass "Frontmatter name matches folder"
else
  fail "Frontmatter name does not match folder"
fi

if [[ "$name_field" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  pass "Frontmatter name is kebab-case"
else
  fail "Frontmatter name must be kebab-case"
fi

if printf '%s' "$frontmatter" | grep -Eq '[<>]'; then
  fail "Frontmatter contains forbidden angle brackets"
else
  pass "Frontmatter has no forbidden angle brackets"
fi

description_plain="$description_field"
description_plain="${description_plain#\"}"
description_plain="${description_plain%\"}"

if ((${#description_plain} <= 1024)); then
  pass "Description length <= 1024"
else
  fail "Description exceeds 1024 chars"
fi

if printf '%s' "$description_plain" | grep -Eiq 'use when'; then
  pass "Description contains usage trigger guidance"
else
  fail "Description should include a 'Use when' clause"
fi

if printf '%s' "$description_plain" | grep -Eiq '(do not use|not for)'; then
  pass "Description includes negative trigger guidance"
else
  printf 'WARN: Description has no explicit negative trigger clause\n'
fi

required_headings=(
  "## Scope"
  "## Success Criteria"
  "## Workflow"
  "## Trigger Tests"
)

for heading in "${required_headings[@]}"; do
  require_heading "$heading"
done

test_matrix_file="$skill_dir/references/test-matrix.md"
if [[ -f "$test_matrix_file" ]]; then
  if rg -q 'TODO|TBD|FIXME' "$test_matrix_file"; then
    fail "references/test-matrix.md should not contain placeholder markers"
  else
    pass "references/test-matrix.md has no placeholder markers"
  fi

  if rg -q '^\| (Scenario|Test) \| (Prompt|Prompt type|Prompt or setup) \| (Expected Behavior|Expected result) \| (Pass/Fail|Validation Check|Validation rule|Evidence|Evidence / Threshold) \|$' "$test_matrix_file"; then
    pass "references/test-matrix.md uses approved structured table header"
  else
    fail "references/test-matrix.md must use an approved structured table header"
  fi

  if rg -q '^\| Scenario \| Prompt \| Expected Behavior \| (Pass/Fail|Validation Check|Validation rule|Evidence / Threshold) \|$' "$test_matrix_file"; then
    pass "references/test-matrix.md uses scenario-driven workflow matrix"

    for scenario in "Trigger positive" "Trigger negative" "Functional core" "Functional edge"; do
      if rg -q "^\| ${scenario} \|" "$test_matrix_file"; then
        pass "references/test-matrix.md includes scenario row: ${scenario}"
      else
        fail "references/test-matrix.md missing scenario row: ${scenario}"
      fi
    done

    if rg -q '^\| (Repeatability|Reliability) \|' "$test_matrix_file"; then
      pass "references/test-matrix.md includes repeatability or reliability row"
    else
      fail "references/test-matrix.md missing repeatability or reliability row"
    fi

    if rg -q '^\| Scenario \| Prompt \| Expected Behavior \| Pass/Fail \|$' "$test_matrix_file"; then
      if awk -F'|' '
        BEGIN { ok=1 }
        NR>2 && $0 ~ /^\|/ {
          scenario=$2
          validation=$(NF-1)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", scenario)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", validation)
          if (scenario ~ /^(Trigger positive|Trigger negative|Functional core|Functional edge|Repeatability|Reliability)$/ &&
              validation !~ /^Pass (if|when) /) {
            ok=0
          }
        }
        END { exit ok ? 0 : 1 }
      ' "$test_matrix_file"; then
        pass "references/test-matrix.md includes concrete pass criteria"
      else
        fail "references/test-matrix.md must use concrete 'Pass if' or 'Pass when' checks"
      fi
    fi
  fi
fi

agent_manifest="$skill_dir/agents/openai.yaml"

if [[ -f "$agent_manifest" ]]; then
  pass "agents/openai.yaml exists"

  if grep -Eq '^interface:[[:space:]]*$' "$agent_manifest"; then
    pass "Agent manifest interface block present"
  else
    fail "Agent manifest interface block missing"
  fi

  if grep -Eq '^[[:space:]]{2}display_name:[[:space:]]*.+$' "$agent_manifest"; then
    pass "Agent manifest interface.display_name present"
  else
    fail "Agent manifest interface.display_name missing"
  fi

  if grep -Eq '^[[:space:]]{2}short_description:[[:space:]]*.+$' "$agent_manifest"; then
    pass "Agent manifest interface.short_description present"
  else
    fail "Agent manifest interface.short_description missing"
  fi

  if grep -Eq '^[[:space:]]{2}default_prompt:[[:space:]]*.+$' "$agent_manifest"; then
    pass "Agent manifest interface.default_prompt present"
  else
    fail "Agent manifest interface.default_prompt missing"
  fi

  if grep -Eq '^policy:[[:space:]]*$' "$agent_manifest"; then
    if grep -Eq '^[[:space:]]{2}allow_implicit_invocation:[[:space:]]*(true|false)$' "$agent_manifest"; then
      pass "Agent manifest policy.allow_implicit_invocation is valid"
    else
      fail "Agent manifest policy.allow_implicit_invocation must be true or false when policy is present"
    fi
  fi

  if grep -Eq 'type:[[:space:]]*"mcp"' "$agent_manifest"; then
    pass "Agent manifest declares MCP dependency"

    if grep -Eq 'value:[[:space:]]*".+"' "$agent_manifest"; then
      pass "Agent manifest MCP dependency value present"
    else
      fail "Agent manifest MCP dependency value missing"
    fi

    if grep -Eq 'description:[[:space:]]*".+"' "$agent_manifest"; then
      pass "Agent manifest MCP dependency description present"
    else
      fail "Agent manifest MCP dependency description missing"
    fi

    if grep -Eq 'transport:[[:space:]]*".+"' "$agent_manifest"; then
      pass "Agent manifest MCP dependency transport present"
    else
      fail "Agent manifest MCP dependency transport missing"
    fi

    if grep -Eq 'url:[[:space:]]*"https?://.+"' "$agent_manifest"; then
      pass "Agent manifest MCP dependency URL present"
    else
      fail "Agent manifest MCP dependency URL missing"
    fi
  fi
else
  pass "agents/openai.yaml not present"
fi

if ((failures > 0)); then
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi

printf 'Audit finished with no failures.\n'
