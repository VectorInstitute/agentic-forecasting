#!/usr/bin/env bash
# Install the vector-slides skill from the aieng-skills collection into the
# current project. Default: installs to both .cursor/skills/ (Cursor) and
# .claude/skills/ (Claude Code). Pass --cursor or --claude-code to restrict.
# aieng-skills is a PRIVATE VectorInstitute repo — uses authenticated access.
set -euo pipefail

SKILL_NAME="${SKILL_NAME:-vector-slides}"
SKILL_REPO_SLUG="${SKILL_REPO_SLUG:-VectorInstitute/aieng-skills}"
SKILL_REPO_URL="${SKILL_REPO_URL:-https://github.com/${SKILL_REPO_SLUG}.git}"
SKILL_BRANCH="${SKILL_BRANCH:-main}"
TARGET_MODE="both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cursor)      TARGET_MODE="cursor" ;;
    --claude-code) TARGET_MODE="claude-code" ;;
    --target)      TARGET_MODE="${2:?missing value (cursor|claude-code|both)}"; shift ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  SKILL_NAME="$1" ;;
  esac
  shift
done

case "$TARGET_MODE" in
  both)        TARGETS=(".cursor/skills/$SKILL_NAME" ".claude/skills/$SKILL_NAME") ;;
  cursor)      TARGETS=(".cursor/skills/$SKILL_NAME") ;;
  claude-code) TARGETS=(".claude/skills/$SKILL_NAME") ;;
  *) echo "Unknown target: '$TARGET_MODE'. Use: cursor, claude-code, or both." >&2; exit 1 ;;
esac

ROOT="$(pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

clone_collection() {
  local dest="$1"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh repo clone "$SKILL_REPO_SLUG" "$dest" -- --depth 1 --branch "$SKILL_BRANCH"
  elif command -v git >/dev/null 2>&1; then
    git clone --depth 1 --branch "$SKILL_BRANCH" "$SKILL_REPO_URL" "$dest"
  else
    echo "Error: need either 'gh' (authenticated) or 'git' with access to $SKILL_REPO_SLUG." >&2
    exit 1
  fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
clone_collection "$TMP/aieng-skills"

SRC="$TMP/aieng-skills/skills/$SKILL_NAME"
if [[ ! -d "$SRC" ]]; then
  echo "Error: skill '$SKILL_NAME' not found in the collection." >&2
  exit 1
fi

echo "Installing '$SKILL_NAME' → ${TARGETS[*]}"

INSTALLED=()
for TARGET_REL in "${TARGETS[@]}"; do
  ABS="$ROOT/$TARGET_REL"
  mkdir -p "$(dirname "$ABS")"
  rm -rf "$ABS"
  cp -R "$SRC" "$ABS"
  (cd "$ABS" && uv sync)
  INSTALLED+=("$ABS")
done

(cd "${INSTALLED[0]}" && uv run vector-slides doctor 2>/dev/null || true)

if ! command -v soffice >/dev/null 2>&1 && ! [[ -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  echo ""
  echo "Optional (visual QA PNGs): brew install --cask libreoffice poppler"
fi

echo ""
echo "Done. Skill root(s):"
for ABS in "${INSTALLED[@]}"; do
  echo "  $ABS"
done
if [[ " ${TARGETS[*]} " == *" .cursor/skills/$SKILL_NAME "* ]]; then
  echo "Cursor: ensure the '$SKILL_NAME' skill is enabled for this project."
fi
