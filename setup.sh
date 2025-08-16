#!/usr/bin/env bash
#setup.sh — verify ./paramedir -v and update PATH entries

set -Eeuo pipefail

die()  { echo "❌ $*" >&2; exit 1; }
warn() { echo "⚠️  $*" >&2; }
info() { echo "➤ $*"; }
ok()   { echo "✅ $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_profile_file() {
  local shell_name="${SHELL##*/}"
  if [[ "$shell_name" == "zsh" ]]; then
    echo "$HOME/.zshrc"
  elif [[ "$shell_name" == "bash" ]]; then
    [[ -f "$HOME/.bashrc" ]] && echo "$HOME/.bashrc" || echo "$HOME/.bash_profile"
  else
    [[ -f "$HOME/.zshrc" ]] && echo "$HOME/.zshrc" || \
    [[ -f "$HOME/.bashrc" ]] && echo "$HOME/.bashrc" || \
    echo "$HOME/.profile"
  fi
}

ensure_line_in_file() {
  local line="$1"
  local file="$2"
  touch "$file"
  if ! grep -Fq "$line" "$file"; then
    printf "\n# Added by setup.sh on %s\n%s\n" "$(date)" "$line" >> "$file"
    ok "Updated $(basename "$file") with: $line"
  else
    info "Entry already present in $(basename "$file")."
  fi
}

if [[ $# -ne 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  ./setup.sh /path/to/paramedir_folder

Where:
  /path/to/paramedir_folder  A directory that contains an executable named "paramedir"
USAGE
  exit 1
fi

PARAMEDIR_DIR="$1"
PARAMEDIR_BIN="$PARAMEDIR_DIR/paramedir"

info "Checking paramedir at: $PARAMEDIR_BIN"
[[ -e "$PARAMEDIR_BIN" ]] || die "paramedir not found at: $PARAMEDIR_BIN"
[[ -f "$PARAMEDIR_BIN" ]] || die "paramedir exists but is not a regular file: $PARAMEDIR_BIN"
[[ -x "$PARAMEDIR_BIN" ]] || die "paramedir exists but is not executable. Try: chmod +x \"$PARAMEDIR_BIN\""

PARAMEDIR_OK=0
if "$PARAMEDIR_BIN" -v >/dev/null 2>&1; then
  ok "paramedir is runnable with '-v'."
  PARAMEDIR_OK=1
else
  warn "paramedir did not run successfully with '-v'. Will NOT add its folder to PATH."
fi

PROFILE_FILE="$(detect_profile_file)"

REPO_EXPORT_LINE="export PATH=\"$REPO_ROOT:\$PATH\""
info "Persisting repo PATH update in: $PROFILE_FILE"
ensure_line_in_file "$REPO_EXPORT_LINE" "$PROFILE_FILE"

PARA_LINE="skipped"
if [[ "$PARAMEDIR_OK" -eq 1 ]]; then
  PARA_EXPORT_LINE="export PATH=\"$PARAMEDIR_DIR:\$PATH\""
  info "Persisting paramedir PATH update in: $PROFILE_FILE"
  ensure_line_in_file "$PARA_EXPORT_LINE" "$PROFILE_FILE"
  PARA_LINE="$PARAMEDIR_DIR (added)"
fi

cat <<EOF

✅ Setup complete!

Steps performed:
  - Verified paramedir at: $PARAMEDIR_BIN: $([[ "$PARAMEDIR_OK" -eq 1 ]] && echo "OK" || echo "FAILED")
  - Updated PATH in: $PROFILE_FILE
      • Repo root: $REPO_ROOT
      • paramedir dir: $PARA_LINE

Next steps:
  • Open a NEW terminal (or run: source "$PROFILE_FILE") so PATH updates take effect.
  • Then install Python deps (you might need a virtual environment or the flag --break-system-packages):
      pip install -r requirements.txt
  • Run Paraver and launch the CARM GUI from a Paraver timeline or an entire trace.

EOF
