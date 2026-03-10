#!/usr/bin/env bash
#
# Build rethoc release binaries for a range of commits.
#
# Usage:
#   scripts/build-bisect-bins.sh <start-commit>          # from <start-commit> to HEAD
#   scripts/build-bisect-bins.sh <start-commit> <end>     # from <start-commit> to <end>
#
# Binaries are placed in bins/ as:
#   rethoc-<NNN>-<short-hash>
# where NNN is the 1-based position of the commit on the branch
# (oldest = 001).
#
# The script stashes any uncommitted changes, builds each commit in
# sequence, then restores the original branch and stash.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <start-commit> [<end-commit>]" >&2
    exit 1
fi

START="$1"
END="${2:-HEAD}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
BIN_DIR="$REPO_ROOT/bins"
mkdir -p "$BIN_DIR"

# Remember where we are so we can restore later.
ORIG_REF="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD)"

# Stash uncommitted changes if any.
STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -m "build-bisect-bins auto-stash"
    STASHED=true
fi

cleanup() {
    git checkout --quiet "$ORIG_REF" 2>/dev/null || true
    if $STASHED; then
        git stash pop --quiet 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Collect the full commit list for the branch (oldest first) so we can
# assign stable sequence numbers.  We use the merge-base with the
# branch's first parent to enumerate all branch commits.
BRANCH_BASE="$(git rev-list --max-parents=0 HEAD | head -1)"
mapfile -t ALL_COMMITS < <(git rev-list --reverse "$BRANCH_BASE"..HEAD)
# Prepend the root commit itself.
ALL_COMMITS=("$BRANCH_BASE" "${ALL_COMMITS[@]}")

# Build a commit→number map.
declare -A COMMIT_NUM
for i in "${!ALL_COMMITS[@]}"; do
    COMMIT_NUM["${ALL_COMMITS[$i]}"]=$((i + 1))
done

# Resolve the requested range to full hashes.
mapfile -t RANGE_COMMITS < <(git rev-list --reverse "$START^..$END")

TOTAL=${#RANGE_COMMITS[@]}
echo "Building $TOTAL commits ($START..$END)"
echo "Output directory: $BIN_DIR"
echo

BUILT=0
SKIPPED=0
FAILED=0

for HASH in "${RANGE_COMMITS[@]}"; do
    NUM="${COMMIT_NUM[$HASH]:-0}"
    SHORT="$(git rev-parse --short "$HASH")"
    PADDED="$(printf '%03d' "$NUM")"
    NAME="rethoc-${PADDED}-${SHORT}"
    DEST="$BIN_DIR/$NAME"

    if [[ -f "$DEST" ]]; then
        echo "[$((BUILT + SKIPPED + FAILED + 1))/$TOTAL] $NAME — already exists, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -n "[$((BUILT + SKIPPED + FAILED + 1))/$TOTAL] $NAME — building... "
    git checkout --quiet "$HASH"

    if cargo build --release --quiet 2>/dev/null; then
        cp "$REPO_ROOT/target/release/rethoc" "$DEST"
        echo "ok"
        BUILT=$((BUILT + 1))
    else
        echo "FAILED (build error)"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "Done: $BUILT built, $SKIPPED skipped, $FAILED failed"
