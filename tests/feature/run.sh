#!/bin/sh
# Compile and run every crunch feature test (feature-*.scm) in this directory.
# The crunch macro emits C through foreign-declare, so the tests must be
# compiled with csc; they cannot be run under csi.
#
# Usage: tests/feature/run.sh   (CHICKEN_BIN defaults to /usr/local/bin)

CHICKEN_BIN=${CHICKEN_BIN:-/usr/local/bin}
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
failures=0

for src in "$here"/feature-*.scm; do
    name=$(basename "$src" .scm)
    if ! (cd "$here" && "$CHICKEN_BIN/csc" -O2 "$name.scm" -o "$out/$name" -L -lpthread) >"$out/$name.log" 2>&1; then
        echo "$name: COMPILE FAILED"; sed -n '1,5p' "$out/$name.log"
        failures=$((failures + 1))
    elif ! "$out/$name"; then
        echo "$name: RUN FAILED"
        failures=$((failures + 1))
    fi
done

echo "feature tests: $failures failure(s)"
exit $failures
