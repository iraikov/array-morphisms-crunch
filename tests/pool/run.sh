#!/bin/sh
# Build and run the stand-alone thread-pool test (pool-tsan.c) twice:
# optimised, and under ThreadSanitizer.
#
# Usage: tests/pool/run.sh   (CC defaults to cc)

CC=${CC:-cc}
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
failures=0

if ! $CC -O2 -Wall -Wextra -pthread "$here/pool-tsan.c" -o "$out/pool" ||
   ! "$out/pool"; then
    echo "pool test (optimised): FAILED"; failures=$((failures + 1))
fi
if ! $CC -O1 -g -fsanitize=thread -pthread "$here/pool-tsan.c" -o "$out/pool-tsan"; then
    echo "pool test (TSan): COMPILE FAILED"; failures=$((failures + 1))
elif ! TSAN_OPTIONS="halt_on_error=1 die_after_fork=0 ${TSAN_OPTIONS}" setarch "$(uname -m)" -R "$out/pool-tsan"; then
    echo "pool test (TSan): FAILED"; failures=$((failures + 1))
fi

echo "pool tests: $failures failure(s)"
exit $failures
