#!/usr/bin/env bash
#
# #269 -- build and run the ASan/UBSan gate over the vendored FastPFor decode path.
#
#   bash rust/asan/run.sh                  # gate the CURRENT tree; expects HARNESS CLEAN, exit 0
#   bash rust/asan/run.sh --prefix-ref REF # NEGATIVE CONTROL: same gate, but with
#                                          # headers/simdfastpfor.h taken from git REF
#
# The negative control is the point. A sanitizer run that passes proves nothing on its own --
# it could pass because the harness never reaches the defect. So the control has to reproduce a
# SPECIFIC defect, not merely fail somehow.
#
# It therefore runs one isolated case: `gate cexcept-overclaim`. The full sweep is NOT a valid
# control, because against the pre-patch header it dies on the first wheremeta case -- a wild
# pointer that segfaults with or without a sanitizer, proving nothing about reads that stay
# inside mapped memory. `cexcept-overclaim` is the opposite: pre-fix it neither crashed nor
# raised (its first invalid read lands one byte past the input buffer, onto mapped memory,
# and the walk carries on from there), so ONLY a
# sanitizer can see it. The control rejects a clean run and a harness fault (exit 2/3), and
# then requires three independent markers in the output -- see the checks below. A bare
# "something failed and a sanitizer mentioned __decodeArray" is not accepted, because an
# ordinary segfault produces exactly that.
#
# The current tree is gated twice: with assertions live and with `-DNDEBUG`. `rust/build.rs`
# never defines NDEBUG, so the assert-enabled build is what actually ships; the NDEBUG build
# matches the control's flags, keeping that comparison to a single variable (the header).
# For the record, the patched header compiles and passes cleanly under `-DNDEBUG` -- what the
# patch changed is that `length` became load-bearing in ordinary code instead of living only
# inside an `assert`, so NDEBUG can no longer silently drop the bound. It does not fail to
# compile, and the second variant below is what keeps that statement honest.
#
# Requires g++/gcc with -fsanitize=address,undefined. Rust nightly is not required: the gate
# builds the C++ shim and vendored sources only, deliberately bypassing the Rust layer so that
# any report is unambiguously an input-side violation inside FastPFor itself.

set -uo pipefail

PREFIX_REF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix-ref) PREFIX_REF="${2:?--prefix-ref needs a git ref}"; shift 2 ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

RUST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/cellstream-asan-gate.XXXXXX")" \
  || { echo "mktemp -d failed" >&2; exit 2; }
# Guard the trap target: an empty or bare-root BUILD would make `-o "$BUILD/x.o"` write to /.
case "$BUILD" in
  /*/*) [ -d "$BUILD" ] || { echo "build dir '$BUILD' is not a directory" >&2; exit 2; } ;;
  *) echo "refusing to use build dir '$BUILD'" >&2; exit 2 ;;
esac
trap 'rm -rf -- "$BUILD"' EXIT

# shellcheck disable=SC2054  # the comma is inside -fsanitize=address,undefined, not a separator
SAN=(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -g -O1)

# build_gate <include-dir> <tag> [extra flags...]  ->  "$BUILD/<tag>/gate"
build_gate() {
  local inc="$1" tag="$2"; shift 2
  local obj="$BUILD/$tag" f
  mkdir -p "$obj" || return 2
  g++ -std=c++11 -msse4.1 "${SAN[@]}" "$@" -I "$inc" -c "$RUST_DIR/csrc/pfor_shim.cpp" -o "$obj/shim.o" || return 2
  g++ -std=c++11 -msse4.1 "${SAN[@]}" "$@" -I "$inc" -c "$RUST_DIR/asan/gate.cpp" -o "$obj/gate.o" || return 2
  for f in "$RUST_DIR"/vendor/FastPFor/src/*.cpp; do
    g++ -std=c++11 -msse4.1 "${SAN[@]}" "$@" -I "$inc" -c "$f" -o "$obj/$(basename "$f" .cpp).o" || return 2
  done
  for f in "$RUST_DIR"/vendor/FastPFor/src/*.c; do
    [ -e "$f" ] || continue
    gcc -std=c99 -msse4.1 "${SAN[@]}" "$@" -I "$inc" -c "$f" -o "$obj/$(basename "$f" .c).o" || return 2
  done
  g++ "${SAN[@]}" "$@" "$obj"/*.o -o "$obj/gate" || return 2
}

# LeakSanitizer does not work on this host (it needs ptrace) and would turn every run into a
# fatal error, masking the gate's own exit status -- detect_leaks=0 is required, not cosmetic.
run_gate() {
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
    "$@" 2>&1
}

# ------------------------------------------------------------------ negative control
if [ -n "$PREFIX_REF" ]; then
  mkdir -p "$BUILD/hdr"
  cp "$RUST_DIR"/vendor/FastPFor/headers/*.h "$BUILD/hdr/"
  git -C "$RUST_DIR" show "$PREFIX_REF:rust/vendor/FastPFor/headers/simdfastpfor.h" \
    > "$BUILD/hdr/simdfastpfor.h" \
    || { echo "could not read simdfastpfor.h at $PREFIX_REF" >&2; exit 2; }
  echo "negative control: simdfastpfor.h from $PREFIX_REF ($(wc -l < "$BUILD/hdr/simdfastpfor.h") lines)"
  build_gate "$BUILD/hdr" nc -DNDEBUG || { echo "negative-control build FAILED" >&2; exit 2; }

  out="$(run_gate "$BUILD/nc/gate" cexcept-overclaim)"; rc=$?
  printf '%s\n' "$out"
  echo "CONTROL EXIT=$rc"
  case $rc in
    0) echo "VACUOUS GATE: cexcept-overclaim completed with no sanitizer report" >&2
       exit 1 ;;
    2|3) echo "HARNESS FAULT (exit $rc), not a sanitizer report -- control is invalid" >&2
       exit 1 ;;
  esac
  # Nonzero is not enough, and neither is "a sanitizer said something". Three independent
  # conditions, each ruling out a different way to pass for the wrong reason:
  #   MUTATED DECODE REACHED -- execution got past the UNMUTATED precondition decode, so the
  #                             report cannot be from that one.
  #   heap-buffer-overflow   -- an out-of-bounds access, NOT AddressSanitizer:DEADLYSIGNAL,
  #                             which is just an ordinary segfault a sanitizer happened to see
  #                             and which an exit-code test would have caught anyway.
  #   __decodeArray          -- it happened in the function under test.
  for marker in "MUTATED DECODE REACHED" "heap-buffer-overflow" "__decodeArray"; do
    case "$out" in
      *"$marker"*) : ;;
      *) echo "control output lacks '$marker' -- control is invalid" >&2; exit 1 ;;
    esac
  done
  echo "negative control reproduced the out-of-bounds read inside __decodeArray -- gate is live"
  exit 0
fi

# ------------------------------------------------------------------ current tree
status=0
for tag in asserts-live ndebug; do
  case "$tag" in
    asserts-live) flags=() ;;      # what rust/build.rs actually ships
    ndebug)       flags=(-DNDEBUG) ;;
  esac
  echo "===== gate [current tree, $tag] ====="
  # ${a[@]+"${a[@]}"} rather than "${a[@]}": bash < 4.4 treats an empty array as unset and
  # would abort here under `set -u`.
  build_gate "$RUST_DIR/vendor/FastPFor/headers" "$tag" ${flags[@]+"${flags[@]}"} \
    || { echo "build FAILED [$tag]" >&2; exit 2; }
  run_gate "$BUILD/$tag/gate"; rc=$?
  echo "GATE EXIT=$rc  [current tree, $tag]"
  [ "$rc" -eq 0 ] || status=1
done
exit "$status"
