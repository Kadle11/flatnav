#!/usr/bin/env bash
# Speculate-then-validate on SIFT100M with tiered placement (tools/spec_search.cpp).
# Paths come from setup_100m.sh (via $ROOT/.spec_paths.env) unless given explicitly.
#
#   ./run_specval_100m.sh                    # all three sections
#   ONLY=validate ./run_specval_100m.sh      # one section: validate|baseline|spec
#   NQ=50000 ./run_specval_100m.sh           # more queries for the timed sections
#   DEPTHS=1,2 ./run_specval_100m.sh         # narrower depth sweep
#
# Sections
#   validate  bit-exactness of the pipeline at 100M scale. For every depth the returned top-K,
#             the expansion order and the exact-read count must match the exact search. Runs on
#             the default allocator (correctness does not depend on placement) and few queries,
#             since the check is per query and the grid is the cost.
#   baseline  exact search only (no speculation), timed under both placements: everything on the
#             near node, then vectors on the far node. The gap between them is the latency the
#             pipeline is trying to hide, and it is the number the spec runs must beat.
#   spec      the pipeline under far-node vectors, twice: without staging helpers, then with.
#             Without helpers isolates what speculation COSTS (PQ scoring plus discarded work);
#             with helpers adds what it BUYS (far->near copies issued k steps early). Attributing
#             any win needs all three points, which is why the no-helper run is not skipped.
#
# Placement: quad-socket Xeon Gold 6530. Near node 0 (cpus 0-15,64-79; 128 GB) holds the graph,
# the PQ codes and the staging buffer; far node 3 (cpus 48-63,112-127) holds the vectors. The
# index is ~60 GB, so each half fits its node comfortably. The process is pinned to node 0's cpus
# throughout, so only the vector reads cross the interconnect.
#
# Cost note: every invocation retrains the PQ codebook and re-encodes 100M nodes (~1-2 min on
# this node). Six invocations pay that six times; it is not cached between runs.
set -euo pipefail

ROOT=${ROOT:-$HOME/vishal}
PATHS=${PATHS:-$ROOT/.spec_paths.env}
# shellcheck disable=SC1090
[ -f "$PATHS" ] && . "$PATHS"        # only fills IDX/Q/GT/REPO when they are unset

IDX=${IDX:-}
Q=${Q:-}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
BIN=${BIN:-$HOME/spec_search_bench}
SRC=$REPO/tools/spec_search.cpp

T=${T:-16}                       # search threads
EF=${EF:-200}
K=${K:-100}
NQ=${NQ:-20000}                  # queries for the timed sections
NQ_VALIDATE=${NQ_VALIDATE:-2000} # queries for the correctness section
DEPTHS=${DEPTHS:-1,2,4,8}
PQ_M=${PQ_M:-16}

NEAR=${NEAR:-0}                  # graph, PQ codes, staging buffer, and all cpus
FAR=${FAR:-3}                    # vectors
HELPER_CPUS=${HELPER_CPUS:-76,77,78,79}   # node-0 cpus for the staging helpers
BUF_SLOTS=${BUF_SLOTS:-4194304}           # staging capacity in vectors (~2 GB at 512 B)

ONLY=${ONLY:-all}
OUT=${OUT:-$ROOT/specval_100m_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

for v in IDX Q; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -s "$p" ] || { echo "missing $v (${p:-unset}) -- run ./setup_100m.sh, or set $v="; exit 1; }
done
[ -d "$REPO/include/flatnav" ] || { echo "no flatnav checkout at $REPO (set REPO=)"; exit 1; }

# Unlike the other PQ tools this one needs libnuma: it places the two halves of the index on
# different nodes and allocates the staging buffer on the near one.
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -DFLATNAV_USE_NUMA spec_search.cpp"
  g++ -std=c++17 -O3 -march=native -DFLATNAV_USE_NUMA \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread -lnuma \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

PIN=(numactl --cpunodebind="$NEAR")
run() { local tag=$1; shift; echo "[run] $tag"; "$@" > "$OUT/$tag.log" 2>&1 || { echo "  FAILED (see $OUT/$tag.log)"; return 1; }; }
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

echo "[cfg] idx=$IDX"
echo "[cfg] q=$Q nq=$NQ (validate $NQ_VALIDATE) ef=$EF K=$K threads=$T depths=$DEPTHS"
echo "[cfg] near=$NEAR far=$FAR helpers=$HELPER_CPUS -> $OUT"
numactl -H | grep -E "^node ($NEAR|$FAR) (cpus|size|free)" || true

# --- 1. validate: the pipeline returns exactly what the exact search returns ----------------
if want validate; then
  run validate env NQ="$NQ_VALIDATE" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" DIAG=1 \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
  # The oracle pass scores speculation with exact distances: ranking errors and the reachability
  # floor must both go to zero, which separates the pipeline machinery from PQ's own error.
  run validate_oracle env NQ="$NQ_VALIDATE" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" DIAG=1 ORACLE=1 \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- 2. baseline: exact search, near vs far vectors, no speculation -------------------------
# DEPTHS= runs the exact pass and skips the grid entirely.
if want baseline; then
  run baseline_near env NQ="$NQ" DEPTHS="" PQ_M="$PQ_M" \
      VEC_NODE="$NEAR" GRAPH_NODE="$NEAR" \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
  run baseline_far env NQ="$NQ" DEPTHS="" PQ_M="$PQ_M" \
      VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- 3. spec: far vectors, without then with staging helpers -------------------------------
if want spec; then
  run spec_nostage env NQ="$NQ" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" \
      VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
  run spec_stage env NQ="$NQ" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" \
      VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
      PF_HELPER_CPUS="$HELPER_CPUS" PF_BUF_SLOTS="$BUF_SLOTS" PF_LOCAL_NODE="$NEAR" \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- summary -------------------------------------------------------------------------------
echo
echo "=== summary ($OUT) ==="

if [ -s "$OUT/validate.log" ]; then
  echo "-- correctness (must read PASS; oracle must show order% and floor% at 0.00) --"
  for f in validate validate_oracle; do
    [ -s "$OUT/$f.log" ] || continue
    printf '%-16s %s\n' "$f" "$(grep -E '^(PASS|FAIL)' "$OUT/$f.log" | head -1)"
    grep -E "^      FAIL" "$OUT/$f.log" || true
  done
  echo "-- oracle rows (k w checks/q miss% floor% rej% tie% order% wasted/q) --"
  grep -E "^[0-9]+ +1 " "$OUT/validate_oracle.log" 2>/dev/null || true
fi

echo
echo "-- exact baseline, near vs far vectors --"
for f in baseline_near baseline_far; do
  [ -s "$OUT/$f.log" ] || continue
  printf '%-16s %s\n' "$f" "$(grep '^\[exact\]' "$OUT/$f.log")"
done

echo
echo "-- pipeline, far vectors (k w checks/q miss% disc/q reads/q wasted/q stalls/q qps) --"
for f in spec_nostage spec_stage; do
  [ -s "$OUT/$f.log" ] || continue
  echo "[$f]"
  grep '^\[exact\]' "$OUT/$f.log" || true
  grep -E "^[0-9]+ +1 |^      staged" "$OUT/$f.log" || true
done

echo
echo "logs: $OUT"
