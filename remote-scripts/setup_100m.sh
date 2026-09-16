#!/usr/bin/env bash
# One-time (per node re-image) setup for the speculate-validate experiments on SIFT100M.
# Everything lives under $ROOT (~/vishal): this locates the dataset wherever it sits under it,
# builds the index if only the raw inputs are there, compiles the two tools, and records the
# resolved paths for run_speculation_100m.sh. Safe to re-run: each step is skipped if done.
#
#   ./setup_100m.sh                    # find inputs under ~/vishal, build what is missing
#   ROOT=/mnt/data ./setup_100m.sh     # different root
#   IDX=/path/idx.bin ./setup_100m.sh  # or name any file explicitly
#   SKIP_INDEX=1 ./setup_100m.sh       # tools only
#
# Node: dual-socket, 2 NUMA nodes (one per package, no sub-NUMA clustering). Experiments run
# on node 0 (cpus 0-31,64-95; ~258 GB); node 1 (cpus 32-63,96-127; ~258 GB) is the far node kept
# for later latency work. The index is ~60 GB and is read into one node's memory, so it must fit
# in that node's ~258 GB.
set -euo pipefail

ROOT=${ROOT:-$HOME/vishal}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
OUT_DIR=${OUT_DIR:-$ROOT/index}          # where a newly built index is written
PATHS=${PATHS:-$ROOT/.spec_paths.env}    # resolved paths, sourced by the run script
M=${M:-32}
DEPTH=${DEPTH:-4}                        # how deep under $ROOT to search

# Regenerating the HNSW base-layer graph (BUILD_MTX=1): hours, ~116 GB RAM (hnswlib keeps its
# own copy of the vectors) and ~41 GB of disk for the text .mtx. Deliberately NOT pinned to one
# NUMA node (left as-is even though ~116 GB now fits in one node's ~258 GB).
BUILD_MTX=${BUILD_MTX:-0}
PY=${PY:-poetry run python}              # the extended hnswlib wheel lives in the poetry env
EFC=${EFC:-200}                          # ef_construction for the HNSW build
BUILD_THREADS=${BUILD_THREADS:-$(nproc)}
BUILD_BATCH=${BUILD_BATCH:-250000}
MTX_CHECK_Q=${MTX_CHECK_Q:-1000}         # queries the build script runs afterwards as a sanity check

say() { echo "[setup] $*"; }

# locate <var> <basename> -- set <var> to the first match under $ROOT unless already set
locate() {
  local var=$1 name=$2 cur hit
  eval "cur=\${$var:-}"
  if [ -n "$cur" ]; then
    [ -s "$cur" ] || { say "MISSING (given) $var=$cur"; return 1; }
    say "$var = $cur (given)"
    return 0
  fi
  # Search $ROOT and the checkout: the 200k query set + its ground truth are committed under
  # the repo (SIFT-200K/), while the index and base vectors live under $ROOT.
  hit=$(find "$ROOT" "$REPO" -maxdepth "$DEPTH" -type f -name "$name" -size +0 2>/dev/null | head -1)
  [ -n "$hit" ] || return 1
  eval "$var=\$hit"
  say "$var = $hit ($(du -h "$hit" | cut -f1))"
  return 0
}

say "host $(hostname) | $(nproc) cpus | root $ROOT | repo $REPO"
numactl -H | grep -E "^node (0|1) (cpus|size|free)" || true
[ -d "$ROOT" ] || { say "no such root: $ROOT"; exit 1; }
[ -d "$REPO/include/flatnav" ] || { say "no flatnav checkout at $REPO (set REPO=)"; exit 1; }

# --- 1. queries + ground truth -------------------------------------------------------------
locate Q  'sift100m*query*.fvecs'  || { say "MISSING queries under $ROOT (sift100m*query*.fvecs)"; exit 1; }
locate GT 'sift100m*gtruth*.ivecs' || locate GT 'bigann_gnd*.ivecs' || {
  say "MISSING ground truth under $ROOT (needed by pq_margin only)"; exit 1; }

# --- 2. the index --------------------------------------------------------------------------
if [ "${SKIP_INDEX:-0}" = "1" ]; then
  say "SKIP_INDEX=1"
elif locate IDX 'sift100m*flatnav*.bin'; then
  :
else
  say "no prebuilt index under $ROOT; building from fvecs + mtx (~21 min, mostly the mtx parse)"
  locate BASE 'sift100m*base*.fvecs' || { say "MISSING base vectors (sift100m*base*.fvecs)"; exit 1; }
  if ! locate MTX '*hnsw_base_layer*.mtx'; then
    [ "$BUILD_MTX" = "1" ] || {
      say "MISSING the M=$M HNSW base-layer graph (*hnsw_base_layer*.mtx)."
      say "  Re-run with BUILD_MTX=1 to regenerate it here (hours, ~116 GB RAM, ~41 GB disk),"
      say "  or copy one onto the node. See SIFT_QUICKSTART.md."
      exit 1; }
    mkdir -p "$OUT_DIR"
    MTX=$OUT_DIR/sift100m_m${M}_hnsw_base_layer.mtx
    say "regenerating $MTX with hnswlib (M=$((M / 2)) -> $M links/node, efc=$EFC, $BUILD_THREADS threads)"
    say "  this runs for hours; log: $OUT_DIR/build_mtx.log"
    # --num-node-links N gives hnswlib M = N/2, i.e. N links per node on the base layer, matching
    # the flatnav index built from it. The query pass afterwards is a small sanity check, not a
    # benchmark. Unpinned on purpose: the build's footprint exceeds one node.
    ( cd "$REPO/experiments" && $PY sift_big_flatnav_recall.py \
        --dataset "$BASE" --queries "$Q" --gtruth "$GT" --metric l2 \
        --num-node-links "$M" --ef-construction "$EFC" \
        --num-build-threads "$BUILD_THREADS" --build-batch-size "$BUILD_BATCH" \
        --num-search-threads "$BUILD_THREADS" --ef-search 200 --k 100 \
        --num-queries "$MTX_CHECK_Q" --graph-tmp-dir "$OUT_DIR" \
        --save-mtx "$MTX" --output-json "$OUT_DIR/build_mtx_recall.json" ) 2>&1 | tee "$OUT_DIR/build_mtx.log"
    [ -s "$MTX" ] || { say "mtx generation failed, see $OUT_DIR/build_mtx.log"; exit 1; }
    say "MTX = $MTX ($(du -h "$MTX" | cut -f1))"
  fi
  mkdir -p "$OUT_DIR"
  IDX=$OUT_DIR/sift100m_flatnav.bin
  g++ -std=c++17 -O3 -march=native -fopenmp \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$REPO/tools/build_sift100m.cpp" -o "$HOME/build_sift100m" -lpthread
  "$HOME/build_sift100m" "$BASE" "$MTX" "$IDX" "$M"
fi

# --- 3. the tools --------------------------------------------------------------------------
# Same flags as the other PQ tools (scripts/pq_vecerr.sh): no cmake, no numa, no openmp.
build_tool() {  # build_tool <src-basename> <out-basename>
  local src=$REPO/tools/$1 out=$HOME/$2
  if [ -x "$out" ] && [ ! "$src" -nt "$out" ] && [ ! "$REPO/include/flatnav/index/Index.h" -nt "$out" ]; then
    say "have $2 (up to date)"
    return 0
  fi
  say "g++ -O3 $1"
  g++ -std=c++17 -O3 -march=native \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$src" -o "$out" -lpthread 2>&1 | grep -iE "error|undefined" && { say "BUILD FAILED: $1"; exit 1; }
  return 0
}

build_tool pq_top1.cpp pq_top1_bench
build_tool pq_margin.cpp pq_margin_bench

# --- 4. record the paths -------------------------------------------------------------------
# `: ${VAR:=...}` assigns only when unset, so an explicit env var still wins in the run script.
{
  echo "# written by setup_100m.sh on $(date)"
  echo ": \"\${IDX:=${IDX:-}}\""
  echo ": \"\${Q:=$Q}\""
  echo ": \"\${GT:=$GT}\""
  echo ": \"\${REPO:=$REPO}\""
} > "$PATHS"

say "ready:"
printf '  %-10s %s\n' index "${IDX:-<skipped>}" queries "$Q" gtruth "$GT" \
                      pq_top1 "$HOME/pq_top1_bench" pq_margin "$HOME/pq_margin_bench" paths "$PATHS"
say "node 0 memory (index needs ~60 GB + ~6 GB working set, node has ~258 GB):"
numactl -H | grep -E "^node 0 (size|free)"
say "next: ./run_speculation_100m.sh"
