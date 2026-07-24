#!/usr/bin/env bash
# Step-windowed PQ traversal: recall@100 vs the window [lo,hi) of expansions scored on PQ
# (tools/pq_stepgate.cpp). M11 refuted the design's "exactness is only needed LATE" claim using
# prefix windows [0,N); sliding a fixed-width window isolates WHICH phase needs exactness.
#
#   ./pq_stepgate.sh                              # default sliding sweep
#   WINDOWS=0:0,0:40,40:80 ./pq_stepgate.sh       # overrides pass through to the binary
set -euo pipefail

REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/sift/sift100m_flatnav.bin}
Q=${Q:-/data/sift/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/sift/sift100m_200k_extra_query.gtruth.ivecs}
K=${K:-100}
T=${T:-32}
EF=${EF:-200}
BIN=$HOME/pq_stepgate_bench
SRC=$REPO/tools/pq_stepgate.cpp
OUT=${OUT:-$HOME/pq_stepgate_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 pq_stepgate.cpp"
  g++ -std=c++17 -O3 -march=native \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

# Pinned to node 0, matching the rest of the study. Search dynamics are placement-invariant
# (M8/M10), so the gate=0 row should reproduce the known ef=200 baseline recall.
# --line-buffered so progress reaches a redirected stdout as it happens, not at exit.
numactl --cpunodebind=0 "$BIN" "$IDX" "$Q" "$GT" "$K" "$T" "$EF" \
  2>&1 | tee "$OUT/sweep.log" | grep --line-buffered -E "RESULT|^\[load\]|^\[train\]|^\[encode\]|^\[pq-err\]"

echo "raw: $OUT/sweep.log"
