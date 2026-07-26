#!/usr/bin/env bash
# PQ speculation divergence: from the exact search's beam at a window's lower edge, how far does
# a PQ scout diverge INSIDE the window? (tools/pq_specdiverge.cpp). Sliding windows of the sizes
# asked for -- 30 and 60 -- reporting expanded-node-set drift (Jaccard) and fetch-target ranking
# agreement (Kendall/top1/pool_ov) per window.
#
#   ./pq_specdiverge.sh                    # default: WIDTHS=30,60 swept over offsets 0,w,2w,...
#   WIDTHS=30 MAXOFF=300 ./pq_specdiverge.sh
#   NQ=20000 ./pq_specdiverge.sh           # cap queries for a faster pass
set -euo pipefail

REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/sift/sift100m_flatnav.bin}
Q=${Q:-/data/sift/sift100m_200k_extra_query.fvecs}
T=${T:-32}
EF=${EF:-200}
K=${K:-100}
BIN=$HOME/pq_specdiverge_bench
SRC=$REPO/tools/pq_specdiverge.cpp
OUT=${OUT:-$HOME/pq_specdiverge_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 pq_specdiverge.cpp"
  g++ -std=c++17 -O3 -march=native \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

# Pinned to node 0, matching the rest of the study (search dynamics are placement-invariant).
# DRIFT_CSV -> per-window drift histogram (20 bins over [0,1]) for plotting the distribution.
DRIFT_CSV=${DRIFT_CSV:-$OUT/drift_hist.csv} \
numactl --cpunodebind=0 "$BIN" "$IDX" "$Q" "$T" "$EF" "$K" \
  2>&1 | tee "$OUT/sweep.log" | grep --line-buffered -E "SPEC|drift dist|^\[load\]|^\[train\]|^\[encode\]|^\[exact\]|^#"

echo "raw: $OUT/sweep.log"
