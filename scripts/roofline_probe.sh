#!/usr/bin/env bash
# Builds tools/roofline_probe.cpp once and sweeps the working-set size N for the
# real SquaredL2Distance kernel. No NUMA, no graph, single-threaded -- isolates
# whether the kernel itself is compute-bound (throughput flat in cache) or
# memory-bound (throughput falls and re-flattens once N*dim*4B >> LLC).
#
#   DIM=128 DUR=2 ./scripts/roofline_probe.sh
#
# Default N range caps at 4 GB (N_MAX=1<<23) to stay well under available RAM
# and avoid swap-thrashing skewing the largest points.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
BIN=$REPO/.claude/assets/roofline_probe_bench
SRC=$REPO/tools/roofline_probe.cpp

DIM=${DIM:-128}
N_MIN=${N_MIN:-1024}
N_MAX=${N_MAX:-8388608}   # 8M vectors * 128 * 4B = 4 GB
DUR=${DUR:-2}
OUT=$REPO/.claude/assets/roofline_probe_$(date +%Y%m%d_%H%M%S).log

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -march=native roofline_probe.cpp"
  g++ -O3 -march=native -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN"
fi

L1=$(lscpu | awk -F: '/L1d cache/{print $2}' | xargs)
L2=$(lscpu | awk -F: '/L2 cache/{print $2}' | xargs)
L3=$(lscpu | awk -F: '/L3 cache/{print $2}' | xargs)
{
  echo "# dim=$DIM N=[$N_MIN,$N_MAX] dur=${DUR}s/point  L1d=$L1 L2=$L2 L3=$L3"
  "$BIN" "$DIM" "$N_MIN" "$N_MAX" "$DUR"
} | tee "$OUT"

echo "log: $OUT"
