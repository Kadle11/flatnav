#!/usr/bin/env bash
# Same question as scripts/roofline_probe.sh (compute-bound vs memory-bound) but
# for the real graph search instead of the bare kernel: builds tools/phase_time.cpp
# once and runs it on an existing index, then converts its "mean steps/query" and
# "mean us/query" into an effective GB/s of neighbor-vector reads (steps * M links
# * dim * 4B), so it can be compared by eye against the in-cache vs DRAM plateaus
# reported by roofline_probe.sh.
#
#   INDEX=data/sift1m/sift1m_m32.bin QUERIES=data/sift1m/sift/sift_query.fvecs \
#     M=32 DIM=128 EF=200 T=1 ./scripts/roofline_fullsearch.sh
#
# T=1 isolates single-core throughput (comparable to roofline_probe's single-
# threaded kernel numbers); raise T to see the loaded/contended case.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
BIN=$REPO/.claude/assets/phase_time_bench
SRC=$REPO/tools/phase_time.cpp

INDEX=${INDEX:-$REPO/data/sift1m/sift1m_m32.bin}
QUERIES=${QUERIES:-$REPO/data/sift1m/sift/sift_query.fvecs}
M=${M:-32}
DIM=${DIM:-128}
EF=${EF:-200}
T=${T:-1}
OUT=$REPO/.claude/assets/roofline_fullsearch_$(date +%Y%m%d_%H%M%S).log

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -march=native -DFLATNAV_USE_NUMA -DFLATNAV_PROFILE_PHASE phase_time.cpp"
  g++ -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA -DFLATNAV_PROFILE_PHASE \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lnuma -lm
fi

"$BIN" "$INDEX" "$QUERIES" "$EF" "$T" | tee "$OUT"

read US_PER_Q STEPS_PER_Q < <(awk '/mean us\/query/{
  for(i=1;i<=NF;i++){ split($i,a,"="); if(a[1]=="us/query")u=a[2]; if(a[1]=="steps/query")s=a[2] }
  print u, s
}' "$OUT")

GBPS=$(awk -v us="$US_PER_Q" -v steps="$STEPS_PER_Q" -v m="$M" -v dim="$DIM" \
  'BEGIN{ bytes = steps * m * dim * 4; printf "%.3f", bytes / (us * 1e-6) / 1e9 }')

echo
echo "effective read BW: ${STEPS_PER_Q} steps/query * M=$M * dim=$DIM * 4B / ${US_PER_Q}us/query = ${GBPS} GB/s"
echo "compare against the in-cache and DRAM plateaus in the latest roofline_probe log:"
ls -t "$REPO"/.claude/assets/roofline_probe_*.log 2>/dev/null | head -1
echo "log: $OUT"
