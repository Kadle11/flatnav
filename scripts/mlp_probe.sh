#!/usr/bin/env bash
# Builds tools/mlp_probe.cpp once and sweeps K (independent vectors in flight per
# group) under perf, on the remote CloudLab node. Confirms/refutes the "cores are
# MLP-starved" hypothesis: if throughput rises and mem_stall falls as K grows, the
# short-and-many stalls can be grouped into large-and-few ones by batching, and the
# saturation K sizes the query batch.
#
#   DATANODE=1 CPUNODE=0 T=16 DUR=5 ./scripts/mlp_probe.sh
#
# Sweep T=16 first (1 thread/phys-core, isolates per-core MLP from SMT), then T=32.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
BIN=$HOME/mlp_probe_bench
SRC=$REPO/tools/mlp_probe.cpp

N=${N:-$(( 8 * 1024 * 1024 * 1024 / 512 ))}   # ~8 GB working set (>> 22 MB L3)
DIM=${DIM:-128}
PFL=${PFL:-1}                                  # first-lines prefetched/vector (0 = pure demand)
T=${T:-16}
DATANODE=${DATANODE:-1}                        # remote (vectors) node
CPUNODE=${CPUNODE:-0}                          # compute node
DUR=${DUR:-5}
KS=${KS:-"1 2 4 6 8 10 12 16 24"}
OUT=$HOME/mlp_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

# --- build once (rebuild if source newer) ---
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/util/NumaAllocation.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -march=native -DFLATNAV_USE_NUMA mlp_probe.cpp"
  g++ -std=c++17 -O3 -march=native -DFLATNAV_USE_NUMA \
    -I "$REPO/include" \
    "$SRC" -o "$BIN" -lnuma -lpthread
fi

# mem_stall = cycle_activity.stalls_mem_any / cycles ; remote BW = node1 IMC read.
EVENTS="cycles,cycle_activity.stalls_mem_any,uncore_imc/cas_count.read/,uncore_imc/cas_count.write/"

echo "N=$N dim=$DIM PFL=$PFL T=$T data=node$DATANODE cpu=node$CPUNODE dur=${DUR}s" | tee "$OUT/summary.txt"
echo "K   Macc/s   /thread   GB/s   mem_stall%   S1_rd_GBs" | tee -a "$OUT/summary.txt"

for K in $KS; do
  LOG="$OUT/k${K}.log"
  perf stat -a --per-socket -e "$EVENTS" -o "$LOG" -- \
    "$BIN" "$N" "$DIM" "$K" "$PFL" "$T" "$DATANODE" "$CPUNODE" "$DUR" > "$OUT/k${K}.out" 2>&1 || true

  # parse: throughput from the tool, mem_stall from perf, node1 read BW from --per-socket
  read MACC PERTH GBS < <(awk '/Macc\/s/{for(i=1;i<=NF;i++){if($i=="Macc/s")m=$(i-1);if($i=="/thread")t=$(i-1);if($i=="GB/s")g=$(i-1)}print m,t,g}' "$OUT/k${K}.out")
  CYC=$(awk '/cycles/{gsub(/,/,"",$1);c=$1}END{print c}' "$LOG")
  STL=$(awk '/stalls_mem_any/{gsub(/,/,"",$1);s=$1}END{print s}' "$LOG")
  STALLPCT=$(awk -v s="$STL" -v c="$CYC" 'BEGIN{if(c>0)printf "%.1f",100*s/c; else print "?"}')
  # node1 (S1) IMC read: perf prints per-socket rows "S1 <ncpus> <count> ... cas_count.read"
  S1RD=$(awk '/S1/ && /cas_count.read/{gsub(/,/,"",$3); printf "%.1f", $3*64/1e9/'"$DUR"'}' "$LOG" | tail -1)

  printf "%-3s %-8s %-8s %-6s %-11s %s\n" "$K" "${MACC:-?}" "${PERTH:-?}" "${GBS:-?}" "${STALLPCT:-?}" "${S1RD:-?}" | tee -a "$OUT/summary.txt"
done

echo "logs in $OUT"
