#!/usr/bin/env bash
# Generic FlatNav C++ search benchmark — replaces the per-experiment perf_*.sh.
# Builds tools/hbw_search.cpp once (NUMA-enabled), runs it with configurable NUMA
# placement / pinning / ef, and (optionally) measures DRAM bandwidth with perf,
# gated to the post-warmup search window. Reports QPS@recall (+ BW/IPC/stall/MLP).
#
# All config via env vars (defaults below). Examples:
#   PERF=1 ./bench.sh                              # baseline, measure BW
#   TAG=graphlocal_vecremote GRAPH_NUMA=0 VECTORS_NUMA=1 PERF=1 ./bench.sh
#   EF=40,80,200 PREFETCH=1 ./bench.sh            # prefetch-on ef sweep, QPS only
set -u
REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/index/sift100m_flatnav.bin}
Q=${Q:-/data/queries/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/queries/sift100m_200k_extra_query.gtruth.ivecs}
K=${K:-100}; T=${T:-32}; EF=${EF:-200}
GRAPH_NUMA=${GRAPH_NUMA:--1}      # node for graph links (-1 = default allocator)
VECTORS_NUMA=${VECTORS_NUMA:--1}  # node for vectors     (-1 = default allocator)
CPUNODE=${CPUNODE:-0}             # bind worker threads to this NUMA node's CPUs
PREFETCH=${PREFETCH:-0}           # 0 => -DFLATNAV_DISABLE_PREFETCH, 1 => prefetch on
PERF=${PERF:-0}                   # 1 => perf IMC+core, BW over search window
PEAK=${PEAK:-102.8}               # GB/s peak for util%
GUARD=${GUARD:-2}                 # seconds after warmup_done before accruing perf
TAG=${TAG:-run}
REBUILD=${REBUILD:-0}
BIN=$HOME/hbw_search_bench
SRC=$REPO/tools/hbw_search.cpp
OUT=$HOME/bench_${TAG}_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

# 1. Build once (rebuild if source newer or REBUILD=1). Prefetch flag affects the binary.
PF_FLAG=""; [ "$PREFETCH" = "0" ] && PF_FLAG="-DFLATNAV_DISABLE_PREFETCH"
if [ "$REBUILD" = "1" ] || [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA $PF_FLAG ..."
  g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA $PF_FLAG \
    -I "$REPO/include" -I "$REPO/external/cereal/include" "$SRC" -o "$BIN" -lnuma \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

ENVV="FLATNAV_GRAPH_NUMA=$GRAPH_NUMA FLATNAV_VECTORS_NUMA=$VECTORS_NUMA"
RUN="env $ENVV numactl --cpunodebind=$CPUNODE $BIN $IDX $Q $GT $K $T $EF"
echo "===== $TAG : graph=node$GRAPH_NUMA vectors=node$VECTORS_NUMA workers=node$CPUNODE prefetch=$PREFETCH ef=$EF =====" | tee "$OUT/summary.txt"

# 2. Run (optionally under perf).
if [ "$PERF" = "0" ]; then
  bash -c "$RUN" 2>&1 | tee "$OUT/driver.log" | grep -E "numa|phase|^ef|^[0-9]" | tee -a "$OUT/summary.txt"
else
  IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
  CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"
  sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$OUT/perf.csv" -- \
    bash -c "$RUN" > "$OUT/driver.log" 2>&1
  grep -E "numa|phase|^ef|^[0-9]" "$OUT/driver.log" | tee -a "$OUT/summary.txt"
  WU=$(awk -F= '/warmup_done elapsed_s/{print $NF}' "$OUT/driver.log")
  awk -F, -v PEAK="$PEAK" -v START="${WU:-0}" -v GUARD="$GUARD" '
    $1+0 > START+GUARD && $4!="" { ev=$4; v=$2;
      if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
      else if(ev=="cycles") cyc+=v; else if(ev=="instructions") ins+=v;
      else if(ev=="l1d_pend_miss.pending") p+=v; else if(ev=="l1d_pend_miss.pending_cycles") pc+=v;
      if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
    END{ w=hi-lo; if(w>0) printf "BW=%.1f GB/s  util=%.0f%%  IPC=%.3f  mem_stall=%.1f%%  MLP=%.2f  (window=%.0fs, after warmup+%ds)\n",
         cas/w/1e9, 100*(cas/w/1e9)/PEAK, ins/cyc, 100*pc/cyc, p/pc, w, GUARD;
         else print "no perf rows after warmup — check warmup_done marker" }' "$OUT/perf.csv" | tee -a "$OUT/summary.txt"
fi
echo "raw: $OUT" | tee -a "$OUT/summary.txt"
