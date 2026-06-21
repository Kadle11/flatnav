#!/usr/bin/env bash
# A/B: DRAM bandwidth of search with the ORIGINAL single-block layout
# (interleaved [data][links][label]) vs the SPLIT (separate vectors/graph)
# layout. Same hbw_search.cpp, prefetch ON, same ef, node-0 pinned, search-only
# (accrue perf intervals after the warmup_done marker). One perf -a pass each
# (IMC bandwidth + core counters; no multiplexing).
#
# Requires: ~/hbw_search built against include/ (split) and include_orig/ (orig),
# and both index binaries. Usage: bash ~/flatnav/scripts/perf_layout_ab.sh
set -u
REPO=$HOME/flatnav
Q=/data/queries/sift100m_200k_extra_query.fvecs
GT=/data/queries/sift100m_200k_extra_query.gtruth.ivecs
SPLIT_IDX=/data/index/sift100m_flatnav.bin
ORIG_IDX=/data/index/sift100m_flatnav_orig.bin
CPUS=$(numactl --hardware | sed -n 's/^node 0 cpus: //p' | tr ' ' ',')
EF=${EF:-80}; REPEAT=${REPEAT:-8}; GUARD=${GUARD:-2}; PEAK=${PEAK:-102.8}
EFL=$(yes "$EF" | head -n "$REPEAT" | paste -sd,)
TS=$(date +%Y%m%d_%H%M%S); OUT=$HOME/layoutab_$TS; mkdir -p "$OUT"

echo "=== layout A/B (orig single-block vs split), ef=$EF, prefetch ON $TS ===" | tee "$OUT/SUMMARY.txt"
g++ -O3 -march=native -fopenmp -std=c++17 -I "$REPO/include"      -I "$REPO/external/cereal/include" "$REPO/tools/hbw_search.cpp" -o ~/hbw_search_split 2>/dev/null && echo "built split"
g++ -O3 -march=native -fopenmp -std=c++17 -I "$REPO/include_orig" -I "$REPO/external/cereal/include" "$REPO/tools/hbw_search.cpp" -o ~/hbw_search_orig  2>/dev/null && echo "built orig"

IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"

run(){ # $1=label  $2=binary  $3=index
  local label=$1 bin=$2 idx=$3 prog="$OUT/$1.prog" csv="$OUT/$1.csv"
  [ -f "$idx" ] || { echo "$label: MISSING index $idx" | tee -a "$OUT/SUMMARY.txt"; return; }
  ( cd "$HOME" && sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$csv" -- \
     bash -c "env FLATNAV_PIN_CPUS=$CPUS numactl --membind=0 $bin $idx $Q $GT 10 32 $EFL" ) > "$prog" 2>/dev/null
  local cut rec qps
  cut=$(awk -v w="$(sed -n 's/.*warmup_done elapsed_s=\([0-9.]*\).*/\1/p' "$prog")" -v g="$GUARD" 'BEGIN{print w+g}')
  rec=$(grep -oE 'recall@[0-9]+=[0-9.]+' "$prog" | tail -1); qps=$(grep -E "^$EF[[:space:]]" "$prog" | tail -1 | awk '{print $3}')
  awk -F, -v CUT="$cut" -v PEAK="$PEAK" -v LB="$label" -v R="$rec" -v Q="$qps" '
    $1+0>CUT && $4!="" { ev=$4; v=$2;
      if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
      else if(ev=="cycles") cyc+=v; else if(ev=="instructions") ins+=v;
      else if(ev=="l1d_pend_miss.pending") p+=v; else if(ev=="l1d_pend_miss.pending_cycles") pc+=v;
      if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
    END{ w=hi-lo; bw=cas/w/1e9;
      printf "%-6s QPS=%-7s %s  IPC=%.3f  mem_stall=%.1f%%  MLP=%.2f  BW=%.1f GB/s  util=%.0f%%\n",
        LB, Q, R, ins/cyc, 100*pc/cyc, p/pc, bw, 100*bw/PEAK }' "$csv" | tee -a "$OUT/SUMMARY.txt"
}

run orig  ~/hbw_search_orig  "$ORIG_IDX"
run split ~/hbw_search_split "$SPLIT_IDX"
echo "raw: $OUT" | tee -a "$OUT/SUMMARY.txt"
