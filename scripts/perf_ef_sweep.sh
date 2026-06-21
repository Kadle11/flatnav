#!/usr/bin/env bash
# ef sweep on one machine state: for each ef, ONE perf -a pass captures IMC DRAM
# bandwidth + core counters (cycles, instructions, l1d_pend_miss.{pending,pending_cycles})
# together (uncore on IMC PMUs + 2 fixed + 2 GP core => no multiplexing). Load phase
# excluded via the hbw_search warmup_done marker. Emits one consistent curve:
#   ef  recall@10  QPS  IPC  mem_stall%  MLP  BW(GB/s)  BWutil%
#
# Usage:  bash ~/flatnav/scripts/perf_ef_sweep.sh        (uses sudo for perf)
set -u
REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/index/sift100m_flatnav.bin}
Q=${Q:-/data/queries/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/queries/sift100m_200k_extra_query.gtruth.ivecs}
CPUS=$(numactl --hardware | sed -n 's/^node 0 cpus: //p' | tr ' ' ',')
GUARD=${GUARD:-2}
PEAK=${PEAK:-102.8}
TS=$(date +%Y%m%d_%H%M%S); OUT=$HOME/efsweep_$TS; mkdir -p "$OUT"

echo "=== ef sweep (search-only, node-0 pinned)  $TS ===" | tee "$OUT/SUMMARY.txt"
g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_DISABLE_PREFETCH \
  -I "$REPO/include" -I "$REPO/external/cereal/include" \
  "$REPO/tools/hbw_search.cpp" -o "$HOME/hbw_search_ts" || { echo BUILD_FAIL; exit 1; }

IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"

printf "%-5s %-9s %-8s %-6s %-11s %-6s %-9s %-7s\n" ef recall QPS IPC mem_stall% MLP BW_GBps util% | tee -a "$OUT/SUMMARY.txt"

# reps per ef so each steady window is ~40-50s
reps_for(){ case $1 in 40) echo 12;; 80) echo 8;; 120) echo 6;; 200) echo 5;; *) echo 6;; esac; }

for EFV in 40 80 120 200; do
  REPEAT=$(reps_for "$EFV")
  EF=$(yes "$EFV" | head -n "$REPEAT" | paste -sd,)
  prog="$OUT/ef${EFV}.prog"; csv="$OUT/ef${EFV}.csv"
  sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$csv" -- \
    bash -c "env FLATNAV_PIN_CPUS=$CPUS numactl --membind=0 $HOME/hbw_search_ts $IDX $Q $GT 10 32 $EF" \
    > "$prog" 2>/dev/null
  cut=$(awk -v w="$(sed -n 's/.*warmup_done elapsed_s=\([0-9.]*\).*/\1/p' "$prog")" -v g="$GUARD" 'BEGIN{print w+g}')
  recall=$(grep -E "^$EFV[[:space:]]" "$prog" | tail -1 | awk '{print $4}')
  qps=$(grep -E "^$EFV[[:space:]]" "$prog" | tail -1 | awk '{print $3}')
  awk -F, -v CUT="$cut" -v PEAK="$PEAK" -v EFV="$EFV" -v R="$recall" -v QPS="$qps" '
    $1+0>CUT && $4!="" { ev=$4; v=$2;
      if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
      else if(ev=="cycles") cyc+=v
      else if(ev=="instructions") ins+=v
      else if(ev=="l1d_pend_miss.pending") pend+=v
      else if(ev=="l1d_pend_miss.pending_cycles") pc+=v
      if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
    END{ w=hi-lo; bw=cas/w/1e9;
      printf "%-5s %-9s %-8s %-6.3f %-11.1f %-6.2f %-9.1f %-7.0f\n",
        EFV, R, QPS, ins/cyc, 100*pc/cyc, pend/pc, bw, 100*bw/PEAK }' "$csv" | tee -a "$OUT/SUMMARY.txt"
done
echo "raw: $OUT" | tee -a "$OUT/SUMMARY.txt"
