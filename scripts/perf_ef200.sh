#!/usr/bin/env bash
# Prolonged ef=200 run: recall + search-only perf (stalls, MLP, load sources) +
# DRAM bandwidth / utilization. Load phase excluded via the hbw_search phase
# markers (accrue perf intervals only after warmup_done + GUARD). No multiplexing
# (each group <=4 GP counters). Prints a SUMMARY and saves raw data.
#
# Usage:  bash ~/flatnav/scripts/perf_ef200.sh         (uses sudo for perf)
set -u
REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/index/sift100m_flatnav.bin}
Q=${Q:-/data/queries/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/queries/sift100m_200k_extra_query.gtruth.ivecs}
CPUS=$(numactl --hardware | sed -n 's/^node 0 cpus: //p' | tr ' ' ',')
EFV=${EFV:-200}
REPEAT=${REPEAT:-8}                        # ef=200 over 200k ~12.5s each -> ~100s search
GUARD=${GUARD:-2}
PEAK=${PEAK:-102.8}                        # node-0 streaming-read peak GB/s (measured)
EF=$(yes "$EFV" | head -n "$REPEAT" | paste -sd,)
TS=$(date +%Y%m%d_%H%M%S); OUT=$HOME/ef${EFV}_$TS; mkdir -p "$OUT"
RUNBASE="env FLATNAV_PIN_CPUS=$CPUS numactl --membind=0 $HOME/hbw_search_ts $IDX $Q $GT 10 32 $EF"

echo "=== prolonged ef=$EFV x$REPEAT  (search-only, node-0 pinned)  $TS ==="
g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_DISABLE_PREFETCH \
  -I "$REPO/include" -I "$REPO/external/cereal/include" \
  "$REPO/tools/hbw_search.cpp" -o "$HOME/hbw_search_ts" || { echo BUILD_FAIL; exit 1; }

cut_of(){ awk -v w="$(sed -n 's/.*warmup_done elapsed_s=\([0-9.]*\).*/\1/p' "$1")" -v g="$GUARD" 'BEGIN{print w+g}'; }

# core per-process group -> writes "$OUT/$label.sum" (event<TAB>value)
run_core(){ local label=$1 ev=$2 prog="$OUT/$1.prog" csv="$OUT/$1.csv"
  sudo perf stat -I 1000 -x , -e "$ev" -o "$csv" -- bash -c "$RUNBASE" > "$prog" 2>/dev/null
  local cut; cut=$(cut_of "$prog")
  awk -F, -v CUT="$cut" '$1+0>CUT && $4!="" {s[$4]+=$2} END{for(e in s) printf "%s\t%.0f\n",e,s[e]}' "$csv" > "$OUT/$label.sum"
  echo "[$label] accrued t>${cut}s"
}

# uncore IMC (system-wide) -> writes "$OUT/imc.sum": gbps + window
run_imc(){ local prog="$OUT/imc.prog" csv="$OUT/imc.csv" imc=""
  for i in 0 1 2 3 4 5; do imc="$imc,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; imc=${imc#,}
  sudo perf stat -a -I 1000 -x , -e "$imc" -o "$csv" -- bash -c "$RUNBASE" > "$prog" 2>/dev/null
  local cut; cut=$(cut_of "$prog")
  awk -F, -v CUT="$cut" '$1+0>CUT && $4 ~ /cas_count/ {v=$2; b=($3=="MiB")?v*1048576:v*64; tot+=b; if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0}
       END{printf "gbps\t%.1f\nwindow_s\t%.1f\n", tot/((hi-lo)*1e9), hi-lo}' "$csv" > "$OUT/imc.sum"
  echo "[imc] accrued t>${cut}s"
}

run_core stalls "cycles,instructions,cycle_activity.stalls_mem_any,cycle_activity.stalls_l1d_miss,cycle_activity.stalls_l2_miss,cycle_activity.stalls_l3_miss"
run_core mlp    "cycles,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,offcore_requests_outstanding.l3_miss_demand_data_rd,offcore_requests_outstanding.cycles_with_l3_miss_demand_data_rd"
run_core loads  "mem_load_retired.l1_hit,mem_load_retired.l1_miss,mem_load_retired.l2_miss,mem_load_retired.l3_miss"
run_imc

g(){ awk -v k="$2" '$1==k{print $2}' "$1"; }   # get value KEY from sum file
S=$OUT/stalls.sum; M=$OUT/mlp.sum; L=$OUT/loads.sum; I=$OUT/imc.sum
echo
echo "================= SUMMARY: ef=$EFV =================" | tee "$OUT/SUMMARY.txt"
{
# recall + QPS (steady: last few measured rows)
grep -E "^$EFV[[:space:]]" "$OUT/stalls.prog" | tail -3 | awk '{printf "recall@10=%s  QPS=%s  (ef=%s, t=%ss)\n",$4,$3,$1,$2}'
awk -v c="$(g $S cycles)" -v ins="$(g $S instructions)" \
    -v ma="$(g $S cycle_activity.stalls_mem_any)" -v l1="$(g $S cycle_activity.stalls_l1d_miss)" \
    -v l2="$(g $S cycle_activity.stalls_l2_miss)" -v l3="$(g $S cycle_activity.stalls_l3_miss)" \
    'BEGIN{printf "IPC=%.3f\n", ins/c;
      printf "memory_stall_fraction (stalls_mem_any/cyc)=%.1f%%\n", 100*ma/c;
      printf "  DRAM(L3miss)=%.1f%%  L3=%.1f%%  L2=%.1f%%  L1lat/other=%.1f%%\n",
        100*l3/c, 100*(l2-l3)/c, 100*(l1-l2)/c, 100*(ma-l1)/c }'
awk -v c="$(g $M cycles)" -v pc="$(g $M l1d_pend_miss.pending_cycles)" -v p="$(g $M l1d_pend_miss.pending)" \
    -v od="$(g $M offcore_requests_outstanding.l3_miss_demand_data_rd)" -v oc="$(g $M offcore_requests_outstanding.cycles_with_l3_miss_demand_data_rd)" \
    'BEGIN{printf "memory_stall_fraction (pending_cycles/cyc)=%.1f%%\n", 100*pc/c;
      printf "MLP (LFB pending/pending_cycles)=%.2f   MLP_DRAM(demand)=%.2f\n", p/pc, od/oc }'
awk -v h="$(g $L mem_load_retired.l1_hit)" -v m="$(g $L mem_load_retired.l1_miss)" \
    -v l3="$(g $L mem_load_retired.l3_miss)" \
    'BEGIN{tot=h+m; printf "loads: L1-hit=%.1f%%  DRAM(l3miss)=%.2f%% of loads\n", 100*h/tot, 100*l3/tot }'
awk -v bw="$(g $I gbps)" -v peak="$PEAK" -v w="$(g $I window_s)" \
    'BEGIN{printf "DRAM bandwidth=%.1f GB/s  (window %.0fs)  utilization=%.1f%% of %.1f GB/s peak\n", bw, w, 100*bw/peak, peak }'
} | tee -a "$OUT/SUMMARY.txt"
echo "raw: $OUT"
