#!/usr/bin/env bash
# Search-ONLY perf, with the index-load phase excluded by construction.
#
# Method (no fixed-delay guess): run `perf stat -I 1000` over the whole process,
# then sum only the intervals AFTER the hbw_search-emitted phase marker
# `[phase] warmup_done elapsed_s=...` (same elapsed-seconds frame as perf's
# interval timestamps), plus a small GUARD for the perf-vs-exec start offset.
#
# Usage:  bash ~/flatnav/scripts/perf_search_only.sh     (uses sudo for perf)
set -u
REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/index/sift100m_flatnav.bin}
Q=${Q:-/data/queries/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/queries/sift100m_200k_extra_query.gtruth.ivecs}
CPUS=$(numactl --hardware | sed -n 's/^node 0 cpus: //p' | tr ' ' ',')
EF=${EF:-80,80,80,80,80,80,80,80,80,80,80,80}   # 12x -> long steady search window
GUARD=${GUARD:-2}                                # s past warmup_done (covers exec offset + 1 interval)
TS=$(date +%Y%m%d_%H%M%S); OUT=$HOME/searchonly_$TS; mkdir -p "$OUT"

echo "=== SEARCH-ONLY perf (interval accrual past warmup_done + ${GUARD}s) $TS ==="
echo "build (non-prefetch, inlined, with phase markers)..."
g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_DISABLE_PREFETCH \
  -I "$REPO/include" -I "$REPO/external/cereal/include" \
  "$REPO/tools/hbw_search.cpp" -o "$HOME/hbw_search_ts" || { echo BUILD_FAIL; exit 1; }

run_group(){   # $1=label  $2=comma-separated events
  local label=$1 ev=$2 prog="$OUT/$1.prog" ivl="$OUT/$1.csv"
  sudo perf stat -I 1000 -x , -e "$ev" -o "$ivl" -- \
    bash -c "env FLATNAV_PIN_CPUS=$CPUS numactl --membind=0 $HOME/hbw_search_ts $IDX $Q $GT 10 32 $EF" \
    > "$prog" 2>/dev/null
  local warm load cut
  load=$(sed -n 's/.*load_done elapsed_s=\([0-9.]*\).*/\1/p' "$prog")
  warm=$(sed -n 's/.*warmup_done elapsed_s=\([0-9.]*\).*/\1/p' "$prog")
  cut=$(awk -v w="$warm" -v g="$GUARD" 'BEGIN{printf "%.3f", w+g}')
  echo "## [$label]  load_done=${load}s  warmup_done=${warm}s  -> accrue perf intervals t > ${cut}s"
  # perf -x , interval CSV: $1=time $2=value $3=unit $4=event $6=%enabled
  awk -F, -v CUT="$cut" '
    $1 ~ /^[0-9]+\.[0-9]+$/ { t=$1+0;
       if (t>CUT) { sum[$4]+=$2; cnt++; pe[$4]=$6; if(t>hi)hi=t; if(lo==0||t<lo)lo=t } }
    END{ printf "   window: t=%.1f..%.1f s (%d interval-rows)\n", lo, hi, cnt;
         for (e in sum) printf "   %-58s %18.0f  (enabled~%s%%)\n", e, sum[e], pe[e] }' "$ivl" | sort
  echo
}

run_group stalls "cycles,instructions,cycle_activity.stalls_mem_any,cycle_activity.stalls_l1d_miss,cycle_activity.stalls_l2_miss,cycle_activity.stalls_l3_miss"
run_group mlp    "cycles,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,offcore_requests_outstanding.l3_miss_demand_data_rd,offcore_requests_outstanding.cycles_with_l3_miss_demand_data_rd"
run_group loads  "mem_load_retired.l1_hit,mem_load_retired.l1_miss,mem_load_retired.l2_miss,mem_load_retired.l3_miss"
echo "raw csv + prog logs in: $OUT"
