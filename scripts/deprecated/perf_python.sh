#!/usr/bin/env bash
# Root-cause why the Python batched search has lower DRAM-bandwidth utilization
# than the C++ driver. Builds _core from CURRENT source (matches the new index
# format), smoke-tests it, then profiles the Python search two ways:
#   (A) default   — no numactl/pin (how the library actually runs)
#   (B) local     — numactl --cpunodebind=0 --membind=0 (forced node-0 local)
# Each: ONE perf -a pass (IMC bandwidth + core counters), load excluded via the
# driver's warmup_done marker. Plus a perf record function profile of (A).
# Compare BW to the C++ ~58 GB/s and see where the gap comes from.
#
# Usage:  bash ~/flatnav/scripts/perf_python.sh        (sudo for perf)
set -u
REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/index/sift100m_flatnav.bin}
Q=${Q:-/data/queries/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/queries/sift100m_200k_extra_query.gtruth.ivecs}
CPUS=$(numactl --hardware | sed -n 's/^node 0 cpus: //p' | tr ' ' ',')
EF=${EF:-80}; THREADS=${THREADS:-32}; REPEAT=${REPEAT:-8}; GUARD=${GUARD:-3}; PEAK=${PEAK:-102.8}
TS=$(date +%Y%m%d_%H%M%S); OUT=$HOME/pyperf_$TS; mkdir -p "$OUT"
DRV=$REPO/scripts/py_search_driver.py

echo "=== python batched-search profile  $TS ===" | tee "$OUT/SUMMARY.txt"
echo "[deps] pybind11 + numpy (user)"; python3 -m pip install --user -q pybind11 numpy 2>&1 | tail -1

echo "[build] _core from current source ..."
EXT=$(python3-config --extension-suffix)
g++ -O3 -march=native -fopenmp -std=c++17 -shared -fPIC \
  $(python3 -m pybind11 --includes) \
  -I "$REPO/include" -I "$REPO/external/cereal/include" -I "$REPO/python-bindings/src/flatnav" \
  "$REPO/python-bindings/src/flatnav/bindings.cpp" -o "$HOME/_core$EXT" 2>&1 | grep -iE "error:" | head
test -f "$HOME/_core$EXT" && echo "built $HOME/_core$EXT" || { echo "BUILD FAILED"; exit 1; }

PYRUN="python3 $DRV $IDX $Q $GT 10 $THREADS $EF $REPEAT"

IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"

profile(){ # $1=label  $2=launch-prefix
  local label=$1 pre=$2 prog="$OUT/$1.prog" csv="$OUT/$1.csv"
  ( cd "$HOME" && sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$csv" -- \
     bash -c "$pre $PYRUN" ) > "$prog" 2>/dev/null
  local cut; cut=$(awk -v w="$(sed -n 's/.*warmup_done elapsed_s=\([0-9.]*\).*/\1/p' "$prog")" -v g="$GUARD" 'BEGIN{print w+g}')
  local rec qps; rec=$(grep -oE 'recall@[0-9]+=[0-9.]+' "$prog" | tail -1)
  qps=$(grep -E "^$EF[[:space:]]" "$prog" | tail -1 | awk '{print $3}')
  awk -F, -v CUT="$cut" -v PEAK="$PEAK" -v LB="$label" -v R="$rec" -v Q="$qps" '
    $1+0>CUT && $4!="" { ev=$4; v=$2;
      if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
      else if(ev=="cycles") cyc+=v; else if(ev=="instructions") ins+=v;
      else if(ev=="l1d_pend_miss.pending") p+=v; else if(ev=="l1d_pend_miss.pending_cycles") pc+=v;
      if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
    END{ w=hi-lo; bw=cas/w/1e9;
      printf "%-9s QPS=%-7s %s  IPC=%.3f  mem_stall=%.1f%%  MLP=%.2f  BW=%.1f GB/s  util=%.0f%%\n",
        LB, Q, R, ins/cyc, 100*pc/cyc, p/pc, bw, 100*bw/PEAK }' "$csv" | tee -a "$OUT/SUMMARY.txt"
}

echo "--- (A) default: no numactl/pin (library default) ---" | tee -a "$OUT/SUMMARY.txt"
profile default ""
echo "--- (B) local: numactl --cpunodebind=0 --membind=0 ---" | tee -a "$OUT/SUMMARY.txt"
profile local "numactl --cpunodebind=0 --membind=0"

echo "--- (C) function profile of (A): perf record ---" | tee -a "$OUT/SUMMARY.txt"
( cd "$HOME" && sudo perf record -F 499 -g -o "$OUT/default.data" -- bash -c "$PYRUN" ) >/dev/null 2>&1
sudo chown "$USER" "$OUT/default.data" 2>/dev/null
perf report -i "$OUT/default.data" --stdio --no-children -g none --percent-limit 1 2>/dev/null \
  | grep -vE '^#|^$' | head -20 | tee -a "$OUT/SUMMARY.txt"
echo "raw: $OUT" | tee -a "$OUT/SUMMARY.txt"
