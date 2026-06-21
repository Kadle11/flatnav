#!/usr/bin/env bash
# Perf profile of FlatNav search on the CloudLab node.
#
#   1) Function-level cycle attribution        (perf record / report)
#   2) Memory-stall attribution + load sources (perf stat: topdown,
#      cycle_activity.stalls_*, mem_load_retired.*)
#
# Setup (per the request):
#   - NON-PREFETCHED build (-DFLATNAV_DISABLE_PREFETCH)
#   - CPU frequency LOCKED + turbo DISABLED (stable cycle attribution)
#   - search threads PINNED to node-0 cores (FLATNAV_PIN_CPUS)
#
# Usage:   bash ~/flatnav/scripts/perf_profile_search.sh
# Uses sudo for perf_event_paranoid / freq / turbo. Prints the log path at end.
set -u

REPO=${REPO:-$HOME/flatnav}
IDX=${IDX:-/data/index/sift100m_flatnav.bin}
Q=${Q:-/data/queries/sift100m_200k_extra_query.fvecs}
GT=${GT:-/data/queries/sift100m_200k_extra_query.gtruth.ivecs}
NODE=${NODE:-0}
K=${K:-10}
EF=${EF:-80}                       # representative ef for attribution
REPEAT=${REPEAT:-5}                # repeat the search so it dominates the load
LOCK_KHZ=${LOCK_KHZ:-2600000}      # Xeon 6142 base = 2.6 GHz (turbo off)
TS=$(date +%Y%m%d_%H%M%S)
LOG=$HOME/perf_search_${TS}.log
DATA=$HOME/perf_search_${TS}.data

# node-NODE cpu list, e.g. "0,2,4,...,62"
CPUS=$(numactl --hardware | sed -n "s/^node ${NODE} cpus: //p" | tr ' ' ',')
NTHREADS=$(echo "$CPUS" | tr ',' '\n' | grep -c .)
# build the repeated ef list, e.g. "80,80,80,80,80"
EFLIST=$(yes "$EF" | head -n "$REPEAT" | paste -sd,)

exec > >(tee "$LOG") 2>&1
echo "=== FlatNav search perf profile  $TS ==="
echo "repo=$REPO  idx=$IDX"
echo "ef=$EF x$REPEAT  K=$K  node=$NODE  pinned_cpus($NTHREADS)=$CPUS"
echo

echo "=== [1/4] stabilize machine (sudo) ==="
sudo sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0 kernel.nmi_watchdog=0
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
sudo cpupower frequency-set -g performance >/dev/null 2>&1
for c in /sys/devices/system/cpu/cpu*/cpufreq; do
  echo "$LOCK_KHZ" | sudo tee "$c/scaling_min_freq" >/dev/null
  echo "$LOCK_KHZ" | sudo tee "$c/scaling_max_freq" >/dev/null
done
echo "no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)  governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "cpu0 cur/min/max kHz = $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)/$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)/$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
echo

echo "=== [2/4] build non-prefetched profiling binary ==="
g++ -std=c++17 -O3 -march=native -fno-omit-frame-pointer -g -fopenmp \
    -DFLATNAV_DISABLE_PREFETCH \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$REPO/tools/hbw_search.cpp" -o "$HOME/hbw_search_prof" || { echo "BUILD FAILED"; exit 1; }
echo "built $HOME/hbw_search_prof (no-prefetch, -g, frame pointers)"
echo

# Workload wrapped so sudo/perf children get the pinning env.
RUN="env FLATNAV_PIN_CPUS=$CPUS numactl --membind=$NODE $HOME/hbw_search_prof $IDX $Q $GT $K $NTHREADS $EFLIST"

echo "=== [3/4] memory-stall attribution (perf stat) ==="
echo "--- top-down (where the pipeline is bound) ---"
sudo perf stat -M TopdownL1 -- bash -c "$RUN" 2>&1 | grep -vE "^\[|recall|^ef|QPS|load\]|pin\]|^[0-9 ]+\." | sed -n '1,40p'
echo
echo "--- stall cycles by memory level ---"
sudo perf stat -e cycles,instructions,cycle_activity.stalls_mem_any,cycle_activity.stalls_l1d_miss,cycle_activity.stalls_l2_miss,cycle_activity.stalls_l3_miss \
  -- bash -c "$RUN" 2>&1 | grep -E "cycles|instructions|stalls|insn per|seconds" | sed -n '1,20p'
echo
echo "--- retired-load source breakdown (L1/L2/L3/DRAM) ---"
sudo perf stat -e mem_load_retired.l1_hit,mem_load_retired.l1_miss,mem_load_retired.l2_hit,mem_load_retired.l2_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss \
  -- bash -c "$RUN" 2>&1 | grep -E "mem_load_retired|seconds" | sed -n '1,20p'
echo

echo "=== [4/4] function cycle profile (perf record/report) ==="
sudo perf record -F 999 --call-graph fp -o "$DATA" -- bash -c "$RUN" >/dev/null 2>&1
sudo chown "$USER" "$DATA" 2>/dev/null
echo "--- top functions (flat, self %) ---"
perf report -i "$DATA" --stdio -n --percent-limit 1 2>/dev/null | grep -vE "^#|^$" | sed -n '1,30p'
echo
echo "--- top call paths (caller view) ---"
perf report -i "$DATA" --stdio -g graph,0.5,caller --percent-limit 3 2>/dev/null | sed -n '1,55p'
echo

echo "=== DONE ==="
echo "log:       $LOG"
echo "perf data: $DATA   (annotate with: perf annotate -i $DATA <symbol>)"
echo "restore turbo/freq: echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo ; sudo cpupower frequency-set -g schedutil"
