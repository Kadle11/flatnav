#!/usr/bin/env bash
# Validation offload on SIFT100M: the exact-distance half of speculate-then-validate runs on the
# far package's cores, next to the vectors, while speculation stays near at full clock
# (tools/spec_search.cpp, VO_CPUS=). Paths come from setup_100m.sh (via $ROOT/.spec_paths.env).
#
#   ./run_offload_100m.sh                    # all sections, full ratio sweep
#   ONLY=offload ./run_offload_100m.sh       # one section: baseline|offload
#   RATIOS="8" ./run_offload_100m.sh         # a single far-memory latency point
#   CORE_MHZ=1500 ./run_offload_100m.sh      # a different far-core clock
#
# The premise. Today a step drags ~20 neighbour vectors (~10 KB) across the interconnect because
# the cores that need them sit on the near node. Offloaded, the step sends ~20 ids and receives
# ~20 floats -- ~160 B -- because the distances are computed where the vectors already are. The
# question this run answers is whether that ~64x cut in cross-socket bytes pays for the far cores
# being slow: near-memory compute is WEAK compute, and validation is the critical path. A negative
# result is a real outcome, not a misconfiguration.
#
# So the far package is throttled twice, and the two knobs are independent:
#   uncore (mesh/IMC) frequency  -- how far away far memory FEELS, swept over RATIOS as in
#                                   run_specval_100m.sh, same MSR/driver handling.
#   core frequency               -- how weak the near-memory compute IS, pinned to CORE_MHZ
#                                   (default 800) via cpufreq, MSR 0x199 as fallback.
# Both are restored on exit, including on abort. Near cores are never touched: the speculative
# lane must run at full clock or the comparison means nothing.
#
# Sections
#   baseline  exact search, no speculation, no offload. Near-only once, then far vectors per
#             ratio with the search on near cores -- today's placement, the number to beat.
#   offload   the pipeline with VO_CPUS set to the far node's cpus, per ratio. Validation runs
#             on throttled far cores; speculation, graph and PQ codes stay near and unthrottled.
#
# Lane count must be >= search threads or the short-handed threads validate inline and the run
# silently measures something else; the tool warns, and T defaults to the far node's cpu count.
#
# Recall is reported for every row (GT from .spec_paths.env). It must not move: the validated lane
# is bit-exact whatever the offload does, so a recall change is a bug, not a tuning result.
#
# Cost note: every invocation retrains the PQ codebook and re-encodes 100M nodes (~1-2 min).
set -euo pipefail

ROOT=${ROOT:-$HOME/vishal}
PATHS=${PATHS:-$ROOT/.spec_paths.env}
# shellcheck disable=SC1090
[ -f "$PATHS" ] && . "$PATHS"        # only fills IDX/Q/GT/REPO when they are unset

IDX=${IDX:-}
Q=${Q:-}
GT=${GT:-}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
BIN=${BIN:-$HOME/spec_search_bench}
SRC=$REPO/tools/spec_search.cpp

EF=${EF:-200}
K=${K:-100}
NQ=${NQ:-20000}
DEPTHS=${DEPTHS:-1,2,4,8}
PQ_M=${PQ_M:-16}

NEAR=${NEAR:-0}                  # graph, PQ codes, and the speculative lane
FAR=${FAR:-3}                    # vectors, and the validation workers
CORE_MHZ=${CORE_MHZ:-800}        # far-core clock: the near-memory compute budget
RATIOS=${RATIOS:-24 16 8 4}      # far-node uncore ratios, 100 MHz units

ONLY=${ONLY:-all}
OUT=${OUT:-$ROOT/offload_100m_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

for v in IDX Q; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -s "$p" ] || { echo "missing $v (${p:-unset}) -- run ./setup_100m.sh, or set $v="; exit 1; }
done
[ -d "$REPO/include/flatnav" ] || { echo "no flatnav checkout at $REPO (set REPO=)"; exit 1; }
[ -n "$GT" ] && [ -s "$GT" ] || echo "[warn] no GT (${GT:-unset}) -- recall will not be reported"

# --- topology: far cpus, and the package check the uncore knob demands --------------------
expand_cpulist() {  # "48-63,112-127" -> "48 49 ... 127"
  local out="" part lo hi
  IFS=, read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ $part == *-* ]]; then lo=${part%-*}; hi=${part#*-}
      for ((c = lo; c <= hi; c++)); do out+="$c "; done
    else out+="$part "; fi
  done
  echo "$out"
}
pkg_of_cpu() { local f="/sys/devices/system/cpu/cpu$1/topology/physical_package_id"; [ -r "$f" ] && cat "$f"; }

for n in "$NEAR" "$FAR"; do
  [ -d "/sys/devices/system/node/node$n" ] || {
    echo "no NUMA node $n on this host. Available: $(ls -d /sys/devices/system/node/node[0-9]* | sed 's#.*/node##' | tr '\n' ' ')"
    exit 1; }
done
FAR_CPUS=$(expand_cpulist "$(cat /sys/devices/system/node/node$FAR/cpulist)")
NEAR_CPU=$(expand_cpulist "$(cat /sys/devices/system/node/node$NEAR/cpulist)"); NEAR_CPU=${NEAR_CPU%% *}
REMOTE_CPU=${REMOTE_CPU:-$(echo "$FAR_CPUS" | awk '{print $1}')}
NEAR_PKG=$(pkg_of_cpu "$NEAR_CPU")
FAR_PKG=$(pkg_of_cpu "$REMOTE_CPU")
[ -n "$NEAR_PKG" ] && [ -n "$FAR_PKG" ] || { echo "cannot read physical_package_id"; exit 1; }
[ "$NEAR_PKG" != "$FAR_PKG" ] || {
  echo "node $NEAR (cpu $NEAR_CPU) and node $FAR (cpu $REMOTE_CPU) are both on package $FAR_PKG."
  echo "Both knobs below are package-scoped: throttling far would slow the near lane too."
  exit 1; }

# One physical core per lane: hyperthread siblings share the FMA pipe, so pairing two lanes onto
# one core would measure SMT contention rather than the near-memory compute budget.
# thread_siblings_list is written either as a range ("48-112") or a list ("48,112") depending on
# the kernel and the topology, so expand it and take the lowest id rather than parsing one form.
VO_CPUS=""
for c in $FAR_CPUS; do
  sib=$(expand_cpulist "$(cat "/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list")" \
        | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
  [ "$sib" = "$c" ] && VO_CPUS+="${VO_CPUS:+,}$c"
done
NLANES=$(tr ',' '\n' <<< "$VO_CPUS" | wc -l)
T=${T:-$NLANES}
[ "$T" -le "$NLANES" ] || { echo "T=$T exceeds $NLANES lanes; the extra threads would validate inline"; exit 1; }

# --- far-core frequency: save, pin to CORE_MHZ, restore on exit ---------------------------
# cpufreq is per-cpu and is the knob the kernel's governor also drives, so min is pinned to max:
# left free, the governor would ramp these cores back up mid-run and the "800 MHz" label would be
# a fiction. The MSR 0x199 fallback covers parts without a cpufreq driver bound.
CORE_KHZ=$((CORE_MHZ * 1000))
CPUFREQ_DIR="/sys/devices/system/cpu/cpu$REMOTE_CPU/cpufreq"
declare -A ORIG_CMAX ORIG_CMIN
if [ -d "$CPUFREQ_DIR" ]; then
  CORE_PATH=driver
  for c in $FAR_CPUS; do
    ORIG_CMAX[$c]=$(< "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq")
    ORIG_CMIN[$c]=$(< "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq")
  done
  restore_cores() {
    for c in $FAR_CPUS; do
      echo "${ORIG_CMAX[$c]}" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq" >/dev/null
      echo "${ORIG_CMIN[$c]}" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq" >/dev/null
    done
    echo "[core] restored node $FAR cpus to their original scaling range"
  }
  set_cores() {
    local got
    for c in $FAR_CPUS; do
      # min down first: it can never be left above the max being written.
      echo "${ORIG_CMIN[$c]}" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq" >/dev/null
      echo "$CORE_KHZ" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq" >/dev/null
      echo "$CORE_KHZ" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq" >/dev/null
    done
    got=$(< "/sys/devices/system/cpu/cpu$REMOTE_CPU/cpufreq/scaling_max_freq")
    echo "[core] node $FAR cpus -> ${CORE_MHZ} MHz (cpu$REMOTE_CPU reports max=$got kHz)"
    [ "$got" = "$CORE_KHZ" ] || \
      echo "[core] WARNING: asked for $CORE_KHZ kHz, hardware reports $got -- far cores are NOT at ${CORE_MHZ} MHz"
  }
else
  CORE_PATH=msr
  sudo modprobe msr 2>/dev/null || true
  command -v rdmsr >/dev/null && command -v wrmsr >/dev/null || { echo "no cpufreq driver and no msr-tools"; exit 1; }
  for c in $FAR_CPUS; do ORIG_CMAX[$c]=$(sudo rdmsr -p "$c" 0x199); done
  restore_cores() {
    for c in $FAR_CPUS; do sudo wrmsr -p "$c" 0x199 "0x${ORIG_CMAX[$c]}"; done
    echo "[core] restored node $FAR cpus (MSR 0x199)"
  }
  set_cores() {
    local ratio=$((CORE_MHZ / 100)) now
    for c in $FAR_CPUS; do sudo wrmsr -p "$c" 0x199 $((ratio << 8)); done
    now=$(sudo rdmsr -p "$REMOTE_CPU" 0x199)
    echo "[core] node $FAR cpus -> ${CORE_MHZ} MHz (ratio $ratio, cpu$REMOTE_CPU MSR 0x199=0x$now)"
    [ $(( 0x$now >> 8 & 0xff )) = "$ratio" ] || \
      echo "[core] WARNING: asked for ratio $ratio, hardware reports 0x$now -- far cores are NOT at ${CORE_MHZ} MHz"
  }
fi

# --- far-package uncore frequency: identical handling to run_specval_100m.sh ---------------
UNCORE_SYS=/sys/devices/system/cpu/intel_uncore_frequency
DOMAIN=$UNCORE_SYS/$(printf 'package_%02d_die_00' "$FAR_PKG")
if [ -d "$DOMAIN" ]; then
  THROTTLE=driver
  ORIG_MAX_KHZ=$(< "$DOMAIN/max_freq_khz"); ORIG_MIN_KHZ=$(< "$DOMAIN/min_freq_khz")
  FLOOR_KHZ=$( [ -r "$DOMAIN/initial_min_freq_khz" ] && cat "$DOMAIN/initial_min_freq_khz" || echo "$ORIG_MIN_KHZ" )
  ORIG_MAX=$(( ORIG_MAX_KHZ / 100000 ))
  wr() { echo "$2" | sudo tee "$DOMAIN/$1" >/dev/null; }
  restore_uncore() { wr min_freq_khz "$FLOOR_KHZ"; wr max_freq_khz "$ORIG_MAX_KHZ"; wr min_freq_khz "$ORIG_MIN_KHZ"
                     echo "[uncore] restored $DOMAIN to ${ORIG_MIN_KHZ}-${ORIG_MAX_KHZ} kHz"; }
  set_ratio() {
    local r=$1 khz=$(( $1 * 100000 )) got_max got_min
    wr min_freq_khz "$FLOOR_KHZ"; wr max_freq_khz "$khz"; wr min_freq_khz "$khz"
    got_max=$(< "$DOMAIN/max_freq_khz"); got_min=$(< "$DOMAIN/min_freq_khz")
    echo "[uncore] ratio=$r ($((r / 10)).$((r % 10)) GHz) -> min=$got_min max=$got_max kHz"
    [ "$got_max" = "$khz" ] && [ "$got_min" = "$khz" ] || \
      echo "[uncore] WARNING: asked for $khz kHz, hardware reports min=$got_min max=$got_max -- this point is NOT the ratio its filename claims"
  }
else
  THROTTLE=msr
  sudo modprobe msr 2>/dev/null || true
  command -v rdmsr >/dev/null && command -v wrmsr >/dev/null || { echo "need msr-tools (rdmsr/wrmsr)"; exit 1; }
  ORIG=$(sudo rdmsr -p "$REMOTE_CPU" 0x620)
  ORIG_MAX=$(( 0x$ORIG & 0x7f ))
  restore_uncore() { sudo wrmsr -p "$REMOTE_CPU" 0x620 "0x$ORIG"; echo "[uncore] restored cpu$REMOTE_CPU MSR0x620=0x$ORIG"; }
  set_ratio() {
    local r=$1 now got_max got_min
    sudo wrmsr -p "$REMOTE_CPU" 0x620 $(( (r << 8) | r ))
    now=$(sudo rdmsr -p "$REMOTE_CPU" 0x620)
    got_max=$(( 0x$now & 0x7f )); got_min=$(( (0x$now >> 8) & 0x7f ))
    echo "[uncore] ratio=$r ($((r / 10)).$((r % 10)) GHz) MSR=0x$now max=$got_max min=$got_min"
    [ "$got_max" = "$r" ] && [ "$got_min" = "$r" ] || \
      echo "[uncore] WARNING: asked for $r, hardware reports max=$got_max min=$got_min -- this point is NOT the ratio its filename claims"
  }
fi

restore() { restore_cores; restore_uncore; }
trap restore EXIT

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ] \
   || [ "$REPO/include/flatnav/util/ValidationOffload.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -DFLATNAV_USE_NUMA spec_search.cpp"
  g++ -std=c++17 -O3 -march=native -DFLATNAV_USE_NUMA \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread -lnuma \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

PIN=(numactl --cpunodebind="$NEAR")
run() { local tag=$1; shift; echo "[run] $tag"; "$@" > "$OUT/$tag.log" 2>&1 || echo "  FAILED (see $OUT/$tag.log)"; }
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

echo "[cfg] idx=$IDX"
echo "[cfg] q=$Q gt=${GT:-none} nq=$NQ ef=$EF K=$K threads=$T depths=$DEPTHS"
echo "[cfg] near=$NEAR (pkg $NEAR_PKG, full clock) far=$FAR (pkg $FAR_PKG) core=${CORE_MHZ}MHz via $CORE_PATH"
echo "[cfg] lanes=$NLANES cpus=$VO_CPUS | uncore ratios=$RATIOS via $THROTTLE -> $OUT"

set_cores

# --- 1. baseline, near half: unaffected by either far knob, so measured once ---------------
if want baseline; then
  run baseline_near env NQ="$NQ" DEPTHS="" PQ_M="$PQ_M" GT="$GT" \
      VEC_NODE="$NEAR" GRAPH_NODE="$NEAR" \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- 2. per uncore ratio: far baseline (search near), then the pipeline with offload -------
for r in $RATIOS; do
  echo "===== FAR UNCORE RATIO=$r ====="
  set_ratio "$r"
  if want baseline; then
    run "baseline_far_r$r" env NQ="$NQ" DEPTHS="" PQ_M="$PQ_M" GT="$GT" \
        VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
        "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
    # Speculation without offload, same placement: isolates what the pipeline costs before any
    # work moves, so an offload win cannot be confused with a speculation win.
    run "spec_nostage_r$r" env NQ="$NQ" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" GT="$GT" \
        VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
        "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
  fi
  if want offload; then
    run "offload_r$r" env NQ="$NQ" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" GT="$GT" \
        VEC_NODE="$FAR" GRAPH_NODE="$NEAR" VO_CPUS="$VO_CPUS" \
        "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
  fi
done

# --- summary -------------------------------------------------------------------------------
echo
echo "=== summary ($OUT) ==="
echo "-- correctness: every row must read PASS, and recall must be identical across all of them --"
grep -l . "$OUT"/*.log >/dev/null 2>&1 && for f in "$OUT"/*.log; do
  printf '%-22s %s\n' "$(basename "$f" .log)" "$(grep -E '^(PASS|FAIL)' "$f" | head -1)"
done

echo
echo "-- exact baseline: near, then far at each uncore ratio --"
[ -s "$OUT/baseline_near.log" ] && printf '%-18s %s\n' "near" "$(grep '^\[exact\]' "$OUT/baseline_near.log")"
for r in $RATIOS; do
  [ -s "$OUT/baseline_far_r$r.log" ] || continue
  printf '%-18s %s\n' "far r=$r" "$(grep '^\[exact\]' "$OUT/baseline_far_r$r.log")"
done

echo
echo "-- pipeline (k w checks/q miss% disc/q reads/q wasted/q stalls/q qps recall) --"
for r in $RATIOS; do
  for f in "spec_nostage_r$r" "offload_r$r"; do
    [ -s "$OUT/$f.log" ] || continue
    echo "[$f]"
    grep -E '^\[exact\]|^\[offload\]' "$OUT/$f.log" || true
    grep -E "^[0-9]+ +1 " "$OUT/$f.log" || true
  done
done

echo
echo "logs: $OUT"
