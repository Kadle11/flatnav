#!/usr/bin/env bash
# Bandwidth-swept roofline probe: does the distance kernel's achieved throughput actually track
# the memory subsystem's clock? scripts/roofline_probe.sh (laptop) inferred memory-bound from a
# working-set-size sweep -- throughput plateaus once N exceeds LLC. That's an inference from cache
# geometry, not a direct test. This script tests it directly: throttle one package's uncore (mesh
# + IMC clock, which sets local DRAM bandwidth) across RATIOS and rerun tools/mlp_probe.cpp with
# BOTH data and threads pinned to that same node (no cross-socket hop -- this isn't the far-memory
# experiment the other remote-scripts run, just a local kernel under a dialed-down memory clock).
#
# If GB/s tracks the ratio (roughly linear), the kernel is bandwidth-bound, directly confirmed --
# no working-set inference needed. If GB/s stays flat as the ratio drops, the earlier plateau read
# was wrong and something else (core-side stalls, fixed per-access latency) is the real limit.
#
#   ./run_roofline_bw_sweep.sh                 # sweep node 1's uncore, default ratios
#   NODE=0 RATIOS="24 16 8 4" ./run_roofline_bw_sweep.sh
#   K=12 T=16 ./run_roofline_bw_sweep.sh        # K,T from the earlier MLP-saturation sweep (mlp_probe.sh)
#
# Uncore throttling: identical handling to run_specval_100m.sh / run_offload_100m.sh (driver path
# via intel_uncore_frequency, MSR 0x620 fallback). Package-scoped, saved and restored on exit,
# verified by read-back. Needs passwordless sudo, plus msr-tools when falling back to the MSR path.
set -euo pipefail

ROOT=${ROOT:-$HOME/vishal}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
BIN=${BIN:-$HOME/mlp_probe_bench}
SRC=$REPO/tools/mlp_probe.cpp

NODE=${NODE:-1}                                # node whose uncore is throttled; data AND threads both live here
N=${N:-$(( 8 * 1024 * 1024 * 1024 / 512 ))}    # ~8 GB working set (>> LLC), same default as mlp_probe.sh
DIM=${DIM:-128}
K=${K:-12}                                     # per-core MLP-saturation point from mlp_probe.sh; override if that moved
PFL=${PFL:-1}
T=${T:-16}                                     # 1/phys-core, isolates from SMT (mlp_probe.sh convention)
DUR=${DUR:-5}
RATIOS=${RATIOS:-24 16 8 4}                    # 100 MHz units, same convention as run_specval_100m.sh

OUT=${OUT:-$ROOT/roofline_bw_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

[ -d "$REPO/include/flatnav" ] || { echo "no flatnav checkout at $REPO (set REPO=)"; exit 1; }
[ -d "/sys/devices/system/node/node$NODE" ] || {
  echo "no NUMA node $NODE on this host. Available: $(ls -d /sys/devices/system/node/node[0-9]* | sed 's#.*/node##' | tr '\n' ' ')"
  exit 1; }

first_cpu_of_node() { local l; l=$(cat "/sys/devices/system/node/node$1/cpulist"); echo "${l%%[-,]*}"; }
pkg_of_cpu() { local f="/sys/devices/system/cpu/cpu$1/topology/physical_package_id"; [ -r "$f" ] && cat "$f"; }
CPU=$(first_cpu_of_node "$NODE")
PKG=$(pkg_of_cpu "$CPU")
[ -n "$PKG" ] || { echo "cannot read physical_package_id for node $NODE cpu $CPU"; exit 1; }

# --- uncore throttle: identical handling to run_specval_100m.sh / run_offload_100m.sh ------
UNCORE_SYS=/sys/devices/system/cpu/intel_uncore_frequency
DOMAIN=$UNCORE_SYS/$(printf 'package_%02d_die_00' "$PKG")
if [ -d "$DOMAIN" ]; then
  THROTTLE=driver
  ORIG_MAX_KHZ=$(< "$DOMAIN/max_freq_khz"); ORIG_MIN_KHZ=$(< "$DOMAIN/min_freq_khz")
  FLOOR_KHZ=$( [ -r "$DOMAIN/initial_min_freq_khz" ] && cat "$DOMAIN/initial_min_freq_khz" || echo "$ORIG_MIN_KHZ" )
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
  ORIG=$(sudo rdmsr -p "$CPU" 0x620)
  restore_uncore() { sudo wrmsr -p "$CPU" 0x620 "0x$ORIG"; echo "[uncore] restored cpu$CPU MSR0x620=0x$ORIG"; }
  set_ratio() {
    local r=$1 now got_max got_min
    sudo wrmsr -p "$CPU" 0x620 $(( (r << 8) | r ))
    now=$(sudo rdmsr -p "$CPU" 0x620)
    got_max=$(( 0x$now & 0x7f )); got_min=$(( (0x$now >> 8) & 0x7f ))
    echo "[uncore] ratio=$r ($((r / 10)).$((r % 10)) GHz) MSR=0x$now max=$got_max min=$got_min"
    [ "$got_max" = "$r" ] && [ "$got_min" = "$r" ] || \
      echo "[uncore] WARNING: asked for ratio $r, hardware reports max=$got_max min=$got_min -- this point is NOT the ratio its filename claims"
  }
fi
trap restore_uncore EXIT

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -march=native -DFLATNAV_USE_NUMA mlp_probe.cpp"
  g++ -std=c++17 -O3 -march=native -DFLATNAV_USE_NUMA \
    -I "$REPO/include" "$SRC" -o "$BIN" -lnuma -lpthread
fi

echo "[cfg] node=$NODE (pkg $PKG) N=$N dim=$DIM K=$K PFL=$PFL T=$T dur=${DUR}s ratios=$RATIOS via $THROTTLE -> $OUT"

echo "ratio GHz GBps" > "$OUT/summary.txt"
for r in $RATIOS; do
  set_ratio "$r"
  LOG="$OUT/r$r.log"
  "$BIN" "$N" "$DIM" "$K" "$PFL" "$T" "$NODE" "$NODE" "$DUR" > "$LOG" 2>&1
  cat "$LOG"
  GBPS=$(awk '{for(i=1;i<=NF;i++) if($i=="GB/s") print $(i-1)}' "$LOG")
  printf '%s %s.%s %s\n' "$r" "$((r / 10))" "$((r % 10))" "${GBPS:-?}" | tee -a "$OUT/summary.txt"
done

echo
echo "-- if GB/s scales with ratio (roughly linear), the kernel is directly confirmed bandwidth-bound --"
cat "$OUT/summary.txt"
echo "logs: $OUT"
