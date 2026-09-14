#!/usr/bin/env bash
# Speculate-then-validate on SIFT100M with tiered placement and a throttled far node
# (tools/spec_search.cpp). Paths come from setup_100m.sh (via $ROOT/.spec_paths.env).
#
#   ./run_specval_100m.sh                    # all three sections, full ratio sweep
#   ONLY=validate ./run_specval_100m.sh      # one section: validate|baseline|spec
#   RATIOS="8" ./run_specval_100m.sh         # a single far-memory latency point
#   NQ=50000 ./run_specval_100m.sh           # more queries for the timed sections
#
# Sections
#   validate  bit-exactness of the pipeline at 100M scale. For every depth the returned top-K,
#             the expansion order and the exact-read count must match the exact search. Runs
#             once, unthrottled, on the default allocator: correctness depends on neither
#             placement nor memory latency, so sweeping it would only burn wall time.
#   baseline  exact search, no speculation. Once with everything on the near node, then once
#             per uncore ratio with vectors on the far node. The near-to-far gap is the latency
#             the pipeline exists to hide, and it is what the spec runs have to beat.
#   spec      the pipeline on far vectors, per ratio, without staging helpers and then with.
#             Without helpers isolates what speculation COSTS (PQ scoring plus discarded work);
#             with helpers adds what it BUYS (far->near copies issued k steps early). A win
#             cannot be attributed to either without all three points at the same ratio.
#
# Placement: dual-socket Xeon Gold 6530 with sub-NUMA clustering, so four NUMA nodes over two
# packages -- nodes 0,1 are the SNC halves of package 0 and nodes 2,3 of package 1. Near node 0
# (cpus 0-15,64-79; 128 GB) holds the graph, the PQ codes and the staging buffer; far node 3
# (cpus 48-63,112-127) holds the vectors. The index is ~60 GB, so each half fits its node. The
# process is pinned to node 0's cpus, so only vector reads cross the interconnect.
#
# NEAR and FAR must sit on different packages. Node 1 looks like a far node by number but shares
# package 0 with node 0: throttling it would slow near memory too, and it is only one SNC hop
# away rather than a socket hop. The package check below refuses that configuration outright.
#
# Throttling: uncore (mesh/IMC) frequency is package-scoped, so the far package is slowed as a
# whole -- node 2 rides along, unused. Preferred path is the intel_uncore_frequency driver, which
# owns this knob on parts that ship it and will otherwise fight direct MSR writes; the MSR 0x620
# path (max=[6:0], min=[14:8], written as (r<<8)|r) is the fallback for parts without it. Ratios
# are 100 MHz units, walking remote memory from full speed toward CXL-like. Either way the
# original setting is saved, verified by read-back, and restored on exit including on abort.
# Needs passwordless sudo, plus msr-tools when falling back to the MSR path.
#
# Cost note: every invocation retrains the PQ codebook and re-encodes 100M nodes (~1-2 min).
# The sweep pays that per run and does not cache it; the codebook is seeded, so runs stay
# comparable regardless.
set -euo pipefail

ROOT=${ROOT:-$HOME/vishal}
PATHS=${PATHS:-$ROOT/.spec_paths.env}
# shellcheck disable=SC1090
[ -f "$PATHS" ] && . "$PATHS"        # only fills IDX/Q/GT/REPO when they are unset

IDX=${IDX:-}
Q=${Q:-}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
BIN=${BIN:-$HOME/spec_search_bench}
SRC=$REPO/tools/spec_search.cpp

T=${T:-16}                       # search threads
EF=${EF:-200}
K=${K:-100}
NQ=${NQ:-20000}                  # queries for the timed sections
NQ_VALIDATE=${NQ_VALIDATE:-2000} # queries for the correctness section
DEPTHS=${DEPTHS:-1,2,4,8}
PQ_M=${PQ_M:-16}

NEAR=${NEAR:-0}                  # graph, PQ codes, staging buffer, and all cpus
FAR=${FAR:-3}                    # vectors
HELPER_CPUS=${HELPER_CPUS:-76,77,78,79}   # node-0 cpus for the staging helpers
BUF_SLOTS=${BUF_SLOTS:-4194304}           # staging capacity in vectors (~2 GB at 512 B)
RATIOS=${RATIOS:-24 16 8 4}               # far-node uncore ratios, 100 MHz units

ONLY=${ONLY:-all}
OUT=${OUT:-$ROOT/specval_100m_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

for v in IDX Q; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -s "$p" ] || { echo "missing $v (${p:-unset}) -- run ./setup_100m.sh, or set $v="; exit 1; }
done
[ -d "$REPO/include/flatnav" ] || { echo "no flatnav checkout at $REPO (set REPO=)"; exit 1; }

# --- pick a cpu in the far package, and refuse to run if throttling would hit the near one ---
# MSR 0x620 is package-scoped, so -p only has to name any cpu on the target socket. With
# sub-NUMA clustering the near and far NUMA nodes can share a package, and throttling the far
# one would then slow local memory too, quietly invalidating every comparison below.
first_cpu_of_node() { local l; l=$(< "/sys/devices/system/node/node$1/cpulist"); echo "${l%%[-,]*}"; }
pkg_of_cpu() { < "/sys/devices/system/cpu/cpu$1/topology/physical_package_id"; }
for n in "$NEAR" "$FAR"; do
  [ -d "/sys/devices/system/node/node$n" ] || {
    echo "no NUMA node $n on this host. Available: $(ls -d /sys/devices/system/node/node[0-9]* | sed 's#.*/node##' | tr '\n' ' ')"
    echo "Set NEAR= and FAR= for this topology (FAR defaults to 3, which assumes a quad-socket node)."
    exit 1; }
done
NEAR_CPU=$(first_cpu_of_node "$NEAR")
REMOTE_CPU=${REMOTE_CPU:-$(first_cpu_of_node "$FAR")}
NEAR_PKG=$(pkg_of_cpu "$NEAR_CPU")
FAR_PKG=$(pkg_of_cpu "$REMOTE_CPU")
[ "$NEAR_PKG" != "$FAR_PKG" ] || {
  echo "node $NEAR and node $FAR are both on package $FAR_PKG (sub-NUMA clustering?)."
  echo "Throttling the far node would slow near memory too. Set FAR= to a node on another package."
  exit 1; }

# --- save the far package's uncore frequency, restore on any exit --------------------------
# Both paths pin min = max so the far uncore cannot ramp back up mid-run, and both verify by
# read-back: a silently clamped write would file a run under a latency it never ran at, which is
# worse than failing because it still looks like data.
UNCORE_SYS=/sys/devices/system/cpu/intel_uncore_frequency
DOMAIN=$UNCORE_SYS/$(printf 'package_%02d_die_00' "$FAR_PKG")

if [ -d "$DOMAIN" ]; then
  THROTTLE=driver
  ORIG_MAX_KHZ=$(< "$DOMAIN/max_freq_khz")
  ORIG_MIN_KHZ=$(< "$DOMAIN/min_freq_khz")
  FLOOR_KHZ=$(< "$DOMAIN/initial_min_freq_khz")
  ORIG_MAX=$(( ORIG_MAX_KHZ / 100000 ))
  wr() { echo "$2" | sudo tee "$DOMAIN/$1" >/dev/null; }
  # min first to the floor so it can never exceed the max being written, then max, then min.
  restore() { wr min_freq_khz "$FLOOR_KHZ"; wr max_freq_khz "$ORIG_MAX_KHZ"; wr min_freq_khz "$ORIG_MIN_KHZ"
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
  MSR=0x620
  sudo modprobe msr 2>/dev/null || true
  command -v rdmsr >/dev/null && command -v wrmsr >/dev/null || { echo "need msr-tools (rdmsr/wrmsr)"; exit 1; }
  ORIG=$(sudo rdmsr -p "$REMOTE_CPU" "$MSR")
  ORIG_MAX=$(( 0x$ORIG & 0x7f ))
  restore() { sudo wrmsr -p "$REMOTE_CPU" "$MSR" "0x$ORIG"; echo "[uncore] restored cpu$REMOTE_CPU MSR$MSR=0x$ORIG"; }
  set_ratio() {
    local r=$1 now got_max got_min
    sudo wrmsr -p "$REMOTE_CPU" "$MSR" $(( (r << 8) | r ))
    now=$(sudo rdmsr -p "$REMOTE_CPU" "$MSR")
    got_max=$(( 0x$now & 0x7f )); got_min=$(( (0x$now >> 8) & 0x7f ))
    echo "[uncore] ratio=$r ($((r / 10)).$((r % 10)) GHz) MSR=0x$now max=$got_max min=$got_min"
    [ "$got_max" = "$r" ] && [ "$got_min" = "$r" ] || \
      echo "[uncore] WARNING: asked for $r, hardware reports max=$got_max min=$got_min -- this point is NOT the ratio its filename claims"
  }
fi
trap restore EXIT

echo "[uncore] path=$THROTTLE domain=package $FAR_PKG | unthrottled max ratio = $ORIG_MAX ($((ORIG_MAX / 10)).$((ORIG_MAX % 10)) GHz)"
for r in $RATIOS; do
  [ "$r" -le "$ORIG_MAX" ] || echo "[uncore] WARNING: requested ratio $r exceeds this part's max $ORIG_MAX"
done

# Unlike the other PQ tools this one needs libnuma: it places the two halves of the index on
# different nodes and allocates the staging buffer on the near one.
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -DFLATNAV_USE_NUMA spec_search.cpp"
  g++ -std=c++17 -O3 -march=native -DFLATNAV_USE_NUMA \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread -lnuma \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

PIN=(numactl --cpunodebind="$NEAR")
# Tolerant on purpose: one bad run should not discard the rest of a long sweep.
run() { local tag=$1; shift; echo "[run] $tag"; "$@" > "$OUT/$tag.log" 2>&1 || echo "  FAILED (see $OUT/$tag.log)"; }
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

echo "[cfg] idx=$IDX"
echo "[cfg] q=$Q nq=$NQ (validate $NQ_VALIDATE) ef=$EF K=$K threads=$T depths=$DEPTHS"
echo "[cfg] near=$NEAR (pkg $NEAR_PKG) far=$FAR (pkg $FAR_PKG, msr cpu $REMOTE_CPU) helpers=$HELPER_CPUS"
echo "[cfg] uncore ratios=$RATIOS | orig MSR$MSR=0x$ORIG -> $OUT"
numactl -H | grep -E "^node ($NEAR|$FAR) (cpus|size|free)" || true

# --- 1. validate: the pipeline returns exactly what the exact search returns ----------------
if want validate; then
  run validate env NQ="$NQ_VALIDATE" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" DIAG=1 \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
  # The oracle pass scores speculation with exact distances: ranking errors and the reachability
  # floor must both go to zero, which separates the pipeline machinery from PQ's own error.
  run validate_oracle env NQ="$NQ_VALIDATE" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" DIAG=1 ORACLE=1 \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- 2. baseline, near half: unaffected by the far node's uncore, so measured once ----------
# DEPTHS= runs the exact pass and skips the grid entirely.
if want baseline; then
  run baseline_near env NQ="$NQ" DEPTHS="" PQ_M="$PQ_M" \
      VEC_NODE="$NEAR" GRAPH_NODE="$NEAR" \
      "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- 3. per uncore ratio: far baseline, then the pipeline without and with staging ----------
# All three run back to back at a given ratio so they share machine state and compare cleanly.
if want baseline || want spec; then
  for r in $RATIOS; do
    echo "===== FAR UNCORE RATIO=$r ====="
    set_ratio "$r"
    if want baseline; then
      run "baseline_far_r$r" env NQ="$NQ" DEPTHS="" PQ_M="$PQ_M" \
          VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
          "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
    fi
    if want spec; then
      run "spec_nostage_r$r" env NQ="$NQ" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" \
          VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
          "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
      run "spec_stage_r$r" env NQ="$NQ" DEPTHS="$DEPTHS" WIDTHS=1 PQ_M="$PQ_M" \
          VEC_NODE="$FAR" GRAPH_NODE="$NEAR" \
          PF_HELPER_CPUS="$HELPER_CPUS" PF_BUF_SLOTS="$BUF_SLOTS" PF_LOCAL_NODE="$NEAR" \
          "${PIN[@]}" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K"
    fi
  done
fi

# --- summary -------------------------------------------------------------------------------
echo
echo "=== summary ($OUT) ==="

if [ -s "$OUT/validate.log" ]; then
  echo "-- correctness (must read PASS; oracle must show order% and floor% at 0.00) --"
  for f in validate validate_oracle; do
    [ -s "$OUT/$f.log" ] || continue
    printf '%-16s %s\n' "$f" "$(grep -E '^(PASS|FAIL)' "$OUT/$f.log" | head -1)"
    grep -E "^      FAIL" "$OUT/$f.log" || true
  done
  echo "-- oracle rows (k w checks/q miss% floor% rej% tie% order% wasted/q) --"
  grep -E "^[0-9]+ +1 " "$OUT/validate_oracle.log" 2>/dev/null || true
fi

echo
echo "-- exact baseline: near, then far at each uncore ratio --"
[ -s "$OUT/baseline_near.log" ] && printf '%-18s %s\n' "near" "$(grep '^\[exact\]' "$OUT/baseline_near.log")"
for r in $RATIOS; do
  [ -s "$OUT/baseline_far_r$r.log" ] || continue
  printf '%-18s %s\n' "far r=$r" "$(grep '^\[exact\]' "$OUT/baseline_far_r$r.log")"
done

echo
echo "-- pipeline on far vectors (k w checks/q miss% disc/q reads/q wasted/q stalls/q qps) --"
for r in $RATIOS; do
  for f in "spec_nostage_r$r" "spec_stage_r$r"; do
    [ -s "$OUT/$f.log" ] || continue
    echo "[$f]"
    grep '^\[exact\]' "$OUT/$f.log" || true
    grep -E "^[0-9]+ +1 |^      staged" "$OUT/$f.log" || true
  done
done

echo
echo "logs: $OUT"
