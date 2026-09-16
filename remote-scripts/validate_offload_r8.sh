#!/usr/bin/env bash
# Forensic validation of the r=8 offload result. run_offload_100m.sh answers "is it faster";
# this answers "is it faster for the reason we think, on the hardware config we think".
#
#   ./validate_offload_r8.sh                 # all four runs
#   ONLY=nulltest ./validate_offload_r8.sh   # one run: baseline_near|baseline_far|offload|nulltest
#   DEPTHS=1,2 ./validate_offload_r8.sh      # more pipeline rows (each adds a measured window)
#
# Everything here is at the single uncore ratio r=8, deliberately. On this box the uncore domain
# floor is 800 MHz == ratio 8, so r=8 is the slowest far memory the hardware can actually be put
# into. r=4 asks for 400 MHz, gets clamped to the floor, and silently duplicates r=8 -- which is
# why the sweep's r8 and r4 columns agree to within 3% in every mode. The ratio is read back and
# asserted below rather than assumed.
#
# Three things the sweep does not do:
#
#   1. config proof   Every knob the result depends on is read back from sysfs and archived NEXT
#                     TO the numbers: uncore floor/ceiling and where the ratio actually landed,
#                     far-core clock, AutoNUMA state, free memory per node, turbo state. The
#                     sweep prints its [uncore]/[core] confirmations to the console, where they
#                     are not captured -- the archived run dirs contain zero evidence of the
#                     config they claim. AutoNUMA especially: left on, the kernel migrates hot
#                     far pages toward the accessing socket mid-run and every "far" number
#                     quietly becomes a near number. It is disabled here and restored on exit.
#
#   2. null test      Offload with the vectors already NEAR. There is then nothing to move the
#                     compute toward, so offload MUST NOT win -- the far lanes now pull vectors
#                     across the interconnect to do the same math, at 800 MHz. If it wins anyway,
#                     the measured advantage was never about locality, and the whole premise
#                     needs rethinking rather than another sweep.
#
#   3. byte counting  The claim is a claim about BYTES (~10 KB of vectors per step becomes ~160 B
#                     of ids and floats), so bytes get measured, not inferred from elapsed time.
#                     Per-socket IMC and UPI counters, gated to each individual search window.
#                     Expected at 4117.9 reads/query x 512 B: ~2.1 MB/query over the interconnect
#                     for baseline_far, ~33 KB/query for offload. If that ~64x does not appear in
#                     the UPI counters, the mechanism is not what the timings suggest.
#
# Windows are gated by timestamping the tool's own output lines as they arrive (it fflushes after
# each row) and working back by each row's reported duration. That beats reconstructing offsets
# from process start, where the pre-load query/GT work is unaccounted slop.
#
# Cost: ~90s of PQ training + encoding per run, then one search pass per configured depth.
set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/hwguard.sh"

ROOT=${ROOT:-$HOME/vishal}
PATHS=${PATHS:-$ROOT/.spec_paths.env}
# shellcheck disable=SC1090
[ -f "$PATHS" ] && . "$PATHS"

IDX=${IDX:-}
Q=${Q:-}
GT=${GT:-}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
BIN=${BIN:-$HOME/spec_search_bench}
SRC=$REPO/tools/spec_search.cpp

EF=${EF:-200}
K=${K:-100}
NQ=${NQ:-200000}
DEPTHS=${DEPTHS:-1}
PQ_M=${PQ_M:-16}

NEAR=${NEAR:-0}
FAR=${FAR:-1}
RATIO=8                          # the floor; not a sweep. See header.
CORE_MHZ=${CORE_MHZ:-800}
GUARD=${GUARD:-1}                # seconds trimmed off each window edge, for perf interval slop
ONLY=${ONLY:-all}
OUT=${OUT:-$ROOT/validate_r8_$(date +%m%d_%H%M)}
mkdir -p "$OUT"
PROOF="$OUT/config_proof.txt"

for v in IDX Q; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -s "$p" ] || { echo "missing $v (${p:-unset}) -- run ./setup_100m.sh, or set $v="; exit 1; }
done
[ -d "$REPO/include/flatnav" ] || { echo "no flatnav checkout at $REPO (set REPO=)"; exit 1; }
[ -n "$GT" ] && [ -s "$GT" ] || echo "[warn] no GT (${GT:-unset}) -- recall will not be reported"
command -v perf >/dev/null || { echo "perf not found"; exit 1; }

say() { echo "$@" | tee -a "$PROOF"; }

# --- topology ------------------------------------------------------------------------------
expand_cpulist() {
  local out="" part lo hi
  IFS=, read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ $part == *-* ]]; then lo=${part%-*}; hi=${part#*-}
      for ((c = lo; c <= hi; c++)); do out+="$c "; done
    else out+="$part "; fi
  done
  echo "$out"
}
for n in "$NEAR" "$FAR"; do
  [ -d "/sys/devices/system/node/node$n" ] || { echo "no NUMA node $n on this host"; exit 1; }
done
FAR_CPUS=$(expand_cpulist "$(cat /sys/devices/system/node/node$FAR/cpulist)")
NEAR_CPU=$(expand_cpulist "$(cat /sys/devices/system/node/node$NEAR/cpulist)"); NEAR_CPU=${NEAR_CPU%% *}
REMOTE_CPU=${REMOTE_CPU:-$(echo "$FAR_CPUS" | awk '{print $1}')}
NEAR_PKG=$(cat "/sys/devices/system/cpu/cpu$NEAR_CPU/topology/physical_package_id")
FAR_PKG=$(cat "/sys/devices/system/cpu/cpu$REMOTE_CPU/topology/physical_package_id")
[ "$NEAR_PKG" != "$FAR_PKG" ] || { echo "nodes $NEAR/$FAR share package $FAR_PKG -- throttling far would slow near too"; exit 1; }

VO_CPUS=""
for c in $FAR_CPUS; do
  sib=$(expand_cpulist "$(cat "/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list")" \
        | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
  [ "$sib" = "$c" ] && VO_CPUS+="${VO_CPUS:+,}$c"
done
NLANES=$(tr ',' '\n' <<< "$VO_CPUS" | wc -l)
T=${T:-$NLANES}
[ "$T" -le "$NLANES" ] || { echo "T=$T exceeds $NLANES lanes; extra threads would validate inline"; exit 1; }

# --- 1. config proof -------------------------------------------------------------------------
{
  echo "=== validate_offload_r8 config proof ==="
  echo "host=$(hostname)  date=$(date -Is)  kernel=$(uname -r)"
  echo "idx=$IDX"
  echo "q=$Q gt=${GT:-none} nq=$NQ ef=$EF K=$K T=$T depths=$DEPTHS"
  echo "near=node$NEAR (pkg $NEAR_PKG)  far=node$FAR (pkg $FAR_PKG)  lanes=$NLANES"
  echo "vo_cpus=$VO_CPUS"
  echo
  echo "--- free memory per node (vectors need ~51 GB on the vector node; a silent fallback"
  echo "--- to the other node would invalidate every placement in this run) ---"
  numactl -H
  echo
  echo "--- turbo / pstate ---"
  for f in /sys/devices/system/cpu/intel_pstate/no_turbo /sys/devices/system/cpu/cpufreq/boost; do
    [ -r "$f" ] && echo "$f = $(cat "$f")"
  done
  echo "scaling_driver=$(cat "/sys/devices/system/cpu/cpu$REMOTE_CPU/cpufreq/scaling_driver" 2>/dev/null || echo none)"
  echo "scaling_governor=$(cat "/sys/devices/system/cpu/cpu$REMOTE_CPU/cpufreq/scaling_governor" 2>/dev/null || echo none)"
  echo
  echo "--- transparent hugepages ---"
  cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unavailable
} > "$PROOF"

# AutoNUMA. It cannot move the index (numa_alloc_onnode binds MPOL_BIND) but it can migrate the
# PQ codes and query arrays, and it can migrate threads across sockets in any multi-node pinning.
say "numa_balancing (before) = $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unavailable)"
hwguard_numa_balancing_off | tee -a "$PROOF"
restore_nb() { hwguard_restore_all; }

# --- far-core clock --------------------------------------------------------------------------
CORE_KHZ=$((CORE_MHZ * 1000))
declare -A ORIG_CMAX ORIG_CMIN
if [ -d "/sys/devices/system/cpu/cpu$REMOTE_CPU/cpufreq" ]; then
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
    echo "[core] restored node $FAR cpus"
  }
  set_cores() {
    for c in $FAR_CPUS; do
      echo "${ORIG_CMIN[$c]}" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq" >/dev/null
      echo "$CORE_KHZ" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq" >/dev/null
      echo "$CORE_KHZ" | sudo tee "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_min_freq" >/dev/null
    done
    local got; got=$(< "/sys/devices/system/cpu/cpu$REMOTE_CPU/cpufreq/scaling_max_freq")
    say "[core] far cpus -> ${CORE_MHZ} MHz (cpu$REMOTE_CPU readback max=$got kHz)"
    [ "$got" = "$CORE_KHZ" ] || say "[core] ASSERT FAILED: asked $CORE_KHZ kHz, got $got -- far cores are NOT at ${CORE_MHZ} MHz"
  }
else
  CORE_PATH=msr
  sudo modprobe msr 2>/dev/null || true
  command -v wrmsr >/dev/null || { echo "no cpufreq driver and no msr-tools"; exit 1; }
  for c in $FAR_CPUS; do ORIG_CMAX[$c]=$(sudo rdmsr -p "$c" 0x199); done
  restore_cores() { for c in $FAR_CPUS; do sudo wrmsr -p "$c" 0x199 "0x${ORIG_CMAX[$c]}"; done; echo "[core] restored (MSR)"; }
  set_cores() {
    local ratio=$((CORE_MHZ / 100)) now
    for c in $FAR_CPUS; do sudo wrmsr -p "$c" 0x199 $((ratio << 8)); done
    now=$(sudo rdmsr -p "$REMOTE_CPU" 0x199)
    say "[core] far cpus -> ${CORE_MHZ} MHz (MSR 0x199=0x$now)"
    [ $(( 0x$now >> 8 & 0xff )) = "$ratio" ] || say "[core] ASSERT FAILED: far cores are NOT at ${CORE_MHZ} MHz"
  }
fi

# --- far-uncore ratio, with the floor on record ----------------------------------------------
# hwguard resets a leaked throttle before ORIG_* is captured, and refuses an out-of-range ratio.
hwguard_uncore_init "$FAR_PKG" | tee -a "$PROOF"
hwguard_require_ratios "$RATIO" | tee -a "$PROOF" || exit 1

UNCORE_SYS=/sys/devices/system/cpu/intel_uncore_frequency
DOMAIN=$UNCORE_SYS/$(printf 'package_%02d_die_00' "$FAR_PKG")
if [ -d "$DOMAIN" ]; then
  THROTTLE=driver
  ORIG_MAX_KHZ=$(< "$DOMAIN/max_freq_khz"); ORIG_MIN_KHZ=$(< "$DOMAIN/min_freq_khz")
  HW_MIN_KHZ=$( [ -r "$DOMAIN/initial_min_freq_khz" ] && cat "$DOMAIN/initial_min_freq_khz" || echo "$ORIG_MIN_KHZ" )
  HW_MAX_KHZ=$( [ -r "$DOMAIN/initial_max_freq_khz" ] && cat "$DOMAIN/initial_max_freq_khz" || echo "$ORIG_MAX_KHZ" )
  say "[uncore] domain=$DOMAIN"
  say "[uncore] hardware range: min=$HW_MIN_KHZ max=$HW_MAX_KHZ kHz (= ratio $((HW_MIN_KHZ/100000))..$((HW_MAX_KHZ/100000)))"
  say "[uncore] as found:       min=$ORIG_MIN_KHZ max=$ORIG_MAX_KHZ kHz"
  wr() { echo "$2" | sudo tee "$DOMAIN/$1" >/dev/null; }
  restore_uncore() { wr min_freq_khz "$HW_MIN_KHZ"; wr max_freq_khz "$ORIG_MAX_KHZ"; wr min_freq_khz "$ORIG_MIN_KHZ"
                     echo "[uncore] restored to ${ORIG_MIN_KHZ}-${ORIG_MAX_KHZ} kHz"; }
  set_ratio() {
    local khz=$((RATIO * 100000)) gmax gmin
    wr min_freq_khz "$HW_MIN_KHZ"; wr max_freq_khz "$khz"; wr min_freq_khz "$khz"
    gmax=$(< "$DOMAIN/max_freq_khz"); gmin=$(< "$DOMAIN/min_freq_khz")
    say "[uncore] ratio=$RATIO asked=$khz kHz -> readback min=$gmin max=$gmax kHz"
    if [ "$gmax" != "$khz" ] || [ "$gmin" != "$khz" ]; then
      say "[uncore] ASSERT FAILED: this point is NOT ratio $RATIO. Requested below the $((HW_MIN_KHZ/100000)) floor?"
      exit 1
    fi
  }
else
  THROTTLE=msr
  sudo modprobe msr 2>/dev/null || true
  command -v wrmsr >/dev/null || { echo "need msr-tools"; exit 1; }
  ORIG=$(sudo rdmsr -p "$REMOTE_CPU" 0x620)
  say "[uncore] MSR 0x620 as found = 0x$ORIG (max=$(( 0x$ORIG & 0x7f )) min=$(( (0x$ORIG >> 8) & 0x7f )))"
  restore_uncore() { sudo wrmsr -p "$REMOTE_CPU" 0x620 "0x$ORIG"; echo "[uncore] restored MSR 0x620"; }
  set_ratio() {
    local now gmax gmin
    sudo wrmsr -p "$REMOTE_CPU" 0x620 $(( (RATIO << 8) | RATIO ))
    now=$(sudo rdmsr -p "$REMOTE_CPU" 0x620); gmax=$(( 0x$now & 0x7f )); gmin=$(( (0x$now >> 8) & 0x7f ))
    say "[uncore] ratio=$RATIO -> MSR=0x$now max=$gmax min=$gmin"
    [ "$gmax" = "$RATIO" ] && [ "$gmin" = "$RATIO" ] || { say "[uncore] ASSERT FAILED: not ratio $RATIO"; exit 1; }
  }
fi
say "[cfg] throttle=$THROTTLE core_path=$CORE_PATH"

restore() { restore_cores; restore_uncore; restore_nb; }
trap restore EXIT

# --- 3. counter discovery --------------------------------------------------------------------
# PMU names differ across uarchs, so discover rather than hardcode. UPI TX data is the number
# that matters: it is what "cross-socket bytes" means.
EVENTS=""; UPI_KIND=none
for base in /sys/bus/event_source/devices/uncore_imc_*; do
  [ -d "$base" ] || continue
  pmu=$(basename "$base")
  [ -f "$base/events/cas_count_read" ]  && EVENTS+="${EVENTS:+,}$pmu/cas_count_read/"
  [ -f "$base/events/cas_count_write" ] && EVENTS+="${EVENTS:+,}$pmu/cas_count_write/"
done
for base in /sys/bus/event_source/devices/uncore_upi_* /sys/bus/event_source/devices/uncore_qpi_*; do
  [ -d "$base" ] || continue
  pmu=$(basename "$base")
  if [ -f "$base/events/upi_data_bandwidth_tx" ]; then
    EVENTS+="${EVENTS:+,}$pmu/upi_data_bandwidth_tx/"; UPI_KIND=named
  else
    # UNC_UPI_TxL_FLITS.ALL_DATA -- raw flits, converted below as flits/9*64 bytes.
    EVENTS+="${EVENTS:+,}$pmu/event=0x02,umask=0x0f/"; UPI_KIND=raw
  fi
done
[ -n "$EVENTS" ] || { echo "no uncore IMC/UPI PMUs found -- cannot measure bytes"; exit 1; }
say "[perf] upi_event_kind=$UPI_KIND"
say "[perf] events=$EVENTS"

# --- build -------------------------------------------------------------------------------
if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ] \
   || [ "$REPO/include/flatnav/util/ValidationOffload.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -DFLATNAV_USE_NUMA spec_search.cpp"
  g++ -std=c++17 -O3 -march=native -DFLATNAV_USE_NUMA \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread -lnuma \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

set_cores
set_ratio

# --- run driver --------------------------------------------------------------------------
# Each line of the tool's output is stamped with the epoch second it arrived. The tool fflushes
# after every phase and every pipeline row, so a row's stamp is the instant its timed region
# ended -- subtract the row's own reported duration and the window is exact, with no dependence
# on when the process started or how long the pre-load query/GT work took.
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

sample_numa() {  # $1=pid $2=outfile -- per-node resident bytes, honouring mixed page sizes
  local pid=$1 out=$2
  # The tool runs as root under perf, so numa_maps needs sudo to read.
  while [ -d "/proc/$pid" ]; do
    { printf '%s ' "$(date +%s.%N)"
      sudo awk '{ ps=4; for(i=1;i<=NF;i++) if($i ~ /^kernelpagesize_kB=/){split($i,k,"="); ps=k[2]}
                  for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split($i,a,"="); n=substr(a[1],2); kb[n]+=a[2]*ps} }
                END{ for(n in kb) printf "node%s=%.2fGB ", n, kb[n]/1048576 }' "/proc/$pid/numa_maps"
      echo
    } >> "$out" 2>/dev/null || true
    sleep 5
  done
}

run() {
  local tag=$1 vecnode=$2 graphnode=$3 vo=$4 depths=$5
  echo "[run] $tag  (vec=node$vecnode graph=node$graphnode offload=${vo:-none} depths=${depths:-none})"
  local t0; t0=$(date +%s.%N)
  echo "$t0" > "$OUT/$tag.t0"

  sudo perf stat -a --per-socket -I 1000 -x , -e "$EVENTS" -o "$OUT/$tag.perf.csv" -- \
    env NQ="$NQ" DEPTHS="$depths" WIDTHS=1 PQ_M="$PQ_M" GT="$GT" \
        VEC_NODE="$vecnode" GRAPH_NODE="$graphnode" ${vo:+VO_CPUS="$vo"} \
        numactl --cpunodebind="$NEAR" "$BIN" "$IDX" "$Q" "$T" "$EF" "$K" 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$OUT/$tag.log" &
  local driver=$!

  # Placement sampling needs the tool's pid, which appears a moment after perf starts.
  local pid=""
  for _ in $(seq 1 60); do
    pid=$(pgrep -n -f "$(basename "$BIN") $IDX" || true); [ -n "$pid" ] && break; sleep 1
  done
  [ -n "$pid" ] && sample_numa "$pid" "$OUT/$tag.numa.txt" &
  local sampler=$!
  wait "$driver" || echo "  FAILED (see $OUT/$tag.log)"
  kill "$sampler" 2>/dev/null || true
}

if want baseline_near;  then run baseline_near  "$NEAR" "$NEAR" ""          ""        ; fi
if want baseline_far;   then run baseline_far   "$FAR"  "$NEAR" ""          ""        ; fi
if want offload;        then run offload        "$FAR"  "$NEAR" "$VO_CPUS"  "$DEPTHS" ; fi
# The null test: vectors already near, offload still on. Nothing to move compute toward.
if want nulltest;       then run nulltest       "$NEAR" "$NEAR" "$VO_CPUS"  "$DEPTHS" ; fi

# --- window extraction + byte accounting -----------------------------------------------------
# For each measured window, sum the uncore counters per socket. Everything is reported per query
# so baseline and offload are directly comparable; the UPI ratio between them is the claim.
report() {
  local tag=$1
  [ -s "$OUT/$tag.log" ] || return 0
  local t0; t0=$(< "$OUT/$tag.t0")
  echo
  echo "=== $tag ==="
  grep -E '\[offload\]|\[exact\]' "$OUT/$tag.log" | sed 's/^[0-9.]* //' || true

  # (stamp, label, duration) for every timed region: the exact baseline, then each pipeline row.
  awk -v nq="$NQ" '
    $2=="[exact]" { print $1, "exact", $6+0 }
    $2 ~ /^[0-9]+$/ && $3=="1" { print $1, "k="$2, nq/($10+0) }
  ' "$OUT/$tag.log" | while read -r stamp label dur; do
    lo=$(awk -v s="$stamp" -v t="$t0" -v d="$dur" -v g="$GUARD" 'BEGIN{printf "%.3f", s-t-d+g}')
    hi=$(awk -v s="$stamp" -v t="$t0" -v g="$GUARD" 'BEGIN{printf "%.3f", s-t-g}')
    # The raw UPI event spec contains a comma, so it shifts the trailing CSV fields. Classify on
    # the whole line and read the running% off the end, both of which survive the shift; the
    # socket ($2) and value ($4) sit before the event name and never move.
    awk -F, -v lo="$lo" -v hi="$hi" -v nq="$NQ" -v label="$label" -v dur="$dur" -v kind="$UPI_KIND" '
      $1+0>=lo && $1+0<=hi && $2 ~ /^S/ && $4 ~ /^[0-9]/ {
        v=$4+0; u=$5; s=$2; rows++
        if ($0 ~ /cas_count_read/)       imcr[s] += (u=="MiB"? v*1048576 : v*64)
        else if ($0 ~ /cas_count_write/) imcw[s] += (u=="MiB"? v*1048576 : v*64)
        else if ($0 ~ /upi|qpi/)         upi[s]  += v
        # perf emits trailing empty fields, so the running% is the last NON-empty one.
        pct=0; for (i=NF; i>=1; i--) if ($i != "") { pct=$i+0; break }
        if (pct > 0 && (minpr==0 || pct < minpr)) minpr=pct
      }
      END{
        if (!rows) { printf "  %-6s no perf rows in window [%.1f,%.1f]\n", label, lo, hi; exit }
        # A named upi event arrives already scaled to bytes; a raw one is flits (64B per 9 flits).
        # Reported per socket, not summed: S1 tx is vector data flowing back to the near socket,
        # which is the quantity the byte-cut claim is actually about.
        tot=0
        for (i=0; i<8; i++) { s="S" i
          if (s in upi) { ub[s] = (kind=="named" ? upi[s] : upi[s]/9*64); tot += ub[s] }
        }
        printf "  %-6s %5.2fs | UPI tx", label, dur
        for (i=0; i<8; i++) { s="S" i; if (s in ub) printf " %s=%.1fGB", s, ub[s]/1e9 }
        printf " = %7.0f B/query | DRAM", tot/nq
        for (i=0; i<8; i++) { s="S" i
          if (s in imcr) printf " %s rd=%.1fGB wr=%.1fGB", s, imcr[s]/1e9, imcw[s]/1e9
        }
        if (minpr > 0 && minpr < 99.5) printf "  [perf multiplexed, min running=%.1f%%]", minpr
        printf "\n"
      }' "$OUT/$tag.perf.csv"
  done

  if [ -s "$OUT/$tag.numa.txt" ]; then
    echo "  placement (last sample): $(tail -1 "$OUT/$tag.numa.txt" | cut -d' ' -f2-)"
  fi
  grep -hE '^[0-9.]+ (PASS|FAIL)' "$OUT/$tag.log" | sed 's/^[0-9.]* //' || true
}

echo
echo "================ summary ($OUT) ================"
echo "Read this as three questions:"
echo "  config   does config_proof.txt show ratio $RATIO applied, ${CORE_MHZ}MHz far cores, numa_balancing=0,"
echo "           and ~51 GB landing on the intended vector node in the placement line?"
echo "  bytes    is offload's UPI B/query ~64x below baseline_far's, at the same DRAM read volume?"
echo "  null     is nulltest NOT faster than baseline_near? If it is, the win is not locality."
for tag in baseline_near baseline_far offload nulltest; do report "$tag"; done
echo
echo "config proof: $PROOF"
echo "logs: $OUT"
