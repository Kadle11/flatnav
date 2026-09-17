#!/usr/bin/env bash
# How much of a search is compute and how much is waiting for memory? (tools/hbw_search.cpp)
#
# Every other script in this directory measures the OUTSIDE of the search -- QPS, recall, latency
# under a throttled far node. None of them says where the cycles actually go, so "graph search is
# memory-bound" has been an inference from working-set geometry (roofline_probe) and from the fact
# that throttling the far node hurts. This takes the measurement directly, off the core's PMU.
#
# What "waiting" means here. On an out-of-order core compute and memory OVERLAP by construction --
# the distance kernel's FMAs on vector i run while the line for vector i+1 is in flight -- so there
# is no honest split into "X ns computing, Y ns waiting". The well-posed version is cycle
# accounting: of all issue slots, what fraction retired useful work, and what fraction went nowhere
# because a load was outstanding. That is exactly TMA's definition, and on Sapphire Rapids the
# level-1 and level-2 buckets come out of PERF_METRICS in hardware -- no slot formula to hand-roll,
# no SMT correction to get wrong. `Backend_Bound.Memory_Bound` IS the answer to the question.
#
#   ./run_stall_profile.sh                             # single-node, everything near
#   VEC_NUMA=1 GRAPH_NUMA=0 ./run_stall_profile.sh     # the tiered placement: far vectors, near graph
#   NQ_EF=80 REPEAT=20 ./run_stall_profile.sh          # longer steady window, less dilution
#
# Passes, and what each one settles:
#   tma     TMA L1+L2 from PERF_METRICS. Retiring vs Frontend/Bad-Spec/Backend, and Backend split
#           into Memory_Bound vs Core_Bound. The headline compute-vs-stall number.
#   stalls  cycle_activity.stalls_* / cycles. Cross-checks tma from a different counter family and
#           attributes the stall to a level: L2-miss, L3-miss (i.e. DRAM). Undercounts overlap, so
#           it reads as a LOWER bound on memory cost -- if it disagrees with tma, tma wins.
#   mlp     l1d_pend_miss.pending / pending_cycles = average misses in flight. Separates the two
#           very different memory-bound worlds: low MLP means latency-bound (the prefetch and
#           speculation work is aimed at exactly this), high MLP means bandwidth-bound and no
#           amount of earlier issue helps.
#   dram    mem_load_l3_miss_retired.local_dram vs .remote_dram. Which side of the interconnect the
#           misses are served from -- the number the near/far tiering is actually trying to move.
#   bw      uncore IMC CAS counts per socket -> achieved read GB/s. Against the part's roofline this
#           says whether the memory system is saturated or idling while the core stalls.
#   mem     PEBS load-latency sampling (ldlat=128), reported by memory level AND symbol. Aggregate
#           percentages cannot tell a graph-link load from a vector load; this can, which is the
#           whole question for hot-links-near / vectors-far.
#
# Search-only by construction. hbw_search prints `[phase] warmup_done elapsed_s=` on the same clock
# perf counts from, so a calibration run (no perf) gives the millisecond at which the search starts
# and every pass is armed with `perf -D`. Index load, NUMA placement and warmup are outside the
# counters rather than diluted into them. Each pass re-reads its own marker and the script refuses
# the run if it drifted from calibration -- a delay that no longer lines up would silently fold
# load-phase cycles into a "search" profile, which is the failure this guards.
#
# Takes the uncore at whatever state it is in; throttled far-memory sweeps live in
# run_roofline_bw_sweep.sh and run_offload_100m.sh. Runs unprivileged: this node has
# perf_event_paranoid=-1, which is checked up front rather than assumed.
set -euo pipefail

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/hwguard.sh"

ROOT=${ROOT:-$HOME/vishal}
PATHS=${PATHS:-$ROOT/.spec_paths.env}
# shellcheck disable=SC1090
[ -f "$PATHS" ] && . "$PATHS"        # only fills IDX/Q/GT/REPO when they are unset

IDX=${IDX:-}
Q=${Q:-}
GT=${GT:-}
REPO=${REPO:-$([ -d "$ROOT/flatnav" ] && echo "$ROOT/flatnav" || echo "$HOME/flatnav")}
BIN=${BIN:-$HOME/hbw_search_stall}
SRC=$REPO/tools/hbw_search.cpp

T=${T:-32}                       # search threads
K=${K:-10}
NQ_EF=${NQ_EF:-80}               # the ef every repeat runs at
REPEAT=${REPEAT:-12}             # repeats of that ef -> one long steady search window, so the
                                 # counters see traversal and not the tail of warmup
EF=$(awk -v n="$REPEAT" -v e="$NQ_EF" 'BEGIN{s=e;for(i=1;i<n;i++)s=s","e;print s}')

NEAR=${NEAR:-0}                  # cpus, and (unless VEC_NUMA/GRAPH_NUMA say otherwise) all memory
FAR=${FAR:-1}
VEC_NUMA=${VEC_NUMA:-}           # -> FLATNAV_VECTORS_NUMA; unset = default allocator
GRAPH_NUMA=${GRAPH_NUMA:-}       # -> FLATNAV_GRAPH_NUMA
LDLAT=${LDLAT:-128}              # PEBS load-latency threshold, cycles. 128 ~ "missed to memory"
GUARD_MS=${GUARD_MS:-300}        # added to the calibrated warmup marker before arming the counters
DRIFT_MS=${DRIFT_MS:-1500}       # per-pass marker drift that invalidates the calibrated delay

ONLY=${ONLY:-all}
OUT=${OUT:-$ROOT/stall_profile_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

for v in IDX Q GT; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -s "$p" ] || { echo "missing $v (${p:-unset}) -- run ./setup_100m.sh, or set $v="; exit 1; }
done
[ -d "$REPO/include/flatnav" ] || { echo "no flatnav checkout at $REPO (set REPO=)"; exit 1; }
command -v perf >/dev/null || { echo "no perf on PATH"; exit 1; }
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 2)
# The bw pass is system-wide and reads the uncore PMU, which needs paranoid < 0. Everything else
# is per-process and works at <= 1. Say which passes are about to come back empty rather than
# letting them fail one by one inside the run.
[ "$PARANOID" -le 0 ] 2>/dev/null || echo "[warn] perf_event_paranoid=$PARANOID: the bw (uncore) pass needs -1. sysctl -w kernel.perf_event_paranoid=-1"

pkg_of_cpu() { local f="/sys/devices/system/cpu/cpu$1/topology/physical_package_id"; [ -r "$f" ] && cat "$f"; }
for n in "$NEAR" "$FAR"; do
  [ -d "/sys/devices/system/node/node$n" ] || {
    echo "no NUMA node $n on this host. Available: $(ls -d /sys/devices/system/node/node[0-9]* | sed 's#.*/node##' | tr '\n' ' ')"
    exit 1; }
done
NEAR_CPU=$(cut -d, -f1 "/sys/devices/system/node/node$NEAR/cpulist" | cut -d- -f1)
NEAR_PKG=$(pkg_of_cpu "$NEAR_CPU")

trap hwguard_restore_all EXIT
hwguard_numa_balancing_off
hwguard_record_config "$OUT/config_proof.txt"

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ] || [ "$REPO/include/flatnav/index/Index.h" -nt "$BIN" ]; then
  echo "[build] g++ -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA hbw_search.cpp"
  g++ -std=c++17 -O3 -march=native -fopenmp -DFLATNAV_USE_NUMA \
    -I "$REPO/include" -I "$REPO/external/cereal/include" \
    "$SRC" -o "$BIN" -lpthread -lnuma \
    2>&1 | grep -iE "error|undefined" && { echo BUILD_FAIL; exit 1; }
fi

# --- placement ------------------------------------------------------------------------------
# cpus always on NEAR. Memory is membind=NEAR only in the untiered case; when VEC_NUMA/GRAPH_NUMA
# are set, hbw_search places each region itself with numa_alloc_onnode and a membind would fight it.
PIN=(numactl --cpunodebind="$NEAR")
ENV=(env)
if [ -n "$VEC_NUMA" ] || [ -n "$GRAPH_NUMA" ]; then
  [ -n "$VEC_NUMA" ]   && ENV+=(FLATNAV_VECTORS_NUMA="$VEC_NUMA")
  [ -n "$GRAPH_NUMA" ] && ENV+=(FLATNAV_GRAPH_NUMA="$GRAPH_NUMA")
else
  PIN+=(--membind="$NEAR")
fi
CMD=("${ENV[@]}" "${PIN[@]}" "$BIN" "$IDX" "$Q" "$GT" "$K" "$T" "$EF")

# --- event probing --------------------------------------------------------------------------
# Ice Lake removed the CYCLE_ACTIVITY L3-miss umasks and several offcore_requests_outstanding
# ones, and SPR inherits that. A name this part does not know is not a soft failure: perf rejects
# the WHOLE -e list and counts nothing, so every event in the group reads zero and a "0% stalled"
# line lands in the summary looking like a measurement. Probe each name on its own and drop it.
#
# Exit status, not message matching -- perf's wording for an unknown event varies by version
# ("event syntax error", "unknown term", "Cannot find PMU"). Deliberately tolerant of
# "<not counted>": `true` exits before some counters tick, and a supported event must not be
# dropped for that. No sudo anywhere in this script: paranoid=-1 on this node, checked below.
have_ev() {
  local out
  out=$(perf stat -e "$1" -x, true 2>&1) || return 1
  printf '%s' "$out" | grep -qi "not supported" && return 1
  return 0
}
keep_evs() { local out="" e; for e in $1; do have_ev "$e" && out+="${out:+,}$e" || echo "  [drop] $e not supported here" >&2; done; echo "$out"; }

marker_of() { sed -n 's/.*warmup_done elapsed_s=\([0-9.]*\).*/\1/p' "$1" | tail -1; }
searchsec_of() { awk '$1 ~ /^[0-9]+$/ && NF>=4 {s+=$2} END{printf "%.3f", s}' "$1"; }

# --- calibration: where does the search actually start? -------------------------------------
echo "[cfg] idx=$IDX"
echo "[cfg] q=$Q gt=$GT K=$K threads=$T ef=${NQ_EF}x${REPEAT}"
echo "[cfg] cpus=node$NEAR (pkg $NEAR_PKG) vectors=${VEC_NUMA:-default} graph=${GRAPH_NUMA:-default} ldlat=$LDLAT"
echo "[cfg] perf_event_paranoid=$PARANOID -> $OUT"
numactl -H | grep -E "^node ($NEAR|$FAR) (cpus|size|free)" || true

echo "[calib] timing run, no counters attached..."
"${CMD[@]}" > "$OUT/calib.log" 2>&1 || { echo "calibration run FAILED (see $OUT/calib.log)"; exit 1; }
WARM=$(marker_of "$OUT/calib.log")
[ -n "$WARM" ] || { echo "no 'warmup_done elapsed_s=' marker in $OUT/calib.log -- is \$BIN really hbw_search?"; exit 1; }
DELAY_MS=$(awk -v w="$WARM" -v g="$GUARD_MS" 'BEGIN{printf "%d", w*1000 + g}')
SEARCH_S=$(searchsec_of "$OUT/calib.log")
echo "[calib] warmup_done=${WARM}s  search=${SEARCH_S}s  -> arming counters at -D ${DELAY_MS}ms"

# --- one profiling pass ---------------------------------------------------------------------
# Tolerant on purpose: a pass whose events this part refuses should not discard the others. But a
# marker that moved means the calibrated delay no longer points at the search, so that is fatal.
run_stat() {   # $1=tag  $2=events  $3...=extra perf args
  local tag=$1 evs=$2; shift 2
  [ -n "$evs" ] || { echo "[run] $tag -- no supported events, skipped"; return 0; }
  echo "[run] $tag"
  perf stat -D "$DELAY_MS" -x, -e "$evs" "$@" -o "$OUT/$tag.csv" \
    -- "${CMD[@]}" > "$OUT/$tag.prog" 2>"$OUT/$tag.err" || { echo "  FAILED (see $OUT/$tag.err)"; return 0; }
  local m; m=$(marker_of "$OUT/$tag.prog")
  [ -n "$m" ] || { echo "  no marker in $tag.prog"; return 0; }
  local d; d=$(awk -v a="$m" -v b="$WARM" 'BEGIN{d=(a-b)*1000; printf "%d", d<0?-d:d}')
  [ "$d" -le "$DRIFT_MS" ] || {
    echo "  ABORT: $tag warmup_done=${m}s drifted ${d}ms from the calibrated ${WARM}s."
    echo "  The -D ${DELAY_MS}ms window no longer starts at the search, so these counts would be"
    echo "  part load phase archived as a search profile. Re-run; raise DRIFT_MS only deliberately."
    exit 1; }
}

# $1=csv $2=event substring -> summed value ("" when absent). Matches by name, not column index,
# so --per-socket rows (which shift every field right) parse with the same helper.
val() { awk -F, -v e="$2" '{for(i=1;i<=NF;i++) if($i==e){gsub(/[^0-9.]/,"",$1); if($1!="") s+=$1}} END{printf "%s", (s==""?"":s)}' "$1" 2>/dev/null; }
pct() { awk -v n="$1" -v d="$2" 'BEGIN{if(d>0 && n!="") printf "%.1f", 100*n/d; else printf "?"}'; }

want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

# tma: PERF_METRICS. slots must lead the group -- these are metric registers, not counters.
if want tma; then
  run_stat tma "{slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound}"
  # The human-readable form too: perf applies the official metric names and SMT handling itself,
  # so it is the reference if the ratios below ever look off.
  perf stat -D "$DELAY_MS" --topdown --td-level=2 -o "$OUT/tma_text.txt" \
    -- "${CMD[@]}" > /dev/null 2>&1 || true
fi

# stalls: note the absence of cycle_activity.stalls_mem_any. It is the event every Skylake-era
# recipe reaches for and Ice Lake removed it; SPR still has stalls_l3_miss, so the umask that
# disappeared is not the one you would guess. stalls_l1d_miss is the broadest memory stall left
# (stalled with an L1D miss outstanding) and stands in for it.
# cycles_mem_any is the counterweight: cycles with a miss in flight WITHOUT requiring the core to
# be stalled. cycles_mem_any - stalls_l1d_miss is the overlap this script's header is about --
# memory in flight while the core still retired work, i.e. latency that cost nothing.
if want stalls; then
  run_stat stalls "$(keep_evs "cycles instructions cycle_activity.stalls_total cycle_activity.stalls_l1d_miss cycle_activity.stalls_l2_miss cycle_activity.stalls_l3_miss cycle_activity.cycles_mem_any")"
fi

if want mlp; then
  run_stat mlp "$(keep_evs "cycles l1d_pend_miss.pending l1d_pend_miss.pending_cycles offcore_requests_outstanding.demand_data_rd offcore_requests_outstanding.cycles_with_demand_data_rd")"
fi

if want dram; then
  run_stat dram "$(keep_evs "cycles mem_load_retired.l3_miss mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram")"
fi

# bw: uncore is package-scoped, so this one is system-wide and per-socket. Note cas_count_read,
# with an UNDERSCORE: the dotted cas_count.read spelling is the pre-ICX one and this kernel
# rejects it. The free_running PMU is listed here too but on this node exposes only dclk/rpq/wpq,
# no data_read, so it is a fallback for other hosts rather than a real second option.
if want bw; then
  BW_EV=""
  for cand in "uncore_imc/cas_count_read/,uncore_imc/cas_count_write/" \
              "uncore_imc/cas_count.read/,uncore_imc/cas_count.write/" \
              "uncore_imc_free_running/data_read/,uncore_imc_free_running/data_write/"; do
    have_ev "${cand%%,*}" && { BW_EV=$cand; break; }
  done
  [ -n "$BW_EV" ] || echo "  [drop] no IMC events exposed (uncore PMU not loaded?)"
  run_stat bw "$BW_EV" -a --per-socket
fi

# mem: PEBS. -c is a sample period, not a frequency -- a fixed period keeps the level histogram
# unbiased across the run instead of over-sampling whatever phase happens to stall most.
#
# The {mem-loads-aux, mem-loads} group is not decoration: from Ice Lake on, the data source is
# carried by a separate aux event and perf refuses the load-latency event on its own
# ("Cannot collect data source with the load latency event alone"). This is the same pair
# `perf mem record` builds internally; it is spelled out here because that wrapper does not
# forward -D, and without -D the samples would be dominated by the index load.
if want mem; then
  echo "[run] mem (PEBS ldlat=$LDLAT)"
  MEM_EV="{cpu/mem-loads-aux/,cpu/mem-loads,ldlat=$LDLAT/P}"
  have_ev "$MEM_EV" || MEM_EV="cpu/mem-loads,ldlat=$LDLAT/P"    # pre-ICL spelling, if ever run there
  if have_ev "$MEM_EV"; then
    perf record -D "$DELAY_MS" -e "$MEM_EV" -d -c 20011 \
      -o "$OUT/mem.data" -- "${CMD[@]}" > "$OUT/mem.prog" 2>"$OUT/mem.err" || echo "  FAILED (see $OUT/mem.err)"
    if [ -s "$OUT/mem.data" ]; then
      perf report -i "$OUT/mem.data" --mem-mode --sort=mem --stdio    > "$OUT/mem_by_level.txt"  2>/dev/null || true
      perf report -i "$OUT/mem.data" --mem-mode --sort=symbol --stdio > "$OUT/mem_by_symbol.txt" 2>/dev/null || true
      chown "$(id -u):$(id -g)" "$OUT/mem.data" 2>/dev/null || true
    fi
  else
    echo "  [drop] cpu/mem-loads,ldlat=$LDLAT/P not available"
  fi
fi

# --- summary ---------------------------------------------------------------------------------
{
  echo "=== stall profile: $(hostname) $(date -Is) ==="
  echo "idx=$IDX ef=${NQ_EF}x${REPEAT} K=$K T=$T cpus=node$NEAR vectors=${VEC_NUMA:-default} graph=${GRAPH_NUMA:-default}"
  echo "window: perf -D ${DELAY_MS}ms (warmup_done=${WARM}s), search=${SEARCH_S}s of program time"
  echo

  SLOTS=$(val "$OUT/tma.csv" slots)
  if [ -n "$SLOTS" ]; then
    echo "-- TMA, share of issue slots (PERF_METRICS, level 1 then the backend split) --"
    printf "  %-22s %6s%%\n" retiring     "$(pct "$(val "$OUT/tma.csv" topdown-retiring)"  "$SLOTS")"
    printf "  %-22s %6s%%\n" bad_speculation "$(pct "$(val "$OUT/tma.csv" topdown-bad-spec)" "$SLOTS")"
    printf "  %-22s %6s%%\n" frontend_bound "$(pct "$(val "$OUT/tma.csv" topdown-fe-bound)" "$SLOTS")"
    printf "  %-22s %6s%%\n" backend_bound  "$(pct "$(val "$OUT/tma.csv" topdown-be-bound)" "$SLOTS")"
    printf "  %-22s %6s%%   <-- time the core had nothing to issue with a load outstanding\n" \
           "  .memory_bound" "$(pct "$(val "$OUT/tma.csv" topdown-mem-bound)" "$SLOTS")"
    echo
  else
    echo "-- TMA: no slots counted (see $OUT/tma.err); perf_event_paranoid=$PARANOID --"; echo
  fi

  CYC=$(val "$OUT/stalls.csv" cycles); INS=$(val "$OUT/stalls.csv" instructions)
  if [ -n "$CYC" ]; then
    echo "-- stall cycles, share of core cycles (lower bound: overlap is not counted) --"
    printf "  %-22s %6s\n"  IPC "$(awk -v i="$INS" -v c="$CYC" 'BEGIN{if(c>0)printf "%.2f",i/c; else printf "?"}')"
    printf "  %-22s %6s%%\n" stalled_any    "$(pct "$(val "$OUT/stalls.csv" cycle_activity.stalls_total)"     "$CYC")"
    printf "  %-22s %6s%%\n" stalled_memory "$(pct "$(val "$OUT/stalls.csv" cycle_activity.stalls_l1d_miss)" "$CYC")"
    printf "  %-22s %6s%%\n" "  .l2_miss"   "$(pct "$(val "$OUT/stalls.csv" cycle_activity.stalls_l2_miss)"  "$CYC")"
    printf "  %-22s %6s%%   <-- stalled with the miss all the way out at DRAM\n" \
           "  .l3_miss"   "$(pct "$(val "$OUT/stalls.csv" cycle_activity.stalls_l3_miss)"  "$CYC")"
    MEMCYC=$(val "$OUT/stalls.csv" cycle_activity.cycles_mem_any)
    STL1=$(val "$OUT/stalls.csv" cycle_activity.stalls_l1d_miss)
    [ -n "$MEMCYC" ] && [ -n "$STL1" ] && \
      printf "  %-22s %6s%%   <-- miss in flight but the core still retired: latency that cost nothing\n" \
        overlapped "$(pct "$(awk -v a="$MEMCYC" -v b="$STL1" 'BEGIN{print (a>b)?a-b:0}')" "$CYC")"
    echo
  fi

  PEND=$(val "$OUT/mlp.csv" l1d_pend_miss.pending); PCYC=$(val "$OUT/mlp.csv" l1d_pend_miss.pending_cycles)
  if [ -n "$PEND" ] && [ -n "$PCYC" ]; then
    echo "-- memory-level parallelism (misses in flight while any is outstanding) --"
    printf "  %-22s %6s    <-- low => latency-bound (issue earlier helps); high => bandwidth-bound\n" \
      avg_l1d_misses "$(awk -v p="$PEND" -v c="$PCYC" 'BEGIN{if(c>0)printf "%.2f",p/c; else printf "?"}')"
    echo
  fi

  L3M=$(val "$OUT/dram.csv" mem_load_retired.l3_miss)
  LOC=$(val "$OUT/dram.csv" mem_load_l3_miss_retired.local_dram)
  REM=$(val "$OUT/dram.csv" mem_load_l3_miss_retired.remote_dram)
  if [ -n "$LOC" ] || [ -n "$REM" ]; then
    echo "-- where the L3 misses were served (retired loads) --"
    printf "  %-22s %6s%%\n" local_dram  "$(pct "$LOC" "$(awk -v a="${LOC:-0}" -v b="${REM:-0}" 'BEGIN{print a+b}')")"
    printf "  %-22s %6s%%\n" remote_dram "$(pct "$REM" "$(awk -v a="${LOC:-0}" -v b="${REM:-0}" 'BEGIN{print a+b}')")"
    printf "  %-22s %6s\n"   l3_miss_loads "${L3M:-?}"
    echo
  fi

  if [ -s "$OUT/bw.csv" ]; then
    echo "-- achieved DRAM bandwidth over the ${SEARCH_S}s search window --"
    # --per-socket -x, rows are: S<n>,<ncpus>,<value>,<unit>,<event>,<runtime>,<pct>
    #
    # The unit field decides the arithmetic and must not be assumed. sysfs declares
    # cas_count_read.scale=6.1e-5 with unit MiB, but this kernel hands back RAW CAS counts and
    # leaves the unit empty, so a cache line is 64 B of traffic per count. A kernel that does
    # apply the scale reports MiB directly, and multiplying those by 64 would overstate
    # bandwidth by ~67 million times -- a number so large it would be caught, unlike the reverse.
    # Read the unit, do the matching conversion, and print which one was used.
    awk -F, -v s="$SEARCH_S" '
      $1 ~ /^S[0-9]+/ && ($5 ~ /cas_count.read|data_read/ || $5 ~ /cas_count.write|data_write/) {
        unit = $4; v = $3; gsub(/[^0-9.]/, "", v)
        if (v == "" || s <= 0) next
        if (unit ~ /MiB/) { gb = v * 1048576 / 1e9; how = "scaled" }
        else              { gb = v * 64 / 1e9;      how = "raw CAS x64B" }
        dir = ($5 ~ /read/) ? "read" : "write"
        printf "  %-22s %6.1f GB/s   (%s)\n", $1 "_" dir, gb / s, how
      }
    ' "$OUT/bw.csv"
    echo
  fi

  if [ -s "$OUT/mem_by_level.txt" ]; then
    echo "-- sampled load latency by memory level (>= $LDLAT cycles) --"
    # Drop the N/A row: those are mem-loads-aux's own samples, which carry no data source. Left in,
    # it shows up as a ~100% bucket and swamps the levels that are the point of the pass.
    grep -v '^#' "$OUT/mem_by_level.txt" | grep -E '^\s+[0-9]' | grep -v 'N/A' | head -12
    echo "  full report: $OUT/mem_by_level.txt ; by symbol: $OUT/mem_by_symbol.txt"
  fi
} | tee "$OUT/summary.txt"

echo
echo "logs: $OUT"
