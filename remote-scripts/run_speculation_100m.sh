#!/usr/bin/env bash
# Reproduce the speculate-validate measurements (.claude/assets/speculate_validate_pipeline.md)
# on SIFT100M. Everything here was measured on SIFT1M locally; this is the 100M rerun.
# Paths come from setup_100m.sh (via $ROOT/.spec_paths.env) unless given explicitly.
#
#   ./run_speculation_100m.sh                 # full set, 20k queries
#   NQ=200000 ./run_speculation_100m.sh       # all queries (slower: the replay is per-step)
#   ONLY=matrix ./run_speculation_100m.sh     # one section: main|matrix|gate|margin
#
# Sections
#   main   pq_top1 at delay 1: mixed vs all-PQ miss, safe delay k*, top-w coverage, and the
#          validation schedules (eager M15 vs lazy decision-relevance)
#   matrix pq_top1 over delays 1,2,3,4,6,8 x widths <=16 -> the depth x width miss matrix
#   gate   pq_top1 with the compression-error gate at 25% and 35% (the random-gate control)
#   margin pq_margin (M15): offline detector curve + online recall/reads frontier
#
# Pinned to node 0 (cpus 0-31,64-95; ~258 GB) of the dual-socket, 2-NUMA-node node. These tools
# measure decisions, not placement, so the pinning only keeps timings comparable; node 1
# (cpus 32-63,96-127) is the far node for later latency work.
set -euo pipefail

ROOT=${ROOT:-$HOME/vishal}
PATHS=${PATHS:-$ROOT/.spec_paths.env}
# shellcheck disable=SC1090
[ -f "$PATHS" ] && . "$PATHS"        # only fills IDX/Q/GT/REPO when they are unset

IDX=${IDX:-}
Q=${Q:-}
GT=${GT:-}
TOP1=${TOP1:-$HOME/pq_top1_bench}
MARGIN=${MARGIN:-$HOME/pq_margin_bench}
T=${T:-64}                  # node 0 has 64 logical cpus (32 cores x 2 HT)
EF=${EF:-200}
K=${K:-100}
NQ=${NQ:-20000}             # the per-step replay is the cost here, not the search
CPUNODE=${CPUNODE:-0}
MEMNODE=${MEMNODE:-0}
ONLY=${ONLY:-all}
OUT=${OUT:-$ROOT/spec_100m_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

for v in IDX Q GT; do
  eval "p=\$$v"
  [ -n "$p" ] && [ -s "$p" ] || { echo "missing $v (${p:-unset}) -- run ./setup_100m.sh, or set $v="; exit 1; }
done
for b in "$TOP1" "$MARGIN"; do [ -x "$b" ] || { echo "missing $b -- run ./setup_100m.sh"; exit 1; }; done

PIN=(numactl --cpunodebind="$CPUNODE" --membind="$MEMNODE")
run() { local tag=$1; shift; echo "[run] $tag"; "$@" > "$OUT/$tag.log" 2>&1 || { echo "  FAILED (see $OUT/$tag.log)"; return 1; }; }
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

echo "[cfg] idx=$IDX"
echo "[cfg] q=$Q nq=$NQ ef=$EF K=$K threads=$T node=$CPUNODE -> $OUT"

# --- main: every table at delay 1 ----------------------------------------------------------
if want main; then
  run main env NQ="$NQ" WMAX=16 SAFE_MAX=16 \
      STEP_CSV="$OUT/step_bins.csv" SAFE_CSV="$OUT/safe_delay.csv" COV_CSV="$OUT/topw_coverage.csv" \
      "${PIN[@]}" "$TOP1" "$IDX" "$Q" "$T" "$EF" "$K"
fi

# --- matrix: depth x width -----------------------------------------------------------------
if want matrix; then
  for k in 1 2 3 4 6 8; do
    run "matrix_delay$k" env NQ="$NQ" DELAY="$k" WMAX=16 \
        COV_CSV="$OUT/topw_coverage_delay$k.csv" STEP_CSV="$OUT/step_bins_delay$k.csv" \
        "${PIN[@]}" "$TOP1" "$IDX" "$Q" "$T" "$EF" "$K"
  done
fi

# --- gate: compression-error gate vs its equal-share random control ------------------------
# The low-error 25% of nodes covers ~35% of discovered neighbors, so the 0.35 run is the
# control that matches PQ share -- that comparison is the point of this section.
if want gate; then
  for frac in 0.25 0.35; do
    run "gate_frac$frac" env NQ="$NQ" PQ_FRAC="$frac" STEP_CSV="$OUT/step_bins_frac$frac.csv" \
        "${PIN[@]}" "$TOP1" "$IDX" "$Q" "$T" "$EF" "$K"
  done
fi

# --- margin: M15 detector + online frontier (needs ground truth) ---------------------------
if want margin; then
  run margin env NQ="$NQ" DET_CSV="$OUT/margin_detector.csv" FRONT_CSV="$OUT/margin_frontier.csv" \
      "${PIN[@]}" "$MARGIN" "$IDX" "$Q" "$GT" "$T" "$EF" "$K"
fi

# --- summary -------------------------------------------------------------------------------
echo
echo "=== summary ($OUT) ==="
if [ -s "$OUT/main.log" ]; then
  echo "-- miss rates / safe delay / top-w (all steps) --"
  grep -E "^all " "$OUT/main.log" || true
  echo "-- validation schedules --"
  sed -n '/Validation schedules/,$p' "$OUT/main.log" | grep -E "^(exact|eager|lazy)" || true
fi
if ls "$OUT"/matrix_delay*.log >/dev/null 2>&1; then
  echo "-- depth x width: coverage per delay (w=1,2,3,4,6,8,16) --"
  for k in 1 2 3 4 6 8; do
    [ -s "$OUT/matrix_delay$k.log" ] || continue
    printf 'k=%s ' "$k"
    sed -n '/Top-w coverage/,/Safe validation/p' "$OUT/matrix_delay$k.log" | grep -E "^all " | cut -c11- || true
  done
fi
[ -s "$OUT/margin.log" ] && { echo "-- M15 frontier --"; grep "^RESULT" "$OUT/margin.log" || true; }
echo
echo "logs + csv: $OUT"
