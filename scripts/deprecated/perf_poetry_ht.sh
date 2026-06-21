#!/usr/bin/env bash
# Hyperthreading A/B on the POETRY baseline (numpy1.26 + cp310 wheel).
# Runs the default runner twice, back-to-back, node-0 memory-bound, perf gated to
# the search phase, identical params except CPU pinning:
#   HT-off : 16 threads on the 16 physical primary cores (0,2,...,30)  -> 1 thr/core
#   HT-on  : 32 threads on all 32 node-0 logical CPUs (0,2,...,62)     -> 2 thr/core (=baseline)
set -u
REPO=$HOME/flatnav
VENV=$(cd "$HOME/poetry_bw" && PATH="$HOME/.local/poetry/bin:$PATH" poetry env info -p)/bin/python
DATASET=/data/index/sift100m_base.fvecs
Q=/data/queries/sift100m_200k_extra_query.fvecs
GT=/data/queries/sift100m_200k_extra_query.gtruth.ivecs
MTX=/data/index/sift100m_m32_hnsw_base_layer.mtx
EFL="200 200 200 200"; SBS=128; PEAK=102.8
export PYTHONPATH="$REPO/experiments"

PHYS=$(seq -s, 0 2 30)     # 16 physical primaries of node-0
ALL=$(seq -s, 0 2 62)      # all 32 logical CPUs of node-0

RUNROOT=$HOME/poetry_ht_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUNROOT"
echo "run root: $RUNROOT"
"$VENV" -c 'import numpy,flatnav; print("venv numpy",numpy.__version__,"flatnav",flatnav.__version__)'

run_one () {
  local TAG=$1 CPUS=$2 THREADS=$3
  local EXP=ht_${TAG}_$(date +%H%M%S)
  local OUT=$RUNROOT/$TAG; mkdir -p "$OUT"
  echo "================ $TAG : ${THREADS}T on cpus=$CPUS ================"
  rm -f "/tmp/measurement/$EXP.ready"
  ( cd "$REPO/experiments" && numactl --membind=0 --physcpubind="$CPUS" \
     "$VENV" sift_big_flatnav_recall_batched.py \
       --dataset "$DATASET" --queries "$Q" --gtruth "$GT" --existing-mtx "$MTX" \
       --metric l2 --num-node-links 32 --ef-search $EFL \
       --num-search-threads "$THREADS" --search-batch-size "$SBS" \
       --search-only --exp-id "$EXP" --output-json "$OUT/recall.json" ) > "$OUT/runner.log" 2>&1 &
  local PYPID=$!
  echo "$TAG runner PID=$PYPID — building (~14 min)..."
  while [ ! -f "/tmp/measurement/$EXP.ready" ] && kill -0 "$PYPID" 2>/dev/null; do sleep 3; done
  kill -0 "$PYPID" 2>/dev/null || { echo "$TAG BUILD_FAILED"; tail -20 "$OUT/runner.log"; return 1; }
  echo "$TAG build done $(date +%T) — measuring search..."
  local IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
  local CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"
  sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$OUT/perf.csv" -- \
    bash -c "while kill -0 $PYPID 2>/dev/null; do sleep 0.3; done" 2>/dev/null
  awk -F, -v PEAK="$PEAK" -v TAG="$TAG" -v T="$THREADS" '
    $1+0>0 && $4!="" { ev=$4; v=$2;
      if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
      else if(ev=="cycles") cyc+=v; else if(ev=="instructions") ins+=v;
      else if(ev=="l1d_pend_miss.pending") p+=v; else if(ev=="l1d_pend_miss.pending_cycles") pc+=v;
      if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
    END{ w=hi-lo; printf "%-7s %2dT  win=%3.0fs  BW=%5.1f GB/s  util=%2.0f%%  IPC=%.3f  mem_stall=%4.1f%%  MLP=%.2f\n",
          TAG, T, w, cas/w/1e9, 100*(cas/w/1e9)/PEAK, ins/cyc, 100*pc/cyc, p/pc }' "$OUT/perf.csv" | tee -a "$RUNROOT/summary.txt"
  "$VENV" - "$OUT/recall.json" "$TAG" <<'PY' | tee -a "$RUNROOT/summary.txt"
import json,sys
d=json.load(open(sys.argv[1]))
for ef,v in d["results"].items():
    print(f"        {sys.argv[2]}: ef={ef} recall@{d['k']}={v['recall']:.4f} qps={v['qps']:.0f}")
PY
}

run_one htoff "$PHYS" 16
run_one hton  "$ALL"  32
echo "===== HT A/B done ====="; cat "$RUNROOT/summary.txt"
echo "raw: $RUNROOT"
