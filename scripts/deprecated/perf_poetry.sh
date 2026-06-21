#!/usr/bin/env bash
# Run the DEFAULT python runner (sift_big_flatnav_recall_batched.py) through the
# POETRY venv (numpy 1.26.4 + locally-built cp310 flatnav wheel, -Ofast -march=native,
# label fix included) and measure DRAM bandwidth over the search phase only.
# Apples-to-apples with the direct _core run: node-0 pinned, 32T, ef=200x4, batch=128.
set -u
REPO=$HOME/flatnav
VENV=$(cd "$HOME/poetry_bw" && PATH="$HOME/.local/poetry/bin:$PATH" poetry env info -p)/bin/python
DATASET=/data/index/sift100m_base.fvecs
Q=/data/queries/sift100m_200k_extra_query.fvecs
GT=/data/queries/sift100m_200k_extra_query.gtruth.ivecs
MTX=/data/index/sift100m_m32_hnsw_base_layer.mtx
THREADS=32
EFL="200 200 200 200"
SBS=128
PEAK=102.8
EXP=poetry_$(date +%H%M%S)
OUT=$HOME/poetry_bw_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
export PYTHONPATH="$REPO/experiments"

echo "venv: $VENV"
"$VENV" -c 'import numpy,flatnav,hnswlib; print("numpy",numpy.__version__,"flatnav",flatnav.__version__)'

rm -f "/tmp/measurement/$EXP.ready"
( cd "$REPO/experiments" && numactl --cpunodebind=0 --membind=0 \
   "$VENV" sift_big_flatnav_recall_batched.py \
     --dataset "$DATASET" --queries "$Q" --gtruth "$GT" --existing-mtx "$MTX" \
     --metric l2 --num-node-links 32 --ef-search $EFL \
     --num-search-threads "$THREADS" --search-batch-size "$SBS" \
     --search-only --exp-id "$EXP" --output-json "$OUT/recall.json" ) > "$OUT/runner.log" 2>&1 &
PYPID=$!
echo "runner PID=$PYPID exp=$EXP out=$OUT — building graph (~14 min)..."

while [ ! -f "/tmp/measurement/$EXP.ready" ] && kill -0 "$PYPID" 2>/dev/null; do sleep 3; done
kill -0 "$PYPID" 2>/dev/null || { echo BUILD_FAILED; tail -25 "$OUT/runner.log"; exit 1; }
echo "build done $(date +%T) — measuring search phase..."

IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"
sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$OUT/perf.csv" -- \
  bash -c "while kill -0 $PYPID 2>/dev/null; do sleep 0.3; done" 2>/dev/null

echo "===== POETRY RUNNER (numpy1.26, cp310 wheel, node-0, ${THREADS}T, ef=$EFL, batch=$SBS) =====" | tee "$OUT/summary.txt"
awk -F, -v PEAK="$PEAK" '
  $1+0>0 && $4!="" { ev=$4; v=$2;
    if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
    else if(ev=="cycles") cyc+=v; else if(ev=="instructions") ins+=v;
    else if(ev=="l1d_pend_miss.pending") p+=v; else if(ev=="l1d_pend_miss.pending_cycles") pc+=v;
    if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
  END{ w=hi-lo; printf "search window=%.0fs  BW=%.1f GB/s  util=%.0f%%  IPC=%.3f  mem_stall=%.1f%%  MLP=%.2f\n",
        w, cas/w/1e9, 100*(cas/w/1e9)/PEAK, ins/cyc, 100*pc/cyc, p/pc }' "$OUT/perf.csv" | tee -a "$OUT/summary.txt"

"$VENV" - "$OUT/recall.json" <<'PY' | tee -a "$OUT/summary.txt"
import json,sys
d=json.load(open(sys.argv[1]))
for ef,v in d["results"].items():
    print(f"  ef={ef}  recall@{d['k']}={v['recall']:.4f}  qps={v['qps']:.0f}")
PY
echo "raw: $OUT" | tee -a "$OUT/summary.txt"
