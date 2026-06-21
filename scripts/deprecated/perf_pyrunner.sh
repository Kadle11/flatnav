#!/usr/bin/env bash
# Profile the DEFAULT python runner (experiments/sift_big_flatnav_recall_batched.py)
# instead of our custom driver. Uses the runner's built-in --search-only ready
# signal (/tmp/measurement/<exp_id>.ready, written after the graph build) to gate
# our perf IMC+core measurement to the search phase only. Node-0 pinned.
#
# NOTE: the default runner REBUILDS the index each run (allocate_nodes + parse the
# 40 GB mtx, ~14 min) — it does not load a saved .bin. Per-batch it also computes
# recall inline in Python, so this captures the real runner overhead.
set -u
REPO=$HOME/flatnav
DATASET=/data/index/sift100m_base.fvecs
Q=/data/queries/sift100m_200k_extra_query.fvecs
GT=/data/queries/sift100m_200k_extra_query.gtruth.ivecs
MTX=/data/index/sift100m_m32_hnsw_base_layer.mtx
THREADS=${THREADS:-32}
EFL="${EFL:-200 200 200 200}"          # repeat ef so the search window is ~50s
SBS=${SBS:-128}                         # runner default search batch size
PEAK=${PEAK:-102.8}
EXP=pyrun_$(date +%H%M%S)
OUT=$HOME/pyrunner_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

# 1. deps + flatnav package importable with the FIXED _core
python3 -m pip install --user -q hnswlib 2>&1 | tail -1
EXT=$(python3-config --extension-suffix)
rm -rf "$HOME/pypkg"; mkdir -p "$HOME/pypkg/flatnav"
cp "$REPO/python-bindings/src/flatnav/__init__.py" "$HOME/pypkg/flatnav/"
cp "$HOME/_core$EXT" "$HOME/pypkg/flatnav/_core$EXT"
export PYTHONPATH="$HOME/pypkg:$REPO/experiments"
python3 -c "import flatnav, hnswlib, numpy; from data_loader import get_data_loader; print('imports OK', flatnav.__version__)" || { echo IMPORT_FAIL; exit 1; }

# 2. launch the default runner (search-only) on node 0
rm -f "/tmp/measurement/$EXP.ready"
( cd "$REPO/experiments" && numactl --cpunodebind=0 --membind=0 \
   python3 sift_big_flatnav_recall_batched.py \
     --dataset "$DATASET" --queries "$Q" --gtruth "$GT" --existing-mtx "$MTX" \
     --metric l2 --num-node-links 32 --ef-search $EFL \
     --num-search-threads "$THREADS" --search-batch-size "$SBS" \
     --search-only --exp-id "$EXP" --output-json "$OUT/recall.json" ) > "$OUT/runner.log" 2>&1 &
PYPID=$!
echo "runner PID=$PYPID exp=$EXP — waiting for graph build (~14 min)..."

# 3. wait for the ready signal (build complete)
while [ ! -f "/tmp/measurement/$EXP.ready" ] && kill -0 "$PYPID" 2>/dev/null; do sleep 2; done
kill -0 "$PYPID" 2>/dev/null || { echo "runner exited during build:"; tail -25 "$OUT/runner.log"; exit 1; }
echo "build done — measuring search phase..."

# 4. perf IMC + core, system-wide, only while the runner is alive (= search phase)
IMC=""; for i in 0 1 2 3 4 5; do IMC="$IMC,uncore_imc_$i/cas_count_read/,uncore_imc_$i/cas_count_write/"; done; IMC=${IMC#,}
CORE="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"
sudo perf stat -a -I 1000 -x , -e "$IMC,$CORE" -o "$OUT/perf.csv" -- \
  bash -c "while kill -0 $PYPID 2>/dev/null; do sleep 0.3; done" 2>/dev/null

# 5. results
echo "===== DEFAULT PYTHON RUNNER (node-0, ${THREADS}T, ef=$EFL, batch=$SBS) ====="
awk -F, -v PEAK="$PEAK" '
  $1+0>0 && $4!="" { ev=$4; v=$2;
    if(ev ~ /cas_count_(read|write)/){ b=($3=="MiB")?v*1048576:v*64; cas+=b }
    else if(ev=="cycles") cyc+=v; else if(ev=="instructions") ins+=v;
    else if(ev=="l1d_pend_miss.pending") p+=v; else if(ev=="l1d_pend_miss.pending_cycles") pc+=v;
    if($1+0>hi)hi=$1+0; if(lo==0||$1+0<lo)lo=$1+0 }
  END{ w=hi-lo; printf "search window=%.0fs  BW=%.1f GB/s  util=%.0f%%  IPC=%.3f  mem_stall=%.1f%%  MLP=%.2f\n",
        w, cas/w/1e9, 100*(cas/w/1e9)/PEAK, ins/cyc, 100*pc/cyc, p/pc }' "$OUT/perf.csv"
python3 - "$OUT/recall.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for ef,v in d["results"].items():
    print(f"  ef={ef}  recall@{d['k']}={v['recall']:.4f}  qps={v['qps']:.0f}  p99_ms={v['p99_time_ms']:.3f}")
PY
echo "raw: $OUT"
