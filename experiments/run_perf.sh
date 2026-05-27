#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if experiment ID is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <experiment_id> [events] [--dataset <path>] [--queries <path>] [--gtruth <path>] [--existing-mtx <path>] [--num-node-links <n>] [--threads N] [--search-only]"
  echo "Example: $0 test_run_01 'cycles,instructions,LLC-loads' --dataset ../data/sift100m_base.fvecs --queries ../data/sift100m_200k_extra_query.fvecs --gtruth ../data/sift100m_200k_extra_query.gtruth.ivecs --existing-mtx ../data/sift100m_hnsw_base_layer.mtx --num-node-links 32 --threads 4 --search-only"
  exit 1
fi

EXP_ID=$1
EVENTS=${2:-"cycles,instructions,L1-dcache-load-misses,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,offcore_requests_outstanding.all_data_rd,LLC-loads,LLC-load-misses"}
NUM_SEARCH_THREADS=1
SEARCH_BATCH_SIZE=128
SEARCH_BATCH_ARG=""
DATASET="../data/sift100m_base.fvecs"
QUERIES="../data/sift100m_200k_extra_query.fvecs"
GTRUTH="../data/sift100m_200k_extra_query.gtruth.ivecs"
EXISTING_MTX="../data/sift100m_hnsw_base_layer.mtx"
NUM_NODE_LINKS=32
SEARCH_ONLY_FLAG=""
PCM_MEMORY_PID=""
PCM_MEMORY_LAUNCHER_PID=""
PCM_MEMORY_LOG=""
PCM_MEMORY_PID_FILE=""

shift 2
while [ "$#" -gt 0 ]; do
  case "$1" in
    --existing-mtx)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --existing-mtx requires a path argument"
        exit 1
      fi
      EXISTING_MTX=$2
      shift 2
      ;;
    --num-node-links)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --num-node-links requires a numeric argument"
        exit 1
      fi
      NUM_NODE_LINKS=$2
      shift 2
      ;;
    --threads|--num-search-threads)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --threads requires a numeric argument"
        exit 1
      fi
      NUM_SEARCH_THREADS=$2
      shift 2
      ;;
    --dataset)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --dataset requires a path argument"
        exit 1
      fi
      DATASET=$2
      shift 2
      ;;
    --search-batch-size)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --search-batch-size requires a numeric argument"
        exit 1
      fi
      SEARCH_BATCH_SIZE=$2
      shift 2
      ;;
    --queries)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --queries requires a path argument"
        exit 1
      fi
      QUERIES=$2
      shift 2
      ;;
    --gtruth)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --gtruth requires a path argument"
        exit 1
      fi
      GTRUTH=$2
      shift 2
      ;;
    --search-only)
      SEARCH_ONLY_FLAG="--search-only"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 <experiment_id> [events] [--dataset <path>] [--queries <path>] [--gtruth <path>] [--existing-mtx <path>] [--num-node-links <n>] [--search-only]"
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ "$NUM_SEARCH_THREADS" -gt 1 ]; then
  SEARCH_BATCH_ARG=(--search-batch-size "$SEARCH_BATCH_SIZE")
else
  SEARCH_BATCH_ARG=()
fi

DATASET_NAME=$(basename "$DATASET")
DATASET_NAME=${DATASET_NAME%.*}

# Calculate abbreviation: first 2 unique letters per event, no special characters, unique across events
ABBR=$(echo "$EVENTS" | tr ',' '\n' | head -n 4 | sed 's/[^a-zA-Z]//g' | awk '{
  s=""; 
  for(i=1;i<=length;i++){
    c=substr($0,i,1); 
    if(!index(tolower(s),tolower(c))) s=s c;
  }
  a=substr(s,1,2); 
  if(!seen[tolower(a)]++) printf "%s", a
}')

if [ -n "$SEARCH_ONLY_FLAG" ]; then
    ABBR="${ABBR}_search-only"
fi

# Create the results directory if it doesn't exist
OUT_DIR="../data/results/${DATASET_NAME}"
mkdir -p "$OUT_DIR"
RESULT_PREFIX="${DATASET_NAME}_${EXP_ID}_${ABBR}"
PCM_CSV="${OUT_DIR}/${RESULT_PREFIX}_pcm_memory.csv"
PCM_MEMORY_LOG="${OUT_DIR}/${RESULT_PREFIX}_pcm_memory.log"
WORKLOAD_PID=""

cleanup() {
  if [ -n "$PCM_MEMORY_PID" ] && kill -0 "$PCM_MEMORY_PID" 2>/dev/null; then
    sudo kill -TERM "$PCM_MEMORY_PID" 2>/dev/null || true
  fi
  if [ -n "$PCM_MEMORY_LAUNCHER_PID" ] && kill -0 "$PCM_MEMORY_LAUNCHER_PID" 2>/dev/null; then
    kill -TERM "$PCM_MEMORY_LAUNCHER_PID" 2>/dev/null || true
  fi
  if [ -n "$WORKLOAD_PID" ] && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
  fi
  if [ -n "$PCM_MEMORY_PID_FILE" ] && [ -f "$PCM_MEMORY_PID_FILE" ]; then
    rm -f "$PCM_MEMORY_PID_FILE"
  fi
}

trap cleanup EXIT INT TERM

echo "Starting experiment with EXP_ID: $EXP_ID, Events abbreviation: $ABBR"
echo "Using dataset: $DATASET"
echo "Using queries: $QUERIES"
echo "Using gtruth: $GTRUTH"
echo "Using existing MTX: $EXISTING_MTX"
echo "Using num-node-links: $NUM_NODE_LINKS"

if [ -n "$SEARCH_ONLY_FLAG" ]; then
    READY_FILE="/tmp/measurement/${EXP_ID}.ready"
    rm -f "$READY_FILE"

    echo "Launching benchmark in background and waiting for graph build..."
    # Choose script based on number of search threads: use batched script when threads>1
    if [ "$NUM_SEARCH_THREADS" -gt 1 ]; then
      SELECT_SCRIPT="sift_big_flatnav_recall_batched.py"
    else
      SELECT_SCRIPT="sift_big_flatnav_recall.py"
    fi

     numactl --cpunodebind=0 --membind=0,2 /users/sshivu3/.local/poetry/bin/poetry run python "$SELECT_SCRIPT" \
       --dataset "$DATASET" \
       --queries "$QUERIES" \
       --gtruth "$GTRUTH" \
       --existing-mtx "$EXISTING_MTX" \
       --metric l2 --num-node-links "$NUM_NODE_LINKS" \
       --ef-search 200 --num-search-threads "$NUM_SEARCH_THREADS" \
       "${SEARCH_BATCH_ARG[@]}" \
       --output-json "${OUT_DIR}/${RESULT_PREFIX}_recall.json" \
       --search-only --exp-id "$EXP_ID" &
    
    WORKLOAD_PID=$!
    
    # Poll every 1ms for the ready file
    while [ ! -f "$READY_FILE" ]; do
        sleep 0.001
    done

    echo "Graph built. Starting pcm-memory capture to: $PCM_CSV"
    PCM_MEMORY_PID_FILE=$(mktemp "/tmp/${EXP_ID}.pcm-memory.XXXXXX.pid")
    sudo bash -c "taskset -c 3 pcm-memory 0.5 -csv='$PCM_CSV' >'$PCM_MEMORY_LOG' 2>&1 & echo \$! > '$PCM_MEMORY_PID_FILE'; wait" &
    PCM_MEMORY_LAUNCHER_PID=$!
    while [ ! -s "$PCM_MEMORY_PID_FILE" ]; do
        sleep 0.05
    done
    PCM_MEMORY_PID=$(cat "$PCM_MEMORY_PID_FILE")
    sleep 0.05
    if ! sudo kill -0 "$PCM_MEMORY_PID" 2>/dev/null; then
      echo "Warning: pcm-memory process $PCM_MEMORY_PID not running. Dumping log: $PCM_MEMORY_LOG" >&2
      [ -f "$PCM_MEMORY_LOG" ] && sed -n '1,200p' "$PCM_MEMORY_LOG" >&2 || true
    fi
    
    echo "Graph built. Attaching perf stat to PID $WORKLOAD_PID..."
    taskset -c 1 perf stat -I 1000 -x , -o "${OUT_DIR}/${RESULT_PREFIX}_perf.csv" \
      -e "$EVENTS" \
      -p $WORKLOAD_PID

    if kill -0 "$WORKLOAD_PID" 2>/dev/null; then
        echo "Search phase complete. Stopping workload PID $WORKLOAD_PID..."
        kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    fi
    wait "$WORKLOAD_PID" 2>/dev/null || true
else
    # Choose script based on number of search threads: use batched script when threads>1
    if [ "$NUM_SEARCH_THREADS" -gt 1 ]; then
      SELECT_SCRIPT="sift_big_flatnav_recall_batched.py"
    else
      SELECT_SCRIPT="sift_big_flatnav_recall.py"
    fi

    taskset -c 1 perf stat -I 1000 -x , -o "${OUT_DIR}/${RESULT_PREFIX}_perf.csv" \
      -e "$EVENTS" \
      -- numactl --cpunodebind=0 --membind=0,2 /users/sshivu3/.local/poetry/bin/poetry run python "$SELECT_SCRIPT" \
         --dataset "$DATASET" \
         --queries "$QUERIES" \
         --gtruth "$GTRUTH" \
       --existing-mtx "$EXISTING_MTX" \
       --metric l2 --num-node-links "$NUM_NODE_LINKS" \
         --ef-search 200 --num-search-threads "$NUM_SEARCH_THREADS" \
         "${SEARCH_BATCH_ARG[@]}" \
         --output-json "${OUT_DIR}/${RESULT_PREFIX}_recall.json"
fi

cleanup

echo "Experiment complete! Results saved in ${OUT_DIR}/${RESULT_PREFIX}_perf.csv and ${OUT_DIR}/${RESULT_PREFIX}_recall.json"
echo "pcm-memory results saved in ${PCM_CSV}"

# Sync and drop caches
echo "Syncing and dropping pagecache..."
sync
sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'
