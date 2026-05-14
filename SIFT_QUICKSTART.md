# FlatNav SIFT-100M Quick Start: Setup, Build & Benchmark

Complete workflow for setting up FlatNav, downloading SIFT-100M, building the index, and benchmarking with memory bandwidth profiling.

## Prerequisites

- Linux machine with ~2TB disk space and sufficient RAM (1TB+ recommended for SIFT-100M)
- `pcm-memory` tool (Intel Performance Counter Monitor) for bandwidth profiling
- `sudo` access for system package installation and bandwidth monitoring

## Step 1: Setup the Library

Run the comprehensive setup script that handles dependencies, CMake, Python, Poetry, and builds both FlatNav and hnswlib wheels:

```bash
cd /mydata/flatnav
bash setup_local_env.sh
source ~/.bashrc
```

This will:
- Install system dependencies (build-essential, CMake, etc.)
- Install Python 3.11.6 via pyenv
- Install Poetry 1.8.2
- Build the FlatNav wheel
- Build the extended hnswlib wheel
- Setup Poetry environment in `experiments/`

## Step 2: Download SIFT-100M Dataset

Download query vectors (10K) and base vectors (first 100M from SIFT-1B):

```bash
cd /mydata/flatnav/data
bash ../bin/download-sift-dataset.sh
```

This creates files in `/mydata/flatnav/data/`:
- `bigann_query.bvecs` (~1.3 MB, 10K vectors)
- `sift100m_base.bvecs` (~13 GB, 100M vectors in uint8 format)
- `bigann_gnd_100M.ivecs` (ground truth nearest neighbors for 100M base vectors, auto-downloaded if not present)

## Step 3: Convert bvecs to fvecs (float32)

Convert uint8 binary format to float32 for use with FlatNav:

```bash
cd /mydata/flatnav/data
bash ../bin/convert_bvecs_to_fvecs.sh
```

By default, the current script converts `extra_queries_200k.bvecs` to `sift100m_200k_extra_query.fvecs`.

If you also need the standard SIFT-100M base/query `.fvecs` files used in the commands below, open `bin/convert_bvecs_to_fvecs.sh` and uncomment these two lines before running:

```python
#convert_bvecs_to_fvecs('bigann_query.bvecs', 'sift100m_query.fvecs')
#convert_bvecs_to_fvecs('sift100m_base.bvecs', 'sift100m_base.fvecs')
```

**Note**: Conversion may take hours depending on disk I/O. The script uses memory-mapping to handle large files on systems with limited RAM.

## Step 4: Generate Ground Truth (Optional - Pre-downloaded Ground Truth Available)

Ground truth file `bigann_gnd.ivecs` is automatically downloaded in Step 2. You can use it directly for recall validation. Alternatively, generate your own exact nearest neighbors using FAISS for independent validation:

```bash
cd /mydata/flatnav/experiments

# 75% CPU utilization (30 of 40 logical cores)
OMP_NUM_THREADS=30 numactl --interleave=all --physcpubind=0-29 \
poetry run python ../tools/generate_faiss_ground_truth.py \
    --base-fvecs ../data/sift100m_base.fvecs \
    --queries-fvecs ../data/sift100m_query.fvecs \
    --k 100 \
    --output-ivecs ../data/sift100m_gtruth_faiss.ivecs \
    --base-batch-size 400000 \
    --query-batch-size 10000 \
    --threads 30 \
    --validation-seed 42
```

This generates `sift100m_gtruth_faiss.ivecs` with independently computed top-100 neighbors for comparison with the pre-downloaded ground truth.

  Add `--validate-after-run` if you want the generator to run a post-check of the produced ground truth.

## Step 4b: Use External Extra Queries + Ground Truth (Optional)

If you already have extra SIFT queries and corresponding ground truth at:

`/proj/prismgt-PG0/vrao79/search_step_traces/sift-queries`

you can skip local query conversion/ground-truth generation for those files and point the recall script directly to them.

Example (adjust filenames to match what exists in that folder):

```bash
cd /mydata/flatnav/experiments

EXTRA_DIR=/proj/prismgt-PG0/vrao79/search_step_traces/sift-queries

poetry run python sift_big_flatnav_recall.py \
  --dataset ../data/sift100m_base.fvecs \
  --queries "$EXTRA_DIR/sift100m_200k_extra_query.fvecs" \
  --gtruth "$EXTRA_DIR/sift100m_200k_extra_query_gtruth.ivecs" \
  --metric l2 \
  --num-node-links 32 \
  --ef-construction 200 \
  --ef-search 100 200 500 1000 \
  --num-build-threads 1 \
  --num-search-threads 30 \
  --save-mtx ../data/sift100m_hnsw_base_layer.mtx \
  --output-json ../data/sift100m_extra_queries_recall.json
```

Optional FAISS consistency check on these external queries:

```bash
cd /mydata/flatnav/experiments

EXTRA_DIR=/proj/prismgt-PG0/vrao79/search_step_traces/sift-queries

poetry run python sift_big_flatnav_recall.py \
  --dataset ../data/sift100m_base.fvecs \
  --queries "$EXTRA_DIR/sift100m_200k_extra_query.fvecs" \
  --gtruth "$EXTRA_DIR/sift100m_200k_extra_query_gtruth.ivecs" \
  --metric l2 \
  --num-node-links 32 \
  --ef-construction 200 \
  --ef-search 100 200 500 1000 \
  --num-build-threads 1 \
  --num-search-threads 30 \
  --validate-faiss-flatl2 \
  --faiss-validate-queries 1000 \
  --faiss-num-threads 30 \
  --save-mtx ../data/sift100m_hnsw_base_layer.mtx \
  --output-json ../data/sift100m_extra_queries_recall_with_faiss.json
```

## Step 5: Construction + Recall Script (Use for Graph Build and Recall Runs)

Use `sift_big_flatnav_recall.py` when your goal is to construct the graph and run recall-oriented query sweeps (with optional FAISS consistency validation).

### 5a: Install pcm-memory (One-time)

```bash
# Install pcm (Intel Performance Counter Monitor)
sudo apt install pcm
```

### 5b: Run Build + Query (No FAISS Validation) with Bandwidth Profiling

**Terminal 1** - Start bandwidth monitoring:

```bash
# Monitor every 0.5 seconds, output to CSV
sudo pcm-memory 0.5 -csv=sift100m_build_query_bandwidth.csv
```
There is a possibility that you might need to do `sudo modprobe msr` before running this.

**Terminal 2** - Build graph + run recall queries:

```bash
cd /mydata/flatnav/experiments

poetry run python sift_big_flatnav_recall.py \
  --dataset ../data/sift100m_base.fvecs \
  --queries ../data/sift100m_query.fvecs \
  --gtruth ../data/bigann_gnd_100M.ivecs \
  --metric l2 \
  --num-node-links 32 \
  --ef-construction 200 \
  --ef-search 100 200 500 1000 \
  --num-build-threads 1 \
  --num-search-threads 30 \
  --build-batch-size 250000 \
  --save-mtx ../data/sift100m_hnsw_base_layer.mtx \
  --output-json ../data/sift100m_recall_no_faiss.json
```
Build time of the graph can take a few hours so saving it the first time helps reducing the benchmarking time overall
When complete, press `Ctrl+C` in Terminal 1 to stop pcm-memory. The CSV captures bandwidth during both graph construction and query search phases.

## Step 6: Run With FAISS Validation Enabled

This mode re-checks your provided ground truth against exact FAISS FlatL2 neighbors before running FlatNav recall.

```bash
cd /mydata/flatnav/experiments

poetry run python sift_big_flatnav_recall.py \
  --dataset ../data/sift100m_base.fvecs \
  --queries ../data/sift100m_query.fvecs \
  --gtruth ../data/bigann_gnd_100M.ivecs \
  --metric l2 \
  --num-node-links 32 \
  --ef-construction 200 \
  --ef-search 100 200 500 1000 \
  --num-build-threads 1 \
  --num-search-threads 30 \
  --build-batch-size 250000 \
  --validate-faiss-flatl2 \
  --faiss-validate-queries 1000 \
  --faiss-num-threads 30 \
  --save-mtx ../data/sift100m_hnsw_base_layer.mtx \
  --output-json ../data/sift100m_recall_with_faiss.json
```

If you already saved an `.mtx` graph in a previous run, skip HNSW rebuild to speed up repeated query experiments:

```bash
cd /mydata/flatnav/experiments

poetry run python sift_big_flatnav_recall.py \
  --dataset ../data/sift100m_base.fvecs \
  --queries ../data/sift100m_query.fvecs \
  --gtruth ../data/bigann_gnd_100M.ivecs \
  --metric l2 \
  --num-node-links 32 \
  --ef-search 100 200 500 1000 \
  --num-build-threads 1 \
  --num-search-threads 30 \
  --existing-mtx ../data/sift100m_hnsw_base_layer.mtx \
  --output-json ../data/sift100m_recall_reuse_mtx.json
```

## Step 7: Analyze Bandwidth Results

After running pcm-memory, analyze the CSV output:

```bash
cd /mydata/flatnav/data

# View build bandwidth stats
echo "=== Build Phase Bandwidth ==="
awk -F',' 'NR>1 {print $2}' sift100m_build_query_bandwidth.csv | \
  awk '{sum+=$1; count++} END {print "Avg: " sum/count " GB/s"; print "Count: " count}'

# Tip: build and query are in the same CSV when run via a single script command.
```

## Complete Workflow Summary

```bash
# 1. Setup
cd /mydata/flatnav
bash setup_local_env.sh
source ~/.bashrc

# 2. Download & Convert
cd data && bash ../bin/download-sift-dataset.sh
# Ensure base/query conversion lines are uncommented in bin/convert_bvecs_to_fvecs.sh
bash ../bin/convert_bvecs_to_fvecs.sh && cd ..

# 3. Generate ground truth (optional - pre-downloaded ground truth available)
# cd experiments
# poetry run python ../tools/generate_faiss_ground_truth.py \
#     --base-fvecs ../data/sift100m_base.fvecs \
#     --queries-fvecs ../data/sift100m_query.fvecs \
#   --k 100 --output-ivecs ../data/sift100m_gtruth_faiss.ivecs

# 4. In Terminal 1: start bandwidth monitor
cd experiments
sudo pcm-memory 0.5 -csv=/mydata/flatnav/data/sift100m_build_query_bandwidth.csv

# 5. In Terminal 2: run build + query + recall
poetry run python sift_big_flatnav_recall.py \
    --dataset ../data/sift100m_base.fvecs \
    --queries ../data/sift100m_query.fvecs \
    --gtruth ../data/bigann_gnd_100M.ivecs \
    --metric l2 --num-node-links 32 --ef-construction 200 \
    --ef-search 100 200 500 1000 \
    --num-build-threads 1 --num-search-threads 30 \
    --validate-faiss-flatl2 --faiss-validate-queries 1000 \
    --output-json ../data/sift100m_recall_summary.json
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory during build | Reduce vector count, lower build batch size, or run on a higher-memory host |
| pcm-memory not found | Run `sudo apt install intel-pcm` |
| Import errors in Poetry | Run `cd experiments && poetry install --no-root` |
| Ground truth validation fails | Check file paths, ensure queries < base vectors in ground truth |
| Slow conversion (bvecs→fvecs) | Normal for 13GB files; consider parallel processing or faster disk |

## Files Generated

| File | Size | Purpose |
|------|------|---------|
| `sift100m_base.bvecs` | 13 GB | Original uint8 base vectors |
| `bigann_query.bvecs` | ~1.3 MB | Original uint8 query vectors (10K) |
| `bigann_gnd_100M.ivecs` | 39 MB | Ground truth neighbors for 100M base (auto-downloaded) |
| `sift100m_base.fvecs` | 48 GB | Converted float32 base vectors |
| `sift100m_query.fvecs` | ~5 MB | Converted float32 query vectors (used by recall script) |
| `sift100m_200k_extra_query.fvecs` | Variable | Converted extra query vectors (if generated) |
| `sift100m_gtruth_faiss.ivecs` | 4 MB | FAISS-generated ground truth top-100 neighbors (optional) |
| `sift100m_hnsw_base_layer.mtx` | Variable | Persisted HNSW base-layer graph for reuse |
| `sift100m_recall_no_faiss.json` | Variable | Recall/latency summary without FAISS validation |
| `sift100m_recall_with_faiss.json` | Variable | Recall/latency summary with FAISS validation |
| `sift100m_extra_queries_recall.json` | Variable | Recall/latency summary for external extra queries |
| `sift100m_extra_queries_recall_with_faiss.json` | Variable | External extra queries summary with FAISS validation |
| `sift100m_build_query_bandwidth.csv` | Variable | Bandwidth samples during build+query run | 

## References

- [Setup & Benchmarking Guide](./experiments/SIFT_BENCHMARK_GUIDE.md)
- [Full Experiments README](./experiments/README.md)
- [Recall Driver Script](./experiments/sift_big_flatnav_recall.py)
- [Profiler Script](./experiments/sift_benchmark_profiler.py)
- [Ground Truth Generation](./tools/generate_faiss_ground_truth.py)
- [FlatNav C++ API](./docs/cpp_api.rst)
