# SIFT-100M Benchmark Profiler - Quick Start Guide

## What's Been Set Up

✅ **New Python Script**: `sift_benchmark_profiler.py`
- Builds FlatNav index on SIFT-100M dataset
- Runs queries at multiple ef-search values
- Tracks timing, memory, and recall@100
- Outputs formatted console summary + optional JSON report

✅ **Three New Make Targets** in Makefile:
- `sift-benchmark-profile-local` - Local testing with `/mydata/flatnav/data`
- `sift-benchmark-profile-docker` - Docker production with `/root/data`
- `sift-benchmark-profile-extended` - Extended ef-search range [50, 100, 200, 500, 1000]

✅ **Documentation** in README.md:
- Local benchmarking instructions
- Docker benchmarking instructions
- Bandwidth profiling workflow with pcm-memory
- Output metrics explanation

## Quick Start

### Option 1: Local Testing (No Docker)

Run the benchmark locally on your machine:

```bash
cd /mydata/flatnav/experiments
./setup-poetry-env.sh  # One-time setup
make sift-benchmark-profile-local
```

Output will show:
```
Build Time:       45.23s
Peak Memory:      2145.3 MB
Search Performance (EF-Search, Time, Recall@100):
  EF=100  |     1.23ms  |   95.45%  |   814 q/s
  EF=200  |     1.89ms  |   98.23%  |   529 q/s
```

### Option 2: Docker (Full Environment)

Run in Docker matching the Dockerfile specification:

```bash
cd /mydata/flatnav
./bin/docker-run.sh sift-benchmark-profile-docker
```

### Option 3: Extended Testing (Comprehensive EF Range)

Test across more ef-search values for tradeoff analysis:

```bash
cd /mydata/flatnav
./bin/docker-run.sh sift-benchmark-profile-extended
```

## Bandwidth Profiling Workflow

To measure memory bandwidth while benchmark runs:

```bash
# Terminal 1: Start monitoring (needs sudo/root)
sudo pcm-memory 0.5 -csv=system_bandwidth.csv

# Terminal 2: Run benchmark in Docker (in another terminal/tmux)
cd /mydata/flatnav
./bin/docker-run.sh sift-benchmark-profile-docker

# Results combine:
# - system_bandwidth.csv: bandwidth traces
# - /root/metrics/sift-benchmark.json: recall/timing metrics
```

## Output Files

- **Console Output**: Human-readable table with build time, memory, latency, recall, throughput
- **JSON Report** (if --output specified): 
  ```json
  {
    "dataset_name": "sift-100m",
    "num_vectors": 100000000,
    "dimension": 128,
    "build_time_sec": 45.23,
    "peak_memory_mb": 2145.3,
    "search_results": {
      "100": {"time_ms": 1.23, "recall": 0.9545},
      "200": {"time_ms": 1.89, "recall": 0.9823}
    }
  }
  ```

## Customization

Run with custom parameters:

```bash
cd /mydata/flatnav/experiments
poetry run python sift_benchmark_profiler.py --local \
    --num-node-links 32 \
    --ef-construction 200 \
    --ef-search-values 50,100,200,400,800 \
    --metric l2 \
    --k 100 \
    --output /tmp/custom-benchmark.json
```

## Troubleshooting

**Issue**: Script can't find data files
- **Local**: Ensure SIFT data exists at `/mydata/flatnav/data/sift-128-euclidean/`
- **Docker**: Ensure data mounted at `/root/data/sift-128-euclidean/` in container

**Issue**: OOM (out of memory) during build
- Reduce vector count by using a subset (SIFT-100M is very large)
- Increase swap or available RAM

**Issue**: Import errors
- Ensure Poetry environment is activated: `cd experiments && source .venv/bin/activate`
- Or use: `poetry run python sift_benchmark_profiler.py --local`

## Files Modified/Created

1. **Created**: `/mydata/flatnav/experiments/sift_benchmark_profiler.py` (new script)
2. **Updated**: `/mydata/flatnav/experiments/Makefile` (3 new targets)
3. **Updated**: `/mydata/flatnav/experiments/README.md` (documentation)
4. **Updated**: `/mydata/flatnav/experiments/setup-poetry-env.sh` (made executable)

## Reference

- Script location: `/mydata/flatnav/experiments/sift_benchmark_profiler.py`
- Make targets: See `/mydata/flatnav/experiments/Makefile` (lines ~710-750)
- Documentation: `/mydata/flatnav/experiments/README.md` (SIFT-100M section)

## SIFT100M Extra Query Ground Truth

For exact FAISS ground-truth generation of `sift100m_200k_extra_query.fvecs`, including
resource-optimized commands (75% and max CPU), NUMA guidance, and validation interpretation,
see:

- `/mydata/flatnav/experiments/README.md` under **Generating Exact Ground Truth for SIFT100M Extra Queries**
