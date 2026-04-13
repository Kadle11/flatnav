# Reproduction of FlatNav vs. HNSWLib Benchmarking Experiments

## Overview
This document provides instructions for reproducing the experimental results comparing our non-hierarchical NSW implementation in FlatNav to the popular open source [HNSWLib](https://github.com/nmslib/hnswlib) library which utilizes a layered hierarchical graph. 

To enable relatively seamless reproducibility, we require the users to do the following:

* A machine with [docker](https://www.docker.com/) installed (and sufficient RAM to build and query indexes for a given workload). If you prefer to run the experiments without docker, we will add additional instructions shortly (though we highly recommend using the docker approach for complete consistency with our reported results)

* Download and preprocess the benchmark datasets into a certain folder named as `data` in the top-level directory of the `flatnav` repository. We provide more detailed instructions for this step in the next sections. 

* Executing the command `./bin/docker-run.sh <make-target>` from top-level directory of the flatnav repository. The `<make-target>` argument specifies the parameters of the benchmarking job to execute. We specify the make targets in the file [Makefile](/experiments/Makefile).

## Example Commands

Assuming you have docker installed and have prepared a benchmark dataset into the `data` directory, one can reproduce any one of our experiments specified in `experiments/Makefile`. For example, the following command will benchmark `flatnav` on the `gist` dataset. 

```shell
./bin/docker-run.sh gist-bench-flatnav
```

The analogous `hnswlib` benchmarking job on `gist` can be executed with a similar command. Again, the details of this make target are specified in the [Makefile](/experiments/Makefile). 

```shell
./bin/docker-run.sh gist-bench-hnsw
```

**NOTE:** We currently mount the data as a volume so that the experiment runner script has access 
to the dataset. What this means for you is that you have to place the dataset you want to use under the 
[data](/data/) subdirectory. Then, when you define a new target, specify the data path as `/root/data/<dataset-name>`. For instance, for the `sift-bench` we specify the dataset path like this:

```
sift-bench: 
	poetry run python run-benchmark.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename sift.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 \
		--ef-search 100 200 300 \
		--metric l2 
```

You may also want to save the experiment logs to a file on disk. You can do so by running 
```
> ./bin/docker-test.sh sift-bench > logs.txt 2>& 1
```

## Running Locally with Poetry (No Docker)

The Dockerfile sets up the experiments environment with Poetry and a forked `hnswlib`
wheel. You can follow the same flow locally:

```shell
# From repo root
cd experiments

# Use the same Poetry version as Dockerfile
python3 -m pip install --user "poetry==1.8.2"

# Install project dependencies into the Poetry environment
poetry install --no-root

# Build the hnswlib wheel from the same fork used in Dockerfile
cd ..
git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git
cd hnswlib-original/python_bindings
poetry run python setup.py bdist_wheel

# Install the built wheel into the experiments Poetry venv and finalize installs
cd ../../experiments
poetry run pip install --no-deps --force-reinstall ../hnswlib-original/python_bindings/dist/*.whl
poetry install --no-root
```

Then run a benchmark with Poetry:

```shell
cd experiments
poetry run python run-benchmark.py \
	--dataset-name mnist-local \
	--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
	--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
	--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
	--index-type flatnav \
	--num-node-links 32 \
	--ef-construction 100 \
	--ef-search 100 \
	--metric l2
```

You can also run the helper script to perform the full setup in one step:

```shell
cd experiments
./setup-poetry-env.sh
```

### Viewing Output Metrics

Once you have run a benchmarking job to completion, the experiment runner will save a set of plots under the `metrics` directory in the top level of the `flatnav` repo. These plots include, amongst others, the latency vs. recall tradeoff curves that we report in the paper. We also save the raw data used to generate these plots in the file `metrics/metrics.json`. 

## Input Data Format

Our experimental benchmark scripts require three input data arguments for
train vectors, query vectors, and ground truth. Supported train/query formats are:

- `.npy` (loaded as `float32`)
- `.fvecs` (loaded as `float32`)
- `.bvecs`
- `.fbin`
- `.u8bin`
- `.i8bin`

Ground truth is expected as `.npy` or `.ivecs` depending on dataset source.
For recall, we use top-100 neighbors by default.

* A `--train` file representing the vectors of each item in the data collection used to build the search index. This is expected to be a numpy array of dimension $N \times d$ where $N$ is the database size and $d$ is the vector dimension

* A `--queries` file representing the query vectors to use to search against the index. This file is expected to be a numpy array of dimension $Q \times d$ where $Q$ is the number of queries. 

* A `--gtruth` file consisting of the true $k$ nearest neighbors for each corresponding query vector. This file is expected to be an integer numpy array of dimension $Q \times k$ where $k$ is the number of near neighbors to return (we default to 100 in our experiments). Each element of this array is expected to be an integer in the range $[0, N-1]$ representing items in the index. 

Example using `.fvecs` with `.ivecs` ground truth:

```shell
cd experiments
poetry run python run-benchmark.py \
	--dataset-name sift-fvecs \
	--dataset /path/to/base.fvecs \
	--queries /path/to/query.fvecs \
	--gtruth /path/to/groundtruth.ivecs \
	--index-type flatnav \
	--num-node-links 32 \
	--ef-construction 100 \
	--ef-search 100 \
	--metric l2
```

## Preparing Datasets from ANN-Benchmarks
## SIFT-100M Benchmark Profiling with Timing & Recall

For detailed profiling of SIFT-100M including index build time, peak memory, query 
latency, and recall@100 metrics at multiple ef-search values, use the dedicated 
benchmark profiler script:

### Local Benchmarking

To run locally (using `/mydata/flatnav/data` paths):

```shell
cd experiments
poetry run python sift_benchmark_profiler.py --local \
	--num-node-links 32 \
	--ef-construction 100 \
	--ef-search-values 100,200 \
	--metric l2 \
	--output /tmp/sift-benchmark-local.json
```

Or via Make:

```shell
cd experiments
make sift-benchmark-profile-local
```

### Docker Benchmarking

To run inside Docker (using `/root/data` paths):

```shell
cd experiments
poetry run python sift_benchmark_profiler.py --docker \
	--num-node-links 32 \
	--ef-construction 100 \
	--ef-search-values 100,200 \
	--metric l2 \
	--output /root/metrics/sift-benchmark.json
```

Or via Make:

```shell
cd experiments
make sift-benchmark-profile-docker
```

### Bandwidth Profiling

To measure memory bandwidth alongside the benchmark, run `pcm-memory` in a separate 
terminal while the benchmark executes:

```bash
# Terminal 1: Start bandwidth monitor (requires root/sudo)
sudo pcm-memory 0.5 -csv=system_bandwidth.csv

# Terminal 2: Run benchmark (Docker)
cd experiments
sudo docker-compose run flatnav-test make sift-benchmark-profile-docker

# Or run directly in Docker container:
./bin/docker-run.sh sift-benchmark-profile-docker
```

Results include:
- **Build Time**: Index construction time in seconds
- **Peak Memory**: Maximum resident set size (VmPeak) in MB
- **Search Latency**: Query time per EF-search value (milliseconds)
- **Recall@100**: Percentage of true top-100 neighbors found at each EF-search value
- **Queries/Sec**: Throughput metric (queries per second)

### Extended EF-Search Range

For a more comprehensive profiling across ef-search values [50, 100, 200, 500, 1000]:

```shell
cd experiments
make sift-benchmark-profile-extended
```

[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) provide HDF5 files for a standard benchmark of near-neighbor datasets, queries and ground-truth results. Our experiment runner expects `.npy` files instead of HDF5 so we provide a helper script to download ANN-Benchmarks and prepare the necessary numpy files.

To generate an [ANNS benchmark datasets](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets), run the following script

```shell
./bin/download_ann_benchmarks_datasets.sh <dataset-name> [--normalize]
```

__IMPORTANT:__ For datasets that use the angular/cosine similarity, you will need to use `--normalize` option so that the distances are computed correctly. 

Available dataset names include:

```shell
_ mnist-784-euclidean
_ sift-128-euclidean
_ glove-25-angular
_ glove-50-angular
_ glove-100-angular
_ glove-200-angular
_ deep-image-96-angular
_ gist-960-euclidean
_ nytimes-256-angular
```

## Preparing Datasets from Big-ANN Benchmarks

[Big-ANN Benchmarks](https://big-ann-benchmarks.com/neurips21.html) Is a more recent set of ANN benchmark datasets focused on extremely large scales. Specifically, Big-ANN Benchmarks provides access to embedding datasets and ground truth near neighbor sets for 10M, 100M, and 1B vectors. In our benchmarks, we focus on the 10M and 100M datasets due to computational resource constraints. To reproduce our results in the paper, we provide a helper script to download the 10M and 100M Big-ANN Benchmark datasets and convert them into the numpy format that `flatnav` expects. 

To process a Big-ANN benchmark datasets, please run the following script. This script will download and process the 10M and 100M versions of the given dataset, including the ground truth file. Then, one can run a benchmarking job as described above (these Big-ANN dataset configurations are already specified in the Makefile)

```shell
./bin/download_bigann_datasets.sh <dataset-name>
```

The available dataset names include:

```shell
- bigann
- deep
- text2image
- msspacev
```
Note that the Microsoft SpaceV Dataset is no longer available via the public link on the [Big-ANN Benchmarks](https://big-ann-benchmarks.com/neurips21.html) website. We instead access this dataset through the [SPTAG](https://github.com/microsoft/SPTAG) GitHub repository.
