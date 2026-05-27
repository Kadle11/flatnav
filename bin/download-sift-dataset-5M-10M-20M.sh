#!/bin/bash
# sift_setup_small.sh

set -e

QUERY_FILE="bigann_query.bvecs"
GND_TAR="bigann_gnd.tar.gz"
BASE_URL="ftp://ftp.irisa.fr/local/texmex/corpus"
GROUND_TRUTH_READY="0"
DATASET_PREFIX="sift-128-euclidean"

download_query_vectors() {
    if [ -f "$QUERY_FILE" ]; then
        echo "$QUERY_FILE already exists. Skipping download."
    else
        echo "Downloading query vectors..."
        wget -c "$BASE_URL/$QUERY_FILE.gz"
        gunzip -f "$QUERY_FILE.gz"
    fi
}

download_base_vectors() {
    local dataset_size="$1"
    local dataset_dir="${DATASET_PREFIX}-${dataset_size}M"
    local output_file="${dataset_dir}/sift${dataset_size}M_base.bvecs"
    local bytes_per_vector=132
    local total_bytes=$((dataset_size * 1000000 * bytes_per_vector))

    mkdir -p "$dataset_dir"

    if [ -f "$output_file" ]; then
        echo "$output_file already exists. Skipping download."
    else
        echo "Streaming first ${dataset_size}M vectors from SIFT-1B..."
        curl -s "$BASE_URL/bigann_base.bvecs.gz" | \
            gunzip -c | head -c "$total_bytes" > "$output_file"
    fi
}

download_ground_truth_archive() {
    if [ "$GROUND_TRUTH_READY" = "1" ] && [ -d "gnd" ]; then
        return
    fi

    if [ -d "gnd" ]; then
        GROUND_TRUTH_READY="1"
        return
    fi

    if [ ! -f "$GND_TAR" ]; then
        echo "Downloading ground truth vectors..."
        wget -c "$BASE_URL/$GND_TAR"
    fi

    tar -xzf "$GND_TAR"
    GROUND_TRUTH_READY="1"
}

download_ground_truth() {
    local dataset_size="$1"
    local dataset_dir="${DATASET_PREFIX}-${dataset_size}M"
    local output_file="${dataset_dir}/bigann_gnd_${dataset_size}M.ivecs"
    local expected_file="gnd/idx_${dataset_size}M.ivecs"

    mkdir -p "$dataset_dir"

    if [ -f "$output_file" ]; then
        echo "$output_file already exists. Skipping download."
        return
    fi

    download_ground_truth_archive

    if [ -f "$expected_file" ]; then
        mv "$expected_file" "$output_file"
        echo "Ground truth extracted to $output_file"
    else
        echo "Error: expected ground truth file $expected_file was not found after extracting $GND_TAR" >&2
        exit 1
    fi
}

copy_query_vectors() {
    local dataset_size="$1"
    local dataset_dir="${DATASET_PREFIX}-${dataset_size}M"
    local output_file="${dataset_dir}/bigann_query.bvecs"

    mkdir -p "$dataset_dir"

    if [ -f "$output_file" ]; then
        echo "$output_file already exists. Skipping copy."
    else
        cp "$QUERY_FILE" "$output_file"
    fi
}

download_query_vectors

for dataset_size in 1 5 10 20; do
    download_base_vectors "$dataset_size"
    copy_query_vectors "$dataset_size"
    download_ground_truth "$dataset_size"
done

if [ -d "gnd" ]; then
    rm -rf gnd
fi

if [ -f "$GND_TAR" ]; then
    rm -f "$GND_TAR"
fi

echo "Done. All datasets ready in ${DATASET_PREFIX}-1M/${DATASET_PREFIX}-5M/${DATASET_PREFIX}-10M/${DATASET_PREFIX}-20M."