#!/bin/bash
# sift_setup.sh

# 1. Download Query Vectors (10,000 vectors)
if [ -f "bigann_query.bvecs" ]; then
    echo "bigann_query.bvecs already exists. Skipping download."
else
    echo "Downloading query vectors..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
    gunzip -f bigann_query.bvecs.gz
fi

# 2. Download Base Vectors (Streaming first 100M vectors)
# Each vector is 132 bytes (4 bytes for dimension + 128 bytes of data).
# Total: 100,000,000 * 132 = 13,200,000,000 bytes.
if [ -f "sift100m_base.bvecs" ]; then
    echo "sift100m_base.bvecs already exists. Skipping download."
else
    echo "Streaming first 100M vectors from SIFT-1B..."
    curl -s ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz | \
    gunzip -c | head -c 13200000000 > sift100m_base.bvecs
fi

# 3. Download Ground Truth
if [ -f "bigann_gnd_100M.ivecs" ]; then
    echo "bigann_gnd_100M.ivecs already exists. Skipping download."
else
    echo "Downloading ground truth vectors..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
    tar -xzf bigann_gnd.tar.gz
    # Extract idx_100M.ivecs from the gnd directory and move to current directory
    if [ -d "gnd" ] && [ -f "gnd/idx_100M.ivecs" ]; then
        mv gnd/idx_100M.ivecs bigann_gnd_100M.ivecs
        rm -rf gnd
    fi
    rm -f bigann_gnd.tar.gz
    echo "Ground truth extracted to bigann_gnd_100M.ivecs"
fi

echo "Done. All datasets ready in current directory."