#!/bin/bash
# sift_setup.sh

# 1. Download Query Vectors (10,000 vectors)
echo "Downloading query vectors..."
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip -f bigann_query.bvecs.gz

# 2. Download Base Vectors (Streaming first 100M vectors)
# Each vector is 132 bytes (4 bytes for dimension + 128 bytes of data).
# Total: 100,000,000 * 132 = 13,200,000,000 bytes.
echo "Streaming first 100M vectors from SIFT-1B..."
curl -s ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz | \
gunzip -c | head -c 13200000000 > sift100m_base.bvecs

echo "Done. Saved to sift100m_base.bvecs"