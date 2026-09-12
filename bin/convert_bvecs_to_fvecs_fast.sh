#!/usr/bin/env bash
# Convert a .bvecs file (uint8, 4-byte dim header + dim uint8) to .fvecs (float32), chunked via
# memmap. Faster than bin/convert_bvecs_to_fvecs.sh, which writes one vector per Python loop
# iteration (hours at 100M vectors); this does it in ~1M-vector blocks (minutes).
#
#   ./convert_bvecs_to_fvecs_fast.sh <in.bvecs> <out.fvecs> [dim=128] [chunk_vectors=1048576]
set -euo pipefail

IN=${1:?usage: $0 <in.bvecs> <out.fvecs> [dim=128] [chunk_vectors]}
OUT=${2:?usage: $0 <in.bvecs> <out.fvecs> [dim=128] [chunk_vectors]}
DIM=${3:-128}
CHUNK=${4:-1048576}

python3 - "$IN" "$OUT" "$DIM" "$CHUNK" <<'PY'
import numpy as np, os, sys

src, dst, dim, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rec = 4 + dim  # bvecs record: int32 dim header + dim uint8

n = os.path.getsize(src) // rec
print(f"[convert] {src} -> {dst} | dim={dim} n={n} chunk={chunk}")
m = np.memmap(src, dtype=np.uint8, mode='r', shape=(n, rec))
hdr = np.array([dim], dtype=np.int32).view(np.float32)[0]  # dim's bit pattern, written as float

with open(dst, 'wb') as f:
    for i in range(0, n, chunk):
        blk = m[i:i + chunk, 4:]
        out = np.empty((blk.shape[0], dim + 1), dtype=np.float32)
        out[:, 0] = hdr
        out[:, 1:] = blk
        out.tofile(f)
        print(f"\r[convert] {min(i + chunk, n)}/{n}", end="", flush=True)
print()
print(f"[convert] done: {os.path.getsize(dst)} bytes")
PY
