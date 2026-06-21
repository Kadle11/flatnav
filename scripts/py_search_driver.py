#!/usr/bin/env python3
# Drives the flatnav Python bindings' batched search (the executeInParallel path)
# so we can profile it. Prints phase markers (load_done / warmup_done, elapsed s
# from process start) aligned with perf stat -I interval timestamps, plus per-rep
# QPS and final recall@K.
#
#   python3 py_search_driver.py <index.bin> <query.fvecs> <gtruth.ivecs> [K=10] [threads=32] [ef=80] [repeat=8]
import sys, time
import numpy as np
import _core

def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    dim = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, dim + 1)[:, 1:].view(np.float32))

def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    w = int(a[0])
    return np.ascontiguousarray(a.reshape(-1, w + 1)[:, 1:])

idx_path, q_path, gt_path = sys.argv[1], sys.argv[2], sys.argv[3]
K       = int(sys.argv[4]) if len(sys.argv) > 4 else 10
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 32
ef      = int(sys.argv[6]) if len(sys.argv) > 6 else 80
repeat  = int(sys.argv[7]) if len(sys.argv) > 7 else 8

t0 = time.monotonic()
queries = read_fvecs(q_path)
gt = read_ivecs(gt_path)
nq = queries.shape[0]
print(f"[load] queries n={nq} dim={queries.shape[1]} gt={gt.shape}", flush=True)

idx = _core.index.IndexL2Float.load_index(idx_path)
idx.set_num_threads(threads)
print(f"[phase] load_done elapsed_s={time.monotonic()-t0:.3f} threads={idx.num_threads}", flush=True)

dists, labels = idx.search(queries, K, ef)        # warmup; search returns (distances, labels)
print(f"[phase] warmup_done elapsed_s={time.monotonic()-t0:.3f}", flush=True)

for r in range(repeat):
    t = time.monotonic()
    dists, labels = idx.search(queries, K, ef)
    dt = time.monotonic() - t
    print(f"{ef} {dt:.4f} {nq/dt:.0f}", flush=True)

# recall@K (once, after the timed loop)
g = gt[:, :K]
hit = sum(len(set(g[i]).intersection(labels[i])) for i in range(nq))
print(f"recall@{K}={hit/(nq*K):.4f}", flush=True)
