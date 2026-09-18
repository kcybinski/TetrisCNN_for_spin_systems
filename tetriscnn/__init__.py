import os
import sys
from pathlib import Path

# environment.yml installs torch via pip (see its comment there for why) alongside
# conda-forge's numpy/scipy, which link their own libomp.dylib on macOS. pip's torch
# wheel bundles a second copy, and loading both aborts the process with "OMP: Error
# #15: Initializing libomp.dylib, but found libomp.dylib already initialized." This
# reproduces on a from-scratch `conda env create -f environment.yml` regardless of
# which torch version in the `>=2.2,<3` range pip resolves (verified against both
# 2.9.1, the version environment.yml calls "the reference environment", and 2.14.0).
#
# KMP_DUPLICATE_LIB_OK=TRUE alone silences the abort but does NOT make this safe: the
# two libomp copies are separate thread pools, and letting both run actual parallel
# regions concurrently (e.g. a conv2d backward through an AdamW step) reproducibly
# segfaults with "OMP: Error #179: Function pthread_mutex_init failed" instead
# (verified with tests/test_equivariant.py's masked-training test). Forcing every
# OpenMP-using library in the process onto a single thread removes the concurrent
# access that corrupts the mutex, which is what actually makes coexistence safe.
# This is a real, if modest, throughput cost for CPU-bound work (numpy/scipy/torch
# lose intra-op parallelism), so it is scoped to macOS, where the duplicate-libomp
# situation actually arises; Linux/CUDA installs are untouched.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

#: Path to the manuscript's shared matplotlib style, packaged alongside the library
#: so the figure notebooks can `plt.style.use(PAPER_MPLSTYLE)` without depending on
#: the repo root. See docs/NOTEBOOKS.md for what it controls.
PAPER_MPLSTYLE = str(Path(__file__).resolve().parent / "paper.mplstyle")
