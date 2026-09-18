"""Shared pytest configuration for the TetrisCNN characterization suite.

These tests are a behavior-preservation safety net for an upcoming refactor: they
capture *current* behavior (including oddities) rather than intended behavior. Where
current behavior looks wrong, it is still asserted on (with a comment/marker calling
it out) rather than silently "fixed" by the test.
"""
import os
import sys
from pathlib import Path

# See tetriscnn/__init__.py for why these are set on macOS (conda-forge numpy/scipy +
# pip-installed torch both link a libomp.dylib; letting both run threaded regions
# concurrently segfaults, not just warns). Set here too, not just there: pytest imports
# conftest.py before any test module, but a test module that does a bare `import torch`
# before importing anything from `tetriscnn` (test_equivariant.py does) would otherwise
# race the package's own fix when that file is run in isolation rather than as part of
# the full suite (where an earlier-collected module happens to import tetriscnn first).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# Several library functions (e.g. tetriscnn.dataprocessing.get_data_root) resolve
# paths relative to the process's current working directory, not this file's location.
# Pin cwd to the repo root so the suite behaves the same regardless of where `pytest`
# was invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _resolve_data_root() -> Path:
    """The dataset tree the library itself will read, so skips agree with it.

    Mirrors tetriscnn.dataprocessing.get_data_root(): TETRISCNN_DATA_ROOT if set, then
    the release layout's datasets/, then the legacy data/. Deliberately not imported
    from the library, so a broken import there cannot silently turn every data test
    into a skip. Whichever exists is returned; a fresh clone has only the tracked
    datasets/README.md and manifest, so the data checks below still report absent.
    """
    override = os.environ.get("TETRISCNN_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for name in ("datasets", "data"):
        candidate = REPO_ROOT / name
        if (candidate / "experimental").is_dir() or (candidate / "simulated").is_dir():
            return candidate
    return REPO_ROOT / "datasets"


DATA_ROOT = _resolve_data_root()


def data_available() -> bool:
    """True if the experimental snapshots (not in a fresh clone) are present."""
    return (DATA_ROOT / "experimental" / "Ising").exists()
