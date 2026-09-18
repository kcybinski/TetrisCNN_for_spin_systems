"""Make TeX binaries reachable regardless of how the Python process was started.

Why this exists
---------------
MacTeX installs its binaries in ``/Library/TeX/texbin`` and puts that directory on
the PATH via ``/etc/paths.d/TeX``.  That file is read by ``path_helper``, which only
runs for *login shells*.  A process started by launchd -- VS Code / Jupyter opened
from the Dock or Spotlight, a GUI-launched notebook kernel -- inherits the bare
launchd PATH and never sees ``/Library/TeX/texbin``.

Matplotlib then cannot run ``kpsewhich``/``luatex`` to resolve font metrics and
reports the confusing::

    FileNotFoundError: Matplotlib's TeX implementation searched for a file named
    'cmss8.tfm' in your texmf tree, but could not find it

It complains about a *font* rather than a missing ``latex`` because the LaTeX run
itself is usually served from ``~/.matplotlib/tex.cache``; the first thing that
actually needs a TeX binary is the .tfm lookup during DVI parsing.  Whether it
breaks therefore depends on how the editor happened to be launched that day.

Import this module (or call ``ensure_tex_on_path()``) before enabling
``text.usetex``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Common install locations, most-specific first.  ``/Library/TeX/texbin`` is the
# MacTeX-managed symlink to whichever distribution is currently selected.
CANDIDATE_TEX_DIRS = (
    "/Library/TeX/texbin",
    "/usr/local/texlive/bin/universal-darwin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
)


def _tex_is_reachable() -> bool:
    """True if the binaries matplotlib's usetex path shells out to are on PATH."""
    return all(shutil.which(exe) for exe in ("latex", "kpsewhich", "dvipng"))


def ensure_tex_on_path(verbose: bool = False) -> bool:
    """Prepend a working TeX bin directory to ``os.environ['PATH']`` if needed.

    Returns True if TeX is reachable afterwards.  Mutating ``os.environ`` is
    enough: matplotlib launches ``latex``/``kpsewhich`` as subprocesses, which
    inherit this process's environment.
    """
    if _tex_is_reachable():
        return True

    for candidate in CANDIDATE_TEX_DIRS:
        if (Path(candidate) / "kpsewhich").exists():
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            if _tex_is_reachable():
                if verbose:
                    print(f"[texpath] added {candidate} to PATH")
                return True

    return False
