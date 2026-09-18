"""The optional symbolic-regression boundary.

PySR (and plotly, which only the SR plots use) is deliberately kept out of the main
install: `requirements.txt` omits both, and `tetriscnn/symbolic_regression.py` raises a
pointed ImportError when PySR is missing. That only holds up if nothing on the ordinary
route imports it. This file pins that boundary.

It exists because it was once broken in exactly the way it is cheap to break: moving the
SR plots into `symbolic_regression.py` also stranded the synthetic spin-configuration
generators there, and `Figure6.ipynb` imported three of them for its out-of-distribution
probes. Nothing about those generators needs PySR -- they are numpy draws -- but the
import chain meant a main-text figure notebook could no longer run without the optional
extra. The generators now live in `tetriscnn/synthetic_configs.py`; this test would have
caught the regression at the time, and catches the next one of its kind.

Only module-level imports are inspected. A `TYPE_CHECKING` annotation import (as
`tetriscnn/utils.py` uses for `SRConfig`) never executes, and prose mentioning a module
by name is not an import.
"""

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The SR route itself, which is allowed -- indeed required -- to need PySR.
SR_MODULES = {
    REPO / "tetriscnn" / "symbolic_regression.py",
    REPO / "tetriscnn" / "sr_experiments.py",
    REPO / "sr_toolbox.py",
}

OPTIONAL = ("pysr", "plotly", "kaleido", "tetriscnn.symbolic_regression")


def _module_level_imports(tree):
    """Names imported at module level, skipping `if TYPE_CHECKING:` bodies."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def _offending(names):
    return sorted(n for n in names if any(n == o or n.startswith(o + ".") for o in OPTIONAL))


def _library_files():
    files = sorted((REPO / "tetriscnn").glob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    files.append(REPO / "main.py")
    return [f for f in files if f not in SR_MODULES]


@pytest.mark.parametrize("path", _library_files(), ids=lambda p: p.name)
def test_library_module_does_not_import_the_optional_extra(path):
    offenders = _offending(_module_level_imports(ast.parse(path.read_text())))
    assert not offenders, (
        f"{path.relative_to(REPO)} imports {offenders} at module level, which makes the "
        "optional PySR/plotly install mandatory for it. Move what it needs into a module "
        "that does not require PySR (see tetriscnn/synthetic_configs.py)."
    )


def _notebooks():
    return sorted((REPO / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.stem)
def test_notebook_runs_without_the_optional_extra(path):
    """Every notebook must execute on a plain install.

    No notebook currently needs PySR: App. I's notebook plots recorded SR runs rather
    than fitting new ones. If one ever does, add it to an allowlist here along with a
    note in docs/NOTEBOOKS.md, so that readers are told before they hit the ImportError.
    """
    nb = json.loads(path.read_text())
    offenders = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:  # IPython magics etc.
            continue
        offenders.update(_offending(_module_level_imports(tree)))
    assert not offenders, (
        f"{path.relative_to(REPO)} imports {sorted(offenders)}, so it cannot run on a "
        "plain install. Import from a PySR-free module instead."
    )
