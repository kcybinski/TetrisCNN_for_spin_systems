"""Numeric arrays in config.json: how they are written, and how old runs are read back.

save_json() used to record numpy arrays through str(), so most runs under Plots_data/
(main-text runs and, nested inside, the App_*_data/ appendix runs) store fields such
as `unique_labels` as a printed array rather than a JSON list. Three separate readers
grew their own fix for that. These tests pin the shared replacement against every
recorded run, so the consolidation is provably a no-op on existing data, and pin the
writer so new runs do not recreate the problem.
"""
import glob
import json
import warnings

import numpy as np
import pytest

from tests.conftest import REPO_ROOT
from tetriscnn.utils import load_json, parse_array_field, save_json


def _recorded_configs():
    # Recursive, so this also picks up the App_*_data/ appendix runs now nested
    # inside Plots_data/ (they used to be separate top-level directories).
    pattern = "Plots_data/**/config.json"
    return sorted(glob.glob(str(REPO_ROOT / pattern), recursive=True))


def _legacy_parse(raw):
    """The inline fallback plot_lbc and lbc_lambdamax_summary.py used to carry."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(raw, str):
            return np.fromstring(raw.strip("[]"), sep=" ")
        return np.array(raw, dtype=float)


RECORDED = _recorded_configs()


@pytest.mark.skipif(not RECORDED, reason="no recorded runs present")
def test_parser_agrees_with_the_legacy_fallback_wherever_that_worked():
    """Identical output on every run the old inline fallback could read.

    The old fallback flattened by stripping brackets, so it raised on a printed 2-D
    column such as a delta label table ("[[7.5e-03]\\n [1.2e-02] ...]"). Those runs
    are checked separately below; everywhere else the consolidation must be a no-op.
    """
    agreed, legacy_failed = 0, 0
    for path in RECORDED:
        raw = json.load(open(path)).get("unique_labels")
        if raw is None:
            continue
        try:
            expected = _legacy_parse(raw)
        except ValueError:
            legacy_failed += 1
            continue
        np.testing.assert_array_equal(parse_array_field(raw).ravel(), np.ravel(expected), err_msg=path)
        agreed += 1
    assert agreed > 0


@pytest.mark.skipif(not RECORDED, reason="no recorded runs present")
def test_parser_reads_the_runs_the_legacy_fallback_crashed_on():
    crashed = []
    for path in RECORDED:
        raw = json.load(open(path)).get("unique_labels")
        if not isinstance(raw, str):
            continue
        try:
            _legacy_parse(raw)
        except ValueError:
            crashed.append((path, raw))
    for path, raw in crashed:
        parsed = parse_array_field(raw)
        # every number in the printed table survives, and the column shape is kept
        n_numbers = len(raw.replace("[", " ").replace("]", " ").split())
        assert parsed.size == n_numbers, path
        assert parsed.ndim == 2, path


@pytest.mark.skipif(not RECORDED, reason="no recorded runs present")
def test_recorded_unique_labels_parse_to_a_usable_sweep():
    for path in RECORDED:
        raw = json.load(open(path)).get("unique_labels")
        if raw is None:
            continue
        labels = parse_array_field(raw)
        assert labels.ndim >= 1 and labels.size > 1, path
        # A sweep axis is strictly increasing; a mangled parse would break that.
        assert np.all(np.diff(labels.reshape(len(labels), -1)[:, 0]) > 0), path


def test_parser_accepts_a_json_list():
    np.testing.assert_array_equal(parse_array_field([0, 250, 500]), [0.0, 250.0, 500.0])


def test_parser_reads_the_printed_one_dimensional_form():
    printed = str(np.array([0.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0,
                            2000.0, 2500.0, 3000.0, 4000.0, 6000.0]))
    assert "\n" in printed            # the realistic case wraps onto a second line
    np.testing.assert_array_equal(parse_array_field(printed)[[0, -1]], [0.0, 6000.0])
    assert parse_array_field(printed).shape == (13,)


def test_parser_keeps_the_shape_of_a_printed_table():
    # A deltaomega label table is two-dimensional; flattening it would silently
    # interleave delta and omega.
    table = np.array([[1.5, -2.0], [3.25, 4.0], [5.0, 6.5]])
    np.testing.assert_array_equal(parse_array_field(str(table)), table)


def test_save_json_writes_arrays_as_lists(tmp_path):
    save_json({"unique_labels": np.array([0.0, 250.0]), "scale": np.float32(0.5),
               "count": np.int64(3)}, tmp_path, "config.json")
    raw = json.load(open(tmp_path / "config.json"))
    assert raw["unique_labels"] == [0.0, 250.0]
    assert raw["scale"] == 0.5 and raw["count"] == 3


def test_save_json_round_trips_through_the_parser(tmp_path):
    table = np.array([[1.0, 2.0], [3.0, 4.0]])
    save_json({"unique_labels": table}, tmp_path, "config.json")
    back = load_json(tmp_path, "config.json")
    np.testing.assert_array_equal(parse_array_field(back["unique_labels"]), table)
