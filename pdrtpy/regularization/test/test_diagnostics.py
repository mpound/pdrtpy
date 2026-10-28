"""Unit tests for pdrtpy.regularization.diagnostics."""

import numpy as np
import pytest
from pdrtpy.regularization import change_summary, format_summary


def test_identical_arrays_have_no_changes():
    a = np.full((4, 4), 100.0)
    s = change_summary(a, a.copy())
    assert s["n_valid"] == 16
    assert s["n_changed"] == 0
    assert s["fraction_changed"] == 0.0
    assert s["max_abs"] == 0.0


def test_known_changed_count_and_magnitude():
    before = np.full((3, 3), 10.0)
    after = before.copy()
    after[0, 0] = 100.0  # +1 dex
    after[1, 1] = 1.0  # -1 dex
    after[2, 2] = 10.0 * 10**0.5  # +0.5 dex
    s = change_summary(before, after, threshold=0.1)
    assert s["n_valid"] == 9
    assert s["n_changed"] == 3
    assert s["fraction_changed"] == pytest.approx(3 / 9)
    assert s["max_abs"] == pytest.approx(1.0)
    assert s["mean_abs"] == pytest.approx(2.5 / 9)
    assert sorted(s["delta"])[0] == pytest.approx(-1.0)


def test_threshold_is_strictly_greater_than():
    before = np.array([1.0, 1.0])
    after = np.array([10**0.5, 10**0.25])
    s = change_summary(before, after, threshold=0.5)
    assert s["n_changed"] == 0  # |delta| == threshold is not "changed"
    assert change_summary(before, after, threshold=0.4)["n_changed"] == 1


def test_nan_and_nonpositive_excluded():
    before = np.array([1.0, np.nan, 1.0, 0.0, 1.0])
    after = np.array([10.0, 1.0, np.nan, 1.0, -1.0])
    s = change_summary(before, after)
    assert s["n_valid"] == 1
    assert s["n_changed"] == 1


def test_valid_mask_applied():
    before = np.ones((2, 2))
    after = np.full((2, 2), 10.0)
    mask = np.array([[True, False], [False, False]])
    s = change_summary(before, after, valid_mask=mask)
    assert s["n_valid"] == 1


def test_no_valid_pixels_gives_nan_stats():
    s = change_summary(np.array([np.nan]), np.array([np.nan]))
    assert s["n_valid"] == 0
    assert s["n_changed"] == 0
    assert np.isnan(s["median_abs"])
    assert np.isnan(s["fraction_changed"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        change_summary(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        change_summary(np.ones(3), np.ones(3), valid_mask=np.ones(4, dtype=bool))


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        change_summary(np.ones(3), np.ones(3), threshold=-1)


def test_format_summary_mentions_counts():
    s = change_summary(np.ones(4), np.array([1.0, 1.0, 10.0, 10.0]))
    text = format_summary(s, "density")
    assert "density" in text
    assert "2 of 4" in text


def test_sigma_counts_changes_beyond_uncertainty():
    before = np.full(4, 10.0)
    after = 10.0 * 10 ** np.array([0.0, 0.05, 0.2, -0.2])
    sigma_dex = 0.1
    sigma = np.full(4, 10.0 * np.log(10.0) * sigma_dex)
    s = change_summary(before, after, threshold=0.01, sigma=sigma)
    assert s["n_sigma_valid"] == 4
    assert s["n_beyond_sigma"] == 2


def test_sigma_unusable_pixels_skipped_and_none_without_sigma():
    before = np.full(3, 10.0)
    after = np.full(3, 100.0)
    s = change_summary(before, after, sigma=np.array([1.0, np.nan, 0.0]))
    assert s["n_sigma_valid"] == 1
    assert change_summary(before, after)["n_beyond_sigma"] is None
    with pytest.raises(ValueError):
        change_summary(before, after, sigma=np.ones(2))
