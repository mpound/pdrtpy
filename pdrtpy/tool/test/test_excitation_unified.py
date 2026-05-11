#!/usr/bin/env python3
"""Parametrized excitation fit tests driven by JSON data files in pdrtpy/testdata/."""

import pytest
from pdrtpy.tool.test.excitation_test_utils import list_excitation_testdata, load_excitation_testcase

_TESTCASES = list_excitation_testdata()


@pytest.mark.parametrize("datafile", _TESTCASES, ids=[p.name for p in _TESTCASES])
def test_excitation_fit(datafile):
    fitter, meta = load_excitation_testcase(datafile)
    fitter.run(components=meta["components"], fit_opr=meta["fit_opr"])

    exp = meta["expected"]

    assert fitter.thot.data == pytest.approx(exp["thot"]["value"], rel=exp["thot"]["rel_tol"])
    assert fitter.tcold.data == pytest.approx(exp["tcold"]["value"], rel=exp["tcold"]["rel_tol"])
    assert fitter.hot_colden.data == pytest.approx(exp["hot_colden"]["value"], rel=exp["hot_colden"]["rel_tol"])
    assert fitter.cold_colden.data == pytest.approx(exp["cold_colden"]["value"], rel=exp["cold_colden"]["rel_tol"])

    if meta["fit_opr"] and exp["opr"] is not None:
        assert fitter.opr.data == pytest.approx(exp["opr"]["value"], rel=exp["opr"]["rel_tol"])
