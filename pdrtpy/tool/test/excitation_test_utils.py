"""Utilities for loading excitation fit test cases from JSON data files."""

import json
from pathlib import Path

import astropy.units as u
import pdrtpy.tool.excitation as excitation_module
import pdrtpy.utils as utils
from astropy.nddata import StdDevUncertainty
from pdrtpy.measurement import Measurement

_TESTDATA_DIR = Path(utils.testdata_dir())


def list_excitation_testdata():
    """Return sorted list of paths to all excitation_*.json files in testdata."""
    return sorted(_TESTDATA_DIR.glob("excitation_*.json"))


def load_excitation_testcase(path):
    """Load an excitation fit test case from a JSON file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    fitter : ExcitationFit subclass instance (not yet run)
    meta : dict with keys: components, fit_opr, expected
    """
    with open(path) as f:
        data = json.load(f)

    unit = u.Unit(data["unit"])
    measurements = [
        Measurement(
            data=[line["intensity"]],
            uncertainty=StdDevUncertainty(line["uncertainty"]),
            identifier=line["identifier"],
            unit=unit,
        )
        for line in data["lines"]
    ]

    fit_class = getattr(excitation_module, data["fit_class"])
    fitter = fit_class(measurements)

    meta = dict()
    for k in data:
        if k != "lines" and k != "fit_opr":
            meta[k] = data[k]
    meta["fit_opr"] = data.get("fit_opr", False)
    return fitter, meta
