"""
Path and I/O utilities for PDR Toolbox.
"""

import datetime
import os.path
from pathlib import Path

from astropy.table import Table


def now():
    """Return a string representing the current date and time in ISO format.

    Returns
    -------
    str
    """
    return datetime.datetime.now().isoformat()


def root_dir():
    """Project root directory, including trailing slash.

    Returns
    -------
    str
    """
    return str(root_path()) + "/"


def root_path():
    """Project root directory as path.

    Returns
    -------
    :py:class:`pathlib.Path`
    """
    # This file lives at pdrtpy/utils/paths.py; parent.parent is pdrtpy/
    return Path(__file__).parent.parent


def testdata_dir():
    """Project test data directory, including trailing slash.

    Returns
    -------
    str
    """
    return os.path.join(root_dir(), "testdata/")


def get_testdata(filename):
    """Get fully qualified pathname to FITS test data file.

    Parameters
    ----------
    filename : str
        Input filename, no path.
    """
    return testdata_dir() + filename


def model_dir():
    """Project model directory, including trailing slash.

    Returns
    -------
    str
    """
    return os.path.join(root_dir(), "models/")


def table_dir():
    """Project ancillary tables directory, including trailing slash.

    Returns
    -------
    str
    """
    return os.path.join(root_dir(), "tables/")


def _tablename(filename):
    """Return fully qualified path of the input table.

    Parameters
    ----------
    filename : str
        Input table file name.

    Returns
    -------
    str
    """
    return table_dir() + filename


def get_table(filename, format="ipac", path=None, **kwargs):
    """Return an astropy Table read from the input filename.

    Parameters
    ----------
    filename : str
        Input filename, no path.
    format : str, optional
        File format. Default: ``"ipac"``.
    path : str, optional
        Path to filename relative to models directory. Default of None means
        look in the ``tables`` directory.
    **kwargs : dict
        Additional arguments to pass to ``Table.read``, e.g. ``header_start``, ``data_start``.

    Returns
    -------
    :class:`astropy.table.Table`
    """
    if path is None:
        return Table.read(_tablename(filename), format=format, **kwargs)
    else:
        return Table.read(model_dir() + path + filename, format=format, **kwargs)
