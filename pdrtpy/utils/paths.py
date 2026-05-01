"""
Path and I/O utilities for PDR Toolbox.
"""

import datetime
import os.path
from pathlib import Path

from astropy.table import Table


def now():
    """
    :returns: a string representing the current date and time in ISO format
    """
    return datetime.datetime.now().isoformat()


def root_dir():
    """Project root directory, including trailing slash

    :rtype: str
    """
    return str(root_path()) + "/"


def root_path():
    """Project root directory as path

    :rtype: :py:mod:`Path`
    """
    # This file lives at pdrtpy/utils/paths.py; parent.parent is pdrtpy/
    return Path(__file__).parent.parent


def testdata_dir():
    """Project test data directory, including trailing slash

    :rtype: str
    """
    return os.path.join(root_dir(), "testdata/")


def get_testdata(filename):
    """Get fully qualified pathname to FITS test data file.

    :param filename: input filename, no path
    :type filename: str
    """
    return testdata_dir() + filename


def model_dir():
    """Project model directory, including trailing slash

    :rtype: str
    """
    return os.path.join(root_dir(), "models/")


def table_dir():
    """Project ancillary tables directory, including trailing slash

    :rtype: str
    """
    return os.path.join(root_dir(), "tables/")


def _tablename(filename):
    """Return fully qualified path of the input table.

    :param filename: input table file name
    :type filename: str
    :rtype: str
    """
    return table_dir() + filename


def get_table(filename, format="ipac", path=None, **kwargs):
    """Return an astropy Table read from the input filename.

    :param filename: input filename, no path
    :type filename: str
    :param format:  file format, Default: "ipac"
    :type format: str
    :param  path: path to filename relative to models directory.  Default of None means look in "tables" directory
    :type path: str
    :param kwargs: additional arguments to pass to Table.read, e.g. `header_start`, `data_start`
    :type kwargs: dict
    :rtype: :class:`astropy.table.Table`

    """
    if path is None:
        return Table.read(_tablename(filename), format=format, **kwargs)
    else:
        return Table.read(model_dir() + path + filename, format=format, **kwargs)
