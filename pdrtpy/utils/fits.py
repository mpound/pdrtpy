"""
FITS keyword utilities for PDR Toolbox.
"""

import numpy as np

from pdrtpy import version
from pdrtpy.utils.paths import now


def addkey(key, value, image):
    """Add a (FITS) keyword,value pair to the image header.

    Parameters
    ----------
    key : str
        The keyword to add to the header.
    value : any
        The value for the keyword.
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to which to add the key,val.
    """
    if key in image.header and isinstance(value, str):
        s = str(image.header[key])
        # avoid concatenating duplicates
        if s != value:
            image.header[key] = str(image.header[key]) + " " + value
    else:
        image.header[key] = value


def comment(value, image):
    """Add a comment to an image header.

    Parameters
    ----------
    value : str
        The value for the comment.
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to which to add the comment.
    """
    # direct assignment will always make a new card for COMMENT
    # See https://docs.astropy.org/en/stable/io/fits/
    image.header["COMMENT"] = value


def history(value, image):
    """Add a history record to an image header.

    Parameters
    ----------
    value : str
        The value for the history record.
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to which to add the HISTORY.
    """
    # direct assignment will always make a new card for HISTORY
    image.header["HISTORY"] = value


def setkey(key, value, image):
    """Set the value of an existing keyword in the image header.

    Parameters
    ----------
    key : str
        The keyword to set in the header.
    value : any
        The value for the keyword.
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to which to add the key,val.
    """
    image.header[key] = value


def dataminmax(image):
    """Set the data maximum and minimum in image header.

    Parameters
    ----------
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to which to add the key,val.
    """
    setkey("DATAMIN", np.nanmin(image.data), image)
    setkey("DATAMAX", np.nanmax(image.data), image)


def signature(image):
    """Add AUTHOR and DATE keywords to the image header.

    Author is ``'PDR Toolbox'``, date as returned by :func:`~pdrtpy.utils.paths.now`.

    Parameters
    ----------
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to which to add the key,val.
    """
    setkey("AUTHOR", "PDR Toolbox " + version(), image)
    setkey("DATE", now(), image)


def firstkey(d):
    """Return the "first" key in a dictionary.

    Parameters
    ----------
    d : dict
        The dictionary.
    """
    return next(iter(d))
