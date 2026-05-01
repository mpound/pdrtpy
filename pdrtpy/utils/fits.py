"""
FITS keyword utilities for PDR Toolbox.
"""

import numpy as np

from pdrtpy import version
from pdrtpy.utils.paths import now


def addkey(key, value, image):
    """Add a (FITS) keyword,value pair to the image header

    :param key:   The keyword to add to the header
    :type key:    str
    :param value: the value for the keyword
    :type value:  any native type
    :param image: The image which to add the key,val to.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    """
    if key in image.header and isinstance(value, str):
        s = str(image.header[key])
        # avoid concatenating duplicates
        if s != value:
            image.header[key] = str(image.header[key]) + " " + value
    else:
        image.header[key] = value


def comment(value, image):
    """Add a comment to an image header

    :param value: the value for the comment
    :type value:  str
    :param image: The image which to add the comment to
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    """
    # direct assignment will always make a new card for COMMENT
    # See https://docs.astropy.org/en/stable/io/fits/
    image.header["COMMENT"] = value


def history(value, image):
    """Add a history record to an image header

    :param value: the value for the history record
    :type value:  str
    :param image: The image which to add the HISTORY to
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    """
    # direct assignment will always make a new card for HISTORY
    image.header["HISTORY"] = value


def setkey(key, value, image):
    """Set the value of an existing keyword in the image header

    :param key:   The keyword to set in the header
    :type key:    str
    :param value: the value for the keyword
    :type value:  any native type
    :param image: The image which to add the key,val to.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    """
    image.header[key] = value


def dataminmax(image):
    """Set the data maximum and minimum in image header

    :param image: The image which to add the key,val to.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    """
    setkey("DATAMIN", np.nanmin(image.data), image)
    setkey("DATAMAX", np.nanmax(image.data), image)


def signature(image):
    """Add AUTHOR and DATE keywords to the image header
    Author is 'PDR Toolbox', date as returned by now()

    :param image: The image which to add the key,val to.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    """
    setkey("AUTHOR", "PDR Toolbox " + version(), image)
    setkey("DATE", now(), image)


def firstkey(d):
    """Return the "first" key in a dictionary

    :param d: the dictionary
    :type d: dict
    """
    return next(iter(d))
