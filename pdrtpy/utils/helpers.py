"""
General helper utilities for PDR Toolbox.
"""

import warnings
from copy import deepcopy

from pdrtpy.utils.fits import comment


def warn(cls, msg):
    """Issue a warning.

    Parameters
    ----------
    cls : class
        The calling class.
    msg : str
        The warning message.
    """
    # use stacklevel=3 so we get a reference to the caller of warn().
    warnings.warn(cls.__class__.__name__ + ": " + msg, stacklevel=3)


def is_image(image):
    """Check if a Measurement is an image.

    To be an image it must have a header with axes keywords and a WCS.
    This is to distinguish Measurements that have a data array with more
    than one member from a true image.

    Parameters
    ----------
    image : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
        The image to check. It must have a :class:`numpy.ndarray` data member
        and :class:`astropy.units.Unit` unit member or a header BUNIT keyword.

    Returns
    -------
    bool
        True if it is an image, False otherwise.
    """
    if getattr(image, "header", None) is None or getattr(image, "wcs", None) is None:
        return False
    if image.wcs.naxis is None or image.wcs.wcs is None:
        return False
    if image.wcs.naxis == 0:  # naxis=1 ok -- a 1-D image is still an image.
        return False
    if image.wcs.wcs.ctype is None:
        return False
    return True


def is_ratio(identifier):
    """Is the identifier a ratio (as opposed to an intensity).

    Returns
    -------
    bool
    """
    # find() returns -1 if char not found.
    # in our case, also rule out that the / is in the zeroth position.
    return identifier.find("/") > 0


def is_even(number):
    """Check if number is even.

    Parameters
    ----------
    number : float
        A number.

    Returns
    -------
    bool
        True if even, False otherwise.
    """
    return abs(number) % 2 == 0


def is_odd(number):
    """Check if number is odd.

    Parameters
    ----------
    number : float
        A number.

    Returns
    -------
    bool
        True if odd, False otherwise.
    """
    return not is_even(number)


def _has_substring(s, ids):
    return any([s in c for c in ids])


def _has_H2(ids):
    return _has_substring("H2", ids)


def _trim_to_H2(image):
    """H2 models in wk2006 are a smaller grid 17x17 vs 25x29. So when performing operations
    involving other models, we have to trim the other models to 17x17;  log(n,G0) from 1 to 5.

    Parameters
    ----------
    image : :class:`~pdrtpy.measurement.Measurement`
        The model to trim.

    Returns
    -------
    :class:`~pdrtpy.measurement.Measurement`
        The trimmed model.
    """
    f = deepcopy(image)
    # Slice the WCS. Note this is in numpy array order, not WCS axis order
    f.wcs = f.wcs[6:23, 0:17]
    f.data = f.data[6:23, 0:17]
    f.meta["NAXIS1"] = 17
    f.meta["NAXIS2"] = 17
    comment("Trimmed model", f)
    return f


def _trim_all_to_H2(models):
    """H2 models in wk2006 are a smaller grid 17x17 vs 25x29. So when performing operations
    involving other models, we have to trim the other models to 17x17;  log(n,G0) from 1 to 5.

    Parameters
    ----------
    models : list or dict of :class:`~pdrtpy.measurement.Measurement`
        Models to trim.
    """
    if type(models) is dict:
        for id in models:
            if "H2" not in id:
                models[id] = _trim_to_H2(models[id])
    else:
        # have to iterate over index to ensure "pass by reference"
        # if we did for m in models: m = ..., then models
        # remains unchanged at end of function.  Wheee, python!
        for j in range(len(models)):
            if "H2" not in models[j].id:
                models[j] = _trim_to_H2(models[j])
