"""
WCS and image array utilities for PDR Toolbox.
"""

import numpy as np

from pdrtpy.utils.units import draine_unit, get_rad, habing_unit, is_rad


def mask_union(arrays):
    """Return the union mask (logical OR) of the input masked arrays.
    This is useful when doing arithmetic on images that don't have identical masks
    and you want the most restrictive mask.

    :param arrays: masked arrays to unionize
    :type arrays: :class:`numpy.ma.masked_array`
    :rtype: mask
    """
    z = list()
    for m in arrays:
        z.append(m.mask)
    return np.any(z, axis=0)


def dropaxis(w):
    """Drop the first single dimension axis from a World Coordiante System.  Returns the modified WCS if it had a single dimension axis or the original WCS if not.

    :param w: a WCS
    :type w: :class:`astropy.wcs.WCS`
    :rtype: :class:`astropy.wcs.WCS`
    """
    for i in range(len(w._naxis)):
        if w._naxis[i] == 1:
            return w.dropaxis(i)
    return w


def has_single_axis(w):
    """Check if the input WCS has any single dimension axes

    :param w: a WCS
    :type w: :class:`astropy.wcs.WCS`
    :return: True if the input WCS has any single dimension axes, False otherwise
    :rtype: bool
    """
    for i in range(len(w._naxis)):
        if w._naxis[i] == 1:
            return True
    return False


def squeeze(image):
    """Remove single-dimensional entries from image data and WCS.

    :param image: the image to convert. It must have a :class:`numpy.ndarray` data member and :class:`astropy.units.Unit` unit member.
    :type image: :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.

    :return: an image with single axes removed
    :rtype: :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement` as input
    """
    while has_single_axis(image.wcs):
        image.wcs = dropaxis(image.wcs)

    # np.squeeze is a no-op if there are no dimensions to squeeze
    if image.data is not None:
        image.data = np.squeeze(image.data)
    if image.uncertainty is not None:
        image.uncertainty.array = np.squeeze(image.uncertainty.array)
    if image.mask is not None:
        image.mask = np.squeeze(image.mask)

    # update the header which can be independent of WCS
    image.header["NAXIS"] = image.wcs.wcs.naxis
    i = image.wcs.wcs.naxis + 1
    nax = "NAXIS" + str(i)
    while image.header.pop(nax, None) is not None:
        i = i + 1
        nax = "NAXIS" + str(i)

    return image


def fliplabel(label):
    """Given a label that has a numerator and a denominator separated by a '/', return the
    reciprocal label.  For example, if the input label is '(x+y)/z' return 'z/(x+y)'.  This
    method simply looks for the '/' and swaps the substrings before and after it.

    :param label: the label to flip
    :type label: str
    :return: the reciprocal label
    :rtype: str
    :raises ValueError: if the input label has no '/'
    """
    ii = label.index("/")
    return label[ii + 1 :] + "/" + label[0:ii]


def get_xy_from_wcs(data, quantity=False, linear=False):
    """Get the x,y axis vectors from the WCS of the input image.

    :param data: the input image
    :type data: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :param quantity: If True, return the arrays as :class:`astropy.units.Quantity`. If False, the returned arrays are :class:`numpy.ndarray`.
    :type quantity: bool
    :param linear: If True, returned arrays are in linear space, if False they are in log space.
    :type linear: bool
    :return: The axis values as arrays.  Values are center of pixel.
    :rtype: :class:`numpy.ndarray` or :class:`astropy.units.Quantity`
    """
    w = data.wcs
    if w is None:
        raise Exception("No WCS in the input image")
    xind = np.arange(w._naxis[0])
    yind = np.arange(w._naxis[1])
    x_pixel_arrays = [xind, xind]
    y_pixel_arrays = [yind, yind]
    for _i in range(2, w.pixel_n_dim):
        x_pixel_arrays.append([0])
        y_pixel_arrays.append([0])

    if quantity:
        x = w.array_index_to_world(*x_pixel_arrays)[0]
        y = w.array_index_to_world(*y_pixel_arrays)[1]
        # Need to handle Habing or Draine units which are non-standard FITS.
        # Can't apply them to a WCS because it will raise an Exception.
        # See ModelSet.get_model
        cunit = data.header.get("CUNIT2", None)
        if cunit == "Habing":
            y._unit = habing_unit
        if cunit == "Draine":
            y._unit = draine_unit
        if linear:
            j = 10 * np.ones(len(x.value))
            k = 10 * np.ones(len(y.value))
            # ugh we are depending on CTYPE being properly indicated as log(whatever)
            if "log" in w.wcs.ctype[0].lower():
                x = np.power(j, x.value) * x.unit
            if "log" in w.wcs.ctype[1].lower():
                y = np.power(k, y.value) * y.unit
    else:
        x = w.array_index_to_world_values(*x_pixel_arrays)[0]
        y = w.array_index_to_world_values(*y_pixel_arrays)[1]
        if linear:
            j = 10 * np.ones(len(x))
            k = 10 * np.ones(len(y))
            if "log" in w.wcs.ctype[0].lower():
                x = np.power(j, x)
            if "log" in w.wcs.ctype[1].lower():
                y = np.power(k, y)
    return (x, y)


def rescale_axis_units(x, from_unit, from_ctype, to_unit, loglabel=True):
    """Rescale axis units and return updated axis values and label.

    :param x: axis values as a Quantity
    :param from_unit: original unit string
    :param from_ctype: original CTYPE string
    :param to_unit: target unit string, or None to keep original
    :param loglabel: if True, prefix label with 'log(...)'
    :return: (x, xlabel) tuple
    """
    import astropy.units as u

    xax_unit = u.Unit(from_unit)
    # cover the case where we had to erase the wcs unit to avoid FITS error
    if x._unit is None or x._unit is u.dimensionless_unscaled:
        x._unit = xax_unit
    if is_rad(xax_unit):
        if loglabel:
            xtype = f"log({get_rad(xax_unit)})"
        else:
            xtype = f"{get_rad(xax_unit)}"
    elif loglabel and "log" in from_ctype:
        if "_" in from_ctype:
            xtype = r"${\rm " + from_ctype + "}$"
        else:
            xtype = from_ctype
    else:
        xtype = from_ctype.replace("log(", "").replace(")", "")
        if "_" in xtype:
            xtype = r"${\rm " + xtype + "}$"
    if to_unit is not None:
        xax_unit = u.Unit(to_unit)
        if is_rad(to_unit):
            if loglabel:
                xtype = f"log({get_rad(xax_unit)})"
            else:
                xtype = f"{get_rad(xax_unit)}"
        elif loglabel and "log" in from_ctype:
            xtype = from_ctype
        else:
            xtype = from_ctype.replace("log(", "").replace(")", "")
        x = x.to(xax_unit)

    xlabel = rf"{xtype} [{xax_unit:latex_inline}]"

    return (x, xlabel)
