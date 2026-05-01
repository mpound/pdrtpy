"""
Radiation field strength units, constants, and unit conversion utilities.
"""

from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.constants import k_B
from astropy.units.format.latex import Latex

# Radiation Field Strength units in cgs
_RFS_UNIT_ = u.erg / (u.second * u.cm * u.cm)
_OBS_UNIT_ = u.erg / (u.second * u.cm * u.cm * u.sr)
_CM = u.Unit("cm")
_CM2 = u.Unit("cm-2")
_K = u.Unit("K")
_KKMS = u.Unit("K km s-1")
LOGE = np.log10(np.e)
LN10 = np.log(10)

# ISRF in other units
# The wavelength of 1110 Ang is the longest wavelength for H2 excitation,
# but photoelectric heating can occur at longer wavelengths.
# We use  6eV to 13.6 eV, or 912 - 2066 Ang. For this range
# Draine is 1.7G_0, and Mathis is 1.13 G_0.
# See Weingartner and Draine 2001, ApJS, 134, 263, section 4.1

habing_unit = u.def_unit("Habing", 1.60e-3 * _RFS_UNIT_)
r"""The Habing radiation field unit

   :math:`{\rm 1~Habing = 1.6\times 10^{-3}~erg~s^{-1}~cm^{-2}}`
"""
u.add_enabled_units(habing_unit)
draine_unit = u.def_unit("Draine", 2.72e-3 * _RFS_UNIT_)
r"""The Draine radiation field unit

    :math:`{\rm 1~Draine = 2.72\times10^{-3}~erg~s^{-1}~cm^{-2}}`
"""
u.add_enabled_units(draine_unit)
mathis_unit = u.def_unit("Mathis", 1.81e-3 * _RFS_UNIT_)
r"""The Mathis radiation field unit

    :math:`{\rm 1~Mathis = 1.81\times10^{-3}~erg~s^{-1}~cm^{-2}}`
"""
u.add_enabled_units(mathis_unit)

_rad_title = dict()
_rad_title["Habing"] = "$G_0$"
_rad_title["Draine"] = r"$\chi$"
_rad_title["Mathis"] = "FUV"


def get_rad(key):
    r"""Get radiation field symbol (LaTeX) given radiation field unit.
    If key is unrecognized, 'FUV Radiation Field' is returned.

    :param key: input field unit name, e.g. 'Habing', 'Draine' or an :class:`astropy.units.Unit`
    :returns: LaTeX string for the radiation field symbol e.g., :math:`G_0`, :math:`\chi`
    :type key: str or :class:`astropy.units.Unit`
    :rtype: str
    """
    skey = str(key)  # in case the passed key was a Unit
    if skey in _rad_title:
        return _rad_title[skey]
    else:
        return "FUV"


def check_units(input_unit, compare_to):
    """Check if the input unit is equivalent to another.

    :param input_unit:  the unit to check.
    :type input_unit:  :class:`astropy.units.Unit`, :class:`astropy.units.Quantity` or str
    :param compare_unit:  the unit to check against
    :type compare_unit:  :class:`astropy.units.Unit`, :class:`astropy.units.Quantity` or str
    :return: `True` if the input unit is equivalent to compare unit, `False` otherwise
    """
    if isinstance(input_unit, u.Unit):
        test_unit = input_unit
    if isinstance(input_unit, u.Quantity):
        test_unit = input_unit.unit
    else:  # assume it is a string
        test_unit = u.Unit(input_unit)

    if isinstance(compare_to, u.Unit):
        compare_unit = compare_to
    if isinstance(compare_to, u.Quantity):
        compare_unit = compare_to.unit
    else:  # assume it is a string
        compare_unit = u.Unit(compare_to)

    return test_unit.is_equivalent(compare_unit)


def is_rad(input_unit):
    return check_units(input_unit, _RFS_UNIT_)


def to(unit, image):
    r"""Convert the image values to another unit.
    While generally this is intended for converting radiation field
    strength maps between Habing, Draine, cgs, etc, it will work for
    any image that has a unit member variable. So, e.g., it would work
    to convert density from :math:`{\rm cm ^{-3}}` to :math:`{\rm m^{-3}}`.
    If the input image is a :class:`~pdrtpy.measurement.Measurement`, its
    uncertainty will also be converted.

    :param unit: identifying the unit to convert to
    :type unit: string or `astropy.units.Unit`
    :param image: the image to convert. It must have a :class:`numpy.ndarray`
        data member and :class:`astropy.units.Unit` unit member.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :return: an image with converted values and units
    """
    value = image.unit.to(unit)
    newmap = deepcopy(image)
    newmap.data = newmap.data * value
    newmap.unit = u.Unit(unit)
    if newmap._uncertainty is not None:
        newmap._uncertainty.array = newmap.uncertainty.array * value
        newmap._uncertainty.unit = u.Unit(unit)
    return newmap


def toHabing(image):
    r"""Convert a radiation field strength image to Habing units :math:`(G_0)`.

    :math:`{\rm G_0 \equiv 1~Habing = 1.6\times10^{-3}~erg~s^{-1}~cm^{-2}}`

    between 6eV and 13.6eV (912-2066 :math:`\unicode{xC5}`).  See `Weingartner and Draine 2001, ApJS, 134, 263 <https://ui.adsabs.harvard.edu/abs/2001ApJS..134..263W/abstract>`_, section 4.1

    :param image: the image to convert. It must have a :class:`numpy.ndarray`
       data member and :class:`astropy.units.Unit` unit member.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :return: an image with converted values and units
    """
    return to("Habing", image)


def toDraine(image):
    r"""Convert a radiation field strength image to Draine units (\chi).

    :math:`{\rm 1~Draine = 2.72\times10^{-3}~erg~s^{-1}~cm^{-2}}`

    between 6eV and 13.6eV (912-2066 :math:`\unicode{xC5}`).  See `Weingartner and Draine 2001, ApJS, 134, 263 <https://ui.adsabs.harvard.edu/abs/2001ApJS..134..263W/abstract>`_, section 4.1

    :param image: the image to convert. It must have a :class:`numpy.ndarray`
       data member and :class:`astropy.units.Unit` unit member.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :return: an image with converted values and units
    """
    return to("Draine", image)


def toMathis(image):
    r"""Convert a radiation field strength image to Mathis units

    :math:`{\rm 1~Mathis = 1.81\times10^{-3}~erg~s^{-1}~cm^{-2}}`

    between 6eV and 13.6eV (912-2066 :math:`\unicode{xC5}`).  See `Weingartner and Draine 2001, ApJS, 134, 263 <https://ui.adsabs.harvard.edu/abs/2001ApJS..134..263W/abstract>`_, section 4.1

    :param image: the image to convert. It must have a :class:`numpy.ndarray`
       data member and :class:`astropy.units.Unit` unit member.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :return: an image with converted values and units
    """
    return to("Mathis", image)


def tocgs(image):
    r"""Convert a radiation field strength image to :math:`{\rm erg~s^{-1}~cm^{-2}}`.

    :param image: the image to convert. It must have a :class:`numpy.ndarray` data member and :class:`astropy.units.Unit` unit member.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :return: an image with converted values and units
    """
    return to(_RFS_UNIT_, image)


def convert_integrated_intensity(image, wavelength=None):
    r"""Convert integrated intensity from :math:`{\rm K~km~s}^{-1}` to
    :math:`{\rm erg~s^{-1}~cm^{-2}~sr^{-1}}`, assuming
    :math:`B_\lambda d\lambda = 2kT/\lambda^3 dV` where :math:`T dV` is the integrated intensity in K km/s and :math:`\lambda` is the wavelength.  The derivation:

    .. math::

       B_\lambda = 2 h c^2/\lambda^5  {1\over{exp[hc/\lambda k T] - 1}}

    The integrated line is :math:`B_\lambda d\lambda` and for :math:`hc/\lambda k T << 1`:

    .. math::

       B_\lambda d\lambda = 2c^2/\lambda^5  \times (\lambda kT/hc)~d\lambda

    The relationship between velocity and wavelength, :math:`dV = \lambda/c~d\lambda`, giving

    .. math::

       B_\lambda d\lambda = 2\times10^5~kT/\lambda^3~dV,

    with :math:`\lambda`  in cm, the factor :math:`10^5` is to convert :math:`dV` in :math:`{\rm km~s}^{-1}` to :math:`{\rm cm~s}^{-1}`.

    :param image: the image to convert. It must have a :class:`numpy.ndarray` data member and :class:`astropy.units.Unit` unit member or header BUNIT keyword. It's units must be K km/s
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
    :param wavelength: the wavelength of the observation. The default is to determine wavelength from the image header RESTFREQ keyword
    :type wavelength: :class:`astropy.units.Quantity`
    :return: an image with converted values and units
    """
    f = image.header.get("RESTFREQ", None)
    rf = getattr(image, "_restfreq", None)
    if f is None and wavelength is None and rf is None:
        raise Exception(
            "Image header has no RESTFREQ and image has no '_restfreq' attribute. You must supply wavelength"
        )
    if f is not None and wavelength is None:
        # FITS restfreq's are in Hz
        wavelength = u.Quantity(f, "Hz").to(_CM, equivalencies=u.spectral())
    elif rf is not None and wavelength is None:
        wavelength = rf.to(_CM, equivalencies=u.spectral())
    if image.header.get("BUNIT", None) is None:
        raise Exception("Image BUNIT must be present and equal to 'K km/s'")
    if u.Unit(image.header.get("BUNIT")) != _KKMS:
        raise Exception("Image BUNIT must be 'K km/s'")
    factor = 2e5 * k_B / wavelength**3
    print(f"Converting K km/s to {_OBS_UNIT_} using Factor = {factor.decompose(u.cgs.bases):+0.3E}")
    newmap = deepcopy(image)
    value = factor.decompose(u.cgs.bases).value
    newmap.data = newmap.data * value
    newmap.unit = _OBS_UNIT_
    if newmap._uncertainty is not None:
        newmap._uncertainty.array = newmap.uncertainty.array * value
        newmap._uncertainty.unit = _OBS_UNIT_
    return newmap


def convert_if_necessary(image):
    r"""Helper method to convert integrated intensity units in an
    image or Measurement from :math:`{\rm K~km~s}^{-1}` to :math:`{\rm
    erg~s^{-1}~cm^{-2}~sr^{-1}}`. If a conversion is necessary, the
    :meth:`convert_integrated_intensity` is called.  If not, the image
    is returned unchanged.

    :param image: the image to convert. It must have a :class:`numpy.ndarray` data member and :class:`astropy.units.Unit` unit member or a header BUNIT keyword. It's units must be :math:`{\rm K~km~s}^{-1}`. It must also have a header RESTFREQ keyword.
    :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.

    :return: an image with converted values and units
    """
    _u1 = u.Unit(image.header["BUNIT"])
    _u2 = u.Unit("K km s-1")
    if _u1 == _u2:
        return convert_integrated_intensity(image)
    else:
        return image


def float_formatter(quantity, precision):
    """Format a quantity as a LaTeX string with the given number of significant figures.

    :param quantity: the quantity to format
    :type quantity: :class:`astropy.units.Quantity`
    :param precision: the number of significant figures
    :type precision: int
    :rtype: str
    """
    format_spec = f".{precision}g"
    number = Latex.format_exponential_notation(np.squeeze(quantity.value), format_spec=format_spec)
    # strip the $ signs
    unit = quantity.unit.to_string("latex_inline")[1:-1]
    return f"{number}~{unit}"
