"""Manage spectral line or continuum observations"""

import warnings
from copy import deepcopy
from os import remove
from os.path import exists

import astropy.units as u
import numpy as np
import numpy.ma as ma
from astropy import log
from astropy.io import fits, registry
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator

from . import utils

log.setLevel("WARNING")  # see issue 163


class Measurement(CCDData):
    r"""Measurement represents one or more observations of a given spectral
    line or continuum.  It is made up of a value array, an
    uncertainty array, units, and a string identifier. It is based
    on :class:`astropy.nddata.CCDData`.  It can represent a single pixel
    observation or an image.   Mathematical operations using Measurements
    will correctly propagate errors.

    Typically, Measurements will be instantiated from a FITS file by using the the :func:`read` or :func:`make_measurement` methods.  For a list of recognized spectral line identifiers, see :meth:`~pdrtpy.modelset.Modelset.supported_lines`.

    Parameters
    ----------
    data : array-like
        The actual data. Note that data is saved by *reference*, so make a
        copy before passing if needed.
    uncertainty : :class:`~astropy.nddata.StdDevUncertainty`, :class:`~astropy.nddata.VarianceUncertainty`, :class:`~astropy.nddata.InverseVariance`, or :class:`numpy.ndarray`
        Uncertainties on the data. If a :class:`numpy.ndarray`, it is stored
        as :class:`~astropy.nddata.StdDevUncertainty`. Required.
    unit : :class:`astropy.units.Unit` or str
        The units of the data. Required.
    identifier : str
        A string indicating what this is an observation of, e.g., ``"CO_10"`` for CO(1-0).
    title : str, optional
        A formatted string (e.g., LaTeX) for plotting. r-strings are accepted,
        e.g., ``r'$^{13}$CO(3-2)'`` gives :math:`^{13}{\rm CO(3-2)}`.
    bmaj : :class:`astropy.units.Quantity`, optional
        Beam major axis diameter. Converted to degrees for FITS header storage.
    bmin : :class:`astropy.units.Quantity`, optional
        Beam minor axis diameter. Converted to degrees for FITS header storage.
    bpa : :class:`astropy.units.Quantity`, optional
        Beam position angle. Converted to degrees for FITS header storage.

    Raises
    ------
    TypeError
        If beam parameters are not Quantities.

    Measurements can also be instantiated by the **read(\\*args, \\**kwargs)**,
    to create an Measurement instance based on a ``FITS`` file.
    This method uses :func:`fits_measurement_reader` with the provided
    parameters.  Example usage:

    .. code-block:: python

       from pdrtpy.measurement import Measurement

       my_obs = Measurement.read("file.fits",identifier="CII_158")
       my_other_obs = Measurement.read("file2.fits",identifier="CO2_1",
                                        unit="K km/s",
                                        bmaj=9.3*u.arcsec,
                                        bmin=14.1*u.arcsec,
                                        bpa=23.2*u.degrees)

    By default image axes with only a single dimension are removed on read.  If you do not want this behavior, used `read(squeeze=False)`. See also: :class:`astropy.nddata.CCDData`.
    """

    def __init__(self, *args, **kwargs):
        warnings.simplefilter("ignore", DeprecationWarning)
        debug = kwargs.pop("debug", False)

        if debug:
            print("args=", *args)
            print("kwargs=", *kwargs)
        self._identifier = kwargs.pop("identifier", "unknown")
        self._title = kwargs.pop("title", None)
        _beam = dict()
        _beam["BMAJ"] = self._beam_convert(kwargs.pop("bmaj", None))
        _beam["BMIN"] = self._beam_convert(kwargs.pop("bmin", None))
        _beam["BPA"] = self._beam_convert(kwargs.pop("bpa", None))
        self._restfreq = kwargs.pop("restfreq", None)
        self._filename = None

        # CCDData raises an exception if unit not given. Whereas having BUNIT
        # in the image header instead would be perfectly reasonable.
        # The side-effect is that a Measurement with no unit given gets "adu".
        self._defunit = "adu"
        unitpresent = "unit" in kwargs
        _unit = kwargs.pop("unit", self._defunit)

        CCDData.__init__(self, *args, **kwargs, unit=_unit)
        # Force single-pixel data to be iterable arrays; CCDData/StdDevUncertainty don't do this.
        if np.shape(self.data) == ():
            self.data = np.array([self.data])
        if self.error is not None and np.shape(self.error) == ():
            self.uncertainty.array = np.array([self.uncertainty.array])

        # If user provided restfreq, insert it into header
        # FITS standard is Hz
        if self._restfreq is not None:
            rf = u.Unit(self._restfreq).to("Hz")
            self.header["RESTFREQ"] = rf
        # Set unit to header BUNIT or put BUNIT into header if it
        # wasn't present AND if unit wasn't given in the constructor
        if not unitpresent and "BUNIT" in self.header:
            self._unit = u.Unit(self.header["BUNIT"])
            if self.uncertainty is not None:
                self.uncertainty._unit = u.Unit(self.header["BUNIT"])
        else:
            # use str in case a astropy.Unit was given
            self.header["BUNIT"] = str(_unit)
        # Ditto beam parameters
        if "BMAJ" not in self.header:
            self.header["BMAJ"] = _beam["BMAJ"]
        if "BMIN" not in self.header:
            self.header["BMIN"] = _beam["BMIN"]
        if "BPA" not in self.header:
            self.header["BPA"] = _beam["BPA"]
        if self.wcs is not None:
            self._set_up_for_interp()

    def _beam_convert(self, bpar):
        if bpar is None:
            return bpar
        if isinstance(bpar, u.Quantity):
            return bpar.to("degree").value
        raise TypeError("Beam parameters must be astropy Quantities")

    @staticmethod
    def make_measurement(datafile, error, outfile, rms=None, masknan=True, overwrite=False, unit="adu"):
        """Create a FITS file with 2 HDUs: the first containing the data, the second containing the uncertainty.

        This format allows the resulting file to be read by the underlying
        :class:`~astropy.nddata.CCDData` class.

        Parameters
        ----------
        datafile : str
            The FITS file containing the data as a function of spatial coordinates.
        error : str
            The errors on the data. Possible values:

             - a filename with the same shape as datafile containing per-pixel errors
             - a percentage string ``'XX%'`` (must include the ``%`` symbol)
             - ``'rms'``: use the ``rms`` parameter if given, otherwise look for the RMS keyword in the FITS header

        outfile : str
            The output FITS file to write the result to.
        rms : float or :class:`astropy.units.Unit`, optional
            If ``error == 'rms'``, the rms value in the same units as the data (e.g. ``'erg s-1 cm-2 sr-1'``).
        masknan : bool, optional
            Whether to mask any pixel where the data or error is NaN. Default: True.
        overwrite : bool, optional
            If True, overwrite the output file if it exists. Default: False.
        unit : :class:`astropy.units.Unit` or str, optional
            Intensity unit for the data; overrides BUNIT in the header if present.

        Raises
        ------
        Exception
            On various FITS header issues.
        OSError
            If ``overwrite`` is False and the output file exists.

        Example usage:

        .. code-block:: python

            # example with percentage error
            Measurement.make_measurement("my_infile.fits",error='10%',outfile="my_outfile.fits")

            # example with measurement in units of K km/s and error
            # indicated by RMS keyword in input file.
            Measurement.make_measurement("my_infile.fits",error='rms',outfile="my_outfile.fits",unit="K km/s",overwrite=True)
        """
        _data = fits.open(datafile)
        needsclose = False
        if error == "rms":
            _error = deepcopy(_data)
            if rms is None:
                rms = _data[0].header.get("RMS", None)
                if rms is None:
                    raise Exception("rms not given as parameter and RMS keyword not present in data header")
                else:
                    print(f"Found RMS in header: {rms:.2E} {_error[0].data.shape}")
            # tmp = np.full(_error[0].data.shape,rms)
            _error[0].data[:] = rms
        elif "%" in error:
            percent = float(error.strip("%")) / 100.0
            _error = deepcopy(_data)
            _error[0].data = _data[0].data * percent
        else:
            _error = fits.open(error)
            needsclose = True

        fb = _data[0].header.get("bunit", str(unit))  # use str in case Unit was given
        eb = _error[0].header.get("bunit", str(unit))
        if fb != eb:
            raise Exception(f"BUNIT must be the same in both data {fb} and error {eb} maps")
        # Sigh, this is necessary since there is no mode available in
        # fits.open that will truncate an existing file for writing
        if overwrite and exists(outfile):
            remove(outfile)
        _out = fits.open(name=outfile, mode="ostream")
        _out.append(_data[0])
        _out[0].header["bunit"] = fb
        _out.append(_error[0])
        _out[1].header["extname"] = "UNCERT"
        _out[1].header["bunit"] = eb
        _out[1].header["utype"] = "StdDevUncertainty"
        if masknan:
            fmasked = ma.masked_invalid(_data[0].data)
            emasked = ma.masked_invalid(_error[0].data)
            final_mask = utils.mask_union([fmasked, emasked])
            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduMask = fits.ImageHDU(final_mask.astype(np.uint8), name="MASK")
            _out.append(hduMask)
        _out.writeto(outfile, overwrite=overwrite, output_verify="silentfix")
        _data.close()
        _out.close()
        if needsclose:
            _error.close()

    @property
    def value(self):
        """Return the underlying data array.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        return self.data

    @property
    def error(self):
        """Return the underlying error array.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        if self.uncertainty is None:
            return None
        return self.uncertainty.array

    @property
    def SN(self):
        """Return the signal to noise ratio (value/error).

        Returns
        -------
        :class:`numpy.ndarray`
        """
        if self.uncertainty is None:
            return None
        return self.value / self.error

    @property
    def id(self):
        """Return the string ID of this measurement, e.g., ``CO_10``.

        Returns
        -------
        str
        """
        return self._identifier

    @id.setter
    def id(self, value):
        """Set the string ID of this measurement, e.g., ``CO_10``.

        Parameters
        ----------
        value : str
            The identifier.
        """
        self._identifier = value

    @property
    def beam(self):
        """Return the beam parameters as astropy Quantities or None if beam is not set"""
        if "BMAJ" in self.header and self.header["BMAJ"] is not None:
            return [self.header["BMAJ"], self.header["BMIN"], self.header["BPA"]] * u.degree
        else:
            return None

    def is_ratio(self):
        """Indicate if this Measurement is a ratio.

        Looks for ``'/'`` past the first character of the identifier, e.g. ``"CII_158/CO_32"``.
        See also :func:`~pdrtpy.utils.helpers.is_ratio`.

        Returns
        -------
        bool
            True if the Measurement is a ratio, False otherwise.
        """
        return utils.is_ratio(self.id)  # pdrutils method

    @property
    def title(self):
        """A formatted title (e.g., LaTeX) that can be used in plotting.

        Returns
        -------
        str or None
        """
        return self._title

    @property
    def filename(self):
        """The FITS file that created this measurement, or None if it didn't originate from a file.

        Returns
        -------
        str or None
        """
        return self._filename

    def write(self, filename, **kwd):
        """Write this Measurement to a FITS file with value in 1st HDU and error in 2nd HDU.

        See :meth:`astropy.nddata.CCDData.write`.

        Parameters
        ----------
        filename : str
            Name of file.
        **kwd
            All additional keywords are passed to :py:mod:`astropy.io.fits`.
        """
        hdu = self.to_hdu()
        hdu.writeto(filename, **kwd)

    def _set_up_for_interp(self, kind="linear"):
        # @TODO this will always return nan if there are nan in the data.
        # See eg. https://stackoverflow.com/questions/35807321/scipy-interpolation-with-masked-data
        """
        We don't want to have to do a call to get a pixel value at a particular WCS every time it's needed.
        So make one call that converts the entire NAXIS1 and NAXIS2 to an array of world coordinates and stash that away
        so we can pass it to scipy.interp2d when needed
        """
        self._world_axis = utils.get_xy_from_wcs(self, quantity=False, linear=False)
        self._world_axis_lin = utils.get_xy_from_wcs(self, quantity=False, linear=True)
        self._interp_log = RegularGridInterpolator(self._world_axis, values=self.data.T, method=kind, bounds_error=True)
        self._interp_lin = RegularGridInterpolator(
            self._world_axis_lin, values=self.data.T, method=kind, bounds_error=True
        )

    def get_pixel(self, world_x, world_y):
        """Return the nearest pixel coordinates to the input world coordinates.

        Parameters
        ----------
        world_x : float
            The horizontal world coordinate.
        world_y : float
            The vertical world coordinate.
        """
        if self.wcs is None:
            raise Exception(f"No wcs in this Measurement {self.id}")
        return tuple(np.round(self.wcs.world_to_pixel_values(world_x, world_y)).astype(int))

    def get(self, world_x, world_y, log=False):
        """Get the value(s) at the given world coordinates.

        Parameters
        ----------
        world_x : float or array-like
            The x value in world units of naxis1.
        world_y : float or array-like
            The y value in world units of naxis2.
        log : bool, optional
            True if the input coords are logarithmic. Default: False.

        Returns
        -------
        float
            The value(s) of the Measurement at input coordinates.
        """
        if log:
            return float(self._interp_log((world_x, world_y)))
        else:
            return float(self._interp_lin((world_x, world_y)))

    @property
    def levels(self):
        if self.value.size != 1:
            raise Exception("This only works for Measurements with a single pixel")
        return np.array(
            [float(self.value[0] - self.error[0]), float(self.value[0]), float(self.value[0] + self.error[0])]
        )

    def _modify_id(self, other, op):
        """Handle ID string for arithmetic operations with Measurements or numbers.

        Parameters
        ----------
        other : :class:`Measurement` or number
            A Measurement or number.
        op : str
            Descriptive string of operation, e.g. ``"+"`` or ``"*"``.
        """
        if getattr(other, "id", None) is not None:
            return self.id + op + other.id
        else:
            return self.id

    def _modify_title_card(self, other, op) -> str:
        """
        Make a TITLE header card for a `Measurement` created from arithmetic
        operation on two other measurements.  The operation is assumed to be
        ``self op other``.
        Note: this modifies the header (meta) entry, not the `title` attribute (LaTeX string)

        Parameters
        ----------
        m1 : str
            The TITLE card or identifer string from the first measurement
        m2 : `Measurement` or number
            The `Measurement or number
        op : str
            The string description of the operand.  One of '+', '-', '/', '*'

        Returns
        -------
        str
            The new TITLE card or an empty string if a new TITLE could not be constructed

        """
        if getattr(other, "header", None) is not None:
            if "TITLE" in self.header and "TITLE" in other.header:
                return f"{self.header['TITLE']}{op}{other.header['TITLE']}"
        return ""

    def add(self, other):
        """Add this Measurement to another, propagating errors, units, and updating identifiers. Masks are logically or'd.

        Parameters
        ----------
        other : :class:`Measurement` or number
            A Measurement or number to add.
        """
        # need to do tricky stuff to preserve unit propogation.
        # super().add() does not work because it instantiates a Measurement
        # with the default unit "adu" and then units for the operation are
        # not conformable.  I blame astropy CCDData authors for making that
        # class so hard to subclass.
        z = CCDData.add(self, other, handle_mask=np.logical_or, handle_meta="first_found")
        z = Measurement(z, unit=z._unit)
        z._identifier = self._modify_id(other, "+")
        z.header["TITLE"] = self._modify_title_card(other, "+")
        z._unit = self.unit
        return z

    def subtract(self, other):
        """Subtract another Measurement from this one, propagating errors, units, and updating identifiers. Masks are logically or'd.

        Parameters
        ----------
        other : :class:`Measurement` or number
            A Measurement or number to subtract.
        """
        z = CCDData.subtract(self, other, handle_mask=np.logical_or, handle_meta="first_found")
        z = Measurement(z, unit=z._unit)
        z._identifier = self._modify_id(other, "-")
        z.header["TITLE"] = self._modify_title_card(other, "-")
        return z

    def multiply(self, other):
        """Multiply this Measurement by another, propagating errors, units, and updating identifiers. Masks are logically or'd.

        Parameters
        ----------
        other : :class:`Measurement` or number
            A Measurement or number to multiply.
        """
        z = CCDData.multiply(self, other, handle_mask=np.logical_or, handle_meta="first_found")
        z = Measurement(z, unit=z._unit)
        z._identifier = self._modify_id(other, "*")
        z.header["TITLE"] = self._modify_title_card(other, "*")
        return z

    def divide(self, other):
        """Divide this Measurement by another, propagating errors, units, and updating identifiers. Masks are logically or'd.

        Parameters
        ----------
        other : :class:`Measurement` or number
            A Measurement or number to divide by.
        """
        z = CCDData.divide(self, other, handle_mask=np.logical_or, handle_meta="first_found")
        z = Measurement(z, unit=z._unit)
        z._identifier = self._modify_id(other, "/")
        z.header["TITLE"] = self._modify_title_card(other, "/")
        return z

    def is_single_pixel(self):
        """Is this Measurement a single value?

        Returns
        -------
        bool
            True if a single value (pixel).
        """
        return self.data.size == 1

    def __add__(self, other):
        """Add this Measurement to another using + operator, propagating errors, units,  and updating identifiers"""
        z = self.add(other)
        return z

    def __sub__(self, other):
        """Subtract another Measurement from this one using - operator, propagating errors, units,  and updating identifiers"""
        z = self.subtract(other)
        return z

    def __mul__(self, other):
        """Multiply this Measurement by another using * operator, propagating errors, units,  and updating identifiers"""
        z = self.multiply(other)
        return z

    def __truediv__(self, other):
        """Divide this Measurement by another using / operator, propagating errors, units,  and updating identifiers"""
        z = self.divide(other)
        return z

    def __repr__(self):
        m = f"{np.squeeze(self.data)} +/- {np.squeeze(self.error)} {self.unit}"
        return m

    def __str__(self):
        return repr(self)

    def __format__(self, spec):
        # todo look more closely how Quantity does this
        # print("using __format__")
        if spec == "":
            return str(self)
        # this can't possibly be the way you are supposed to use this, but it works
        spec = "{:" + spec + "}"
        a = np.array2string(np.squeeze(self.data), formatter={"float": lambda x: spec.format(x)})
        b = np.array2string(np.squeeze(self.error), formatter={"float": lambda x: spec.format(x)})
        # this does not always work
        # a = np.vectorize(spec.__mod__,otypes=[np.float64])(self.data)
        # b = np.vectorize(spec.__mod__,otypes=[np.float64])(self.error)
        return f"{a} +/- {b} {self.unit}"

    def __getitem__(self, index):
        """Allows us to use [] to index into the data array"""
        return self._data[index]

    @staticmethod
    def from_table(filename, format="ipac", array=False):
        r"""Table file reader for Measurement class.
        Create one or more Measurements from a table.
        The input table header must contain the columns:

            *data* - the data value

            *uncertainty* - the error on the data, can be absolute error or percent. If percent, the header unit row entry for this column must be "%"

            *identifier* - the identifier of this Measurement which should match a model in the ModelSet you are using, e.g., "CII_158" for [C II] 158 $\\mu$m

        The following columns are optional:

            *bmaj* - beam major axis size

            *bmin* - beam minor axis size

            *bpa*  - beam position angle

        The table must specify the units of each column, e.g. a unit row in the header for IPAC format.  Leave column entry blank if unitless.  Units of value and error should be the same or conformable. Units must be transformable to a valid astropy.unit.Unit.

        Parameters
        ----------
        filename : str
            Name of table file.
        format : str, optional
            `Astropy Table format <https://docs.astropy.org/en/stable/io/unified.html#built-in-readers-writers>`_,
            e.g., ``ascii``, ``ipac``, ``votable``. Default is
            `IPAC format <https://docs.astropy.org/en/stable/api/astropy.io.ascii.Ipac.html#astropy.io.ascii.Ipac>`_.
        array : bool, optional
            If True, one Measurement is created per table row and a list is returned.
            If False, one Measurement containing all data points is returned, using
            the identifier and beam parameters of the first row.
            For :meth:`~pdrtpy.plot.modelplot.ModelPlot.phasespace`, use ``array=False``.
            Default: False.

        Returns
        -------
        :class:`~pdrtpy.measurement.Measurement` or list of :class:`~pdrtpy.measurement.Measurement`
        """
        # @todo support input of a astropy.Table directly
        t = Table.read(filename, format=format)
        required = ["data", "uncertainty", "identifier"]
        options = ["bmaj", "bmin", "bpa"]
        errmsg = ""
        for r in required:
            if r not in t.colnames:
                errmsg += f"{r} is a required column. "
        if errmsg != "":
            raise Exception(f"Insufficient information in table to create Measurement. {errmsg}")

        # check for beam parameters in table.
        # IFF all beam parameters present, they will be added to the Measurements.
        if sorted(list(set(options) & set(t.colnames))) == sorted(options):
            hasBeams = True
        else:
            hasBeams = False

        if t["data"].unit is None:
            t["data"].unit = ""
        if t["uncertainty"].unit is None:
            t["uncertainty"].unit = ""
        if array:
            a = list()
            for x in t:  # x is a astropy.table.row.Row
                if t.columns["uncertainty"].unit == "%":
                    err = StdDevUncertainty(array=x["uncertainty"] * x["data"] / 100.0, unit=t.columns["data"].unit)
                else:
                    err = StdDevUncertainty(array=x["uncertainty"], unit=t.columns["uncertainty"].unit)
                if hasBeams:
                    # NB: I tried to do something tricky here with Qtable, but it actually became *more* complicated
                    m = Measurement(
                        data=x["data"].data,
                        identifier=x["identifier"],
                        unit=t.columns["data"].unit,
                        uncertainty=err,
                        bmaj=x["bmaj"] * t.columns["bmaj"].unit,
                        bmin=x["bmin"] * t.columns["bmaj"].unit,
                        bpa=x["bpa"] * t.columns["bpa"].unit,
                    )
                else:
                    m = Measurement(
                        data=x["data"].data, identifier=x["identifier"], unit=t.columns["data"].unit, uncertainty=err
                    )
                a.append(m)
            return a
        else:
            if t.columns["uncertainty"].unit == "%":
                err = StdDevUncertainty(t["uncertainty"] * t["data"] / 100.0, unit=t.columns["data"].unit)
            else:
                err = StdDevUncertainty(t["uncertainty"], unit=t.columns["uncertainty"].unit)
            if hasBeams:
                m = Measurement(
                    data=t["data"].data,
                    identifier=t["identifier"][0],
                    unit=t.columns["data"].unit,
                    uncertainty=err,
                    bmaj=t["bmaj"][0] * t["bmaj"].unit,
                    bmin=t["bmin"][0] * t["bmaj"].unit,
                    bpa=t["bpa"][0] * t["bpa"].unit,
                )
            else:
                m = Measurement(
                    data=t["data"].data, identifier=t["identifier"][0], unit=t.columns["data"].unit, uncertainty=err
                )
            return m


def fits_measurement_reader(
    filename, hdu=0, unit=None, hdu_mask="MASK", hdu_flags=None, key_uncertainty_type="UTYPE", **kwd
):
    """FITS file reader for Measurement class, called by :meth:`Measurement.read`.

    Parameters
    ----------
    filename : str
        Name of FITS file.
    identifier : str, optional
        String indicating what this is an observation of, e.g., ``"CO_10"`` for CO(1-0).
    squeeze : bool, optional
        If True, remove single dimension axes from the input image. Default: True.
    hdu : int, optional
        FITS extension from which Measurement should be initialized. If zero
        and no data in the primary extension, searches for the first extension
        with data. Default: 0.
    unit : :class:`astropy.units.Unit`, optional
        Units of the image data. If provided and BUNIT is in the header, this
        argument takes precedence. Default: None.
    hdu_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized.
        If the extension does not exist, uncertainty is ``None``. Default: ``'UNCERT'``.
    hdu_mask : str or None, optional
        FITS extension from which the mask should be initialized.
        If the extension does not exist, mask is ``None``. Default: ``'MASK'``.
    hdu_flags : str or None, optional
        Currently not implemented. Default: None.
    key_uncertainty_type : str, optional
        Header key name where the uncertainty class name is stored. Default: ``'UTYPE'``.
    **kwd
        Additional keyword parameters passed to the FITS reader in :mod:`astropy.io.fits`.

    Raises
    ------
    TypeError
        If the conversion from CCDData to Measurement fails.
    """

    _id = kwd.pop("identifier", "unknown")
    _title = kwd.pop("title", None)
    _squeeze = kwd.pop("squeeze", True)
    _restfreq = kwd.pop("restfreq", None)
    # suppress INFO messages about units in FITS file. e.g. useless ones like:
    # "INFO: using the unit erg / (cm2 s sr) passed to the FITS reader instead of the unit erg s-1 cm-2 sr-1 in the FITS file."
    log.setLevel("WARNING")
    z = CCDData.read(filename, unit=unit)
    if _squeeze:
        z = utils.squeeze(z)

    try:
        z = Measurement(z, unit=z._unit, title=_title, restfreq=_restfreq)
    except Exception as exc:
        raise TypeError(f"Could not convert fits_measurement_reader output to Measurement because {exc}") from exc
    z.id = _id
    # astropy.io.registry.read creates a FileIO object before calling the registered
    # reader (this method), so the filename is FileIO.name.
    z._filename = filename.name
    return z


with registry.delay_doc_updates(Measurement):
    registry.register_reader("fits", Measurement, fits_measurement_reader)
