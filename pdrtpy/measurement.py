"""Manage spectral line or continuum observations"""
#@Todo it would be nice to be able to get Measurment[index] as a Measurement instead of
# a float. This is the behavior for CCDData, somehow lost in Measurement  See NDUncertainty __getitem__
# this will have ripple effects if implemented.
from copy import deepcopy
from os import remove
from os.path import exists

from astropy import log
import astropy.units as u
from astropy.io import fits,registry
from astropy.table import Table
from astropy.nddata import CCDData, StdDevUncertainty 
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp2d
from . import pdrutils as utils

class Measurement(CCDData):
    r"""Measurement represents one or more observations of a given spectral
    line or continuum.  It is made up of a value array, an
    uncertainty array, units, and a string identifier. It is based
    on :class:`astropy.nddata.CCDData`.  It can represent a single pixel
    observation or an image.   Mathematical operations using Measurements
    will correctly propagate errors.

    Typically, Measurements will be instantiated from a FITS file by using the the :func:`read` or :func:`make_measurement` methods.  For a list of recognized spectral line identifiers, see :meth:`~pdrtpy.modelset.Modelset.supported_lines`.

    :param data:  The actual data contained in this :class:`Measurement` object.
        Note that the data will always be saved by *reference*, so you should
        make a copy of the ``data`` before passing it in if that's the desired
        behavior.
    :type data: :class:`numpy.ndarray`-like

    :param uncertainty: Uncertainties on the data. If the uncertainty is a :class:`numpy.ndarray`, it assumed to be, and stored as, a :class:`astropy.nddata.StdDevUncertainty`.  Required.
    :type uncertainty: :class:`astropy.nddata.StdDevUncertainty`, \
            :class:`astropy.nddata.VarianceUncertainty`, \
            :class:`astropy.nddata.InverseVariance` or :class:`numpy.ndarray`

    :param unit: The units of the data.  Required.
    :type unit: :class:`astropy.units.Unit` or str

    :param identifier: A string indicating what this is an observation of, e.g., "CO_10" for CO(1-0)
    :type identifier: str

    :param title: A formatted string (e.g., LaTeX) describing this observation that can be used for plotting. Python r-strings are accepted, e.g., r'$^{13}$CO(3-2)'  would give :math:`^{13}{\rm CO(3-2)}`.
    :type title: str

    :param bmaj: [optional] beam major axis diameter. This will be converted to degrees for storage in FITS header
    :type  bmaj: :class:`astropy.units.Quantity`

    :param bmin: [optional] beam minor axis diameter. This will be converted to degrees for storage in FITS header
    :type  bmin: :class:`astropy.units.Quantity`

    :param bpa: [optional] beam position angle. This will be converted to degrees for storage in FITS header
    :type  bpa: :class:`astropy.units.Quantity`

    :raises TypeError: if beam parameters are not Quantities

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
    def __init__(self,*args, **kwargs):
        debug = kwargs.pop('debug', False)

        if debug:
            print("args=",*args)
            print("kwargs=",*kwargs)
        self._identifier = kwargs.pop('identifier', 'unknown')
        self._title      = kwargs.pop('title', None)
        _beam = dict()
        _beam["BMAJ"] = self._beam_convert(kwargs.pop('bmaj', None))
        _beam["BMIN"] = self._beam_convert(kwargs.pop('bmin', None))
        _beam["BPA"]  = self._beam_convert(kwargs.pop('bpa', None))
        self._restfreq = kwargs.pop('restfreq',None)
        self._filename = None

        #This won't work: On arithmetic operations, this raises the exception.
        #if self._identifier is None:
        #    raise ValueError("an identifier for Measurement must be specified.")
        #On arithmetic operations, this causes an annoying
        # log.info() message from CCDData about overwriting Quantity

        # This workaround is needed because CCDData raises an exception if unit
        # not given. Whereas having BUNIT in the image header instead would be
        # perfectly reasonable...
        # The side-effect of this is that Measurement not instantiated from
        # an image and with no unit given gets "adu" as the unit.
        self._defunit = "adu"
        unitpresent = 'unit' in kwargs
        _unit = kwargs.pop('unit', self._defunit)

        # Also works: super().__init__(*args, **kwargs, unit=_unit)
        CCDData.__init__(self,*args, **kwargs, unit=_unit)
        # force single pixel data to be interable arrays.
        # I consider this a bug in CCDData, StdDevUncertainty that they don't do this.
        # also StdDevUncertainty does not convert float to np.float!
        #print("DU",np.shape(self.data),np.shape(self.uncertainty.array))
        #print(type(self.data))
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

    def _beam_convert(self,bpar):
        if bpar is None:
            return bpar
        if type(bpar) == u.Quantity:
            return bpar.to("degree").value
        raise TypeError("Beam parameters must be astropy Quantities")


    @staticmethod
    def make_measurement(datafile,error,outfile,rms=None,masknan=True,overwrite=False,unit="adu"):
        """Create a FITS files with 2 HDUS, the first being the datavalue and the 2nd being
        the data uncertainty. This format makes allows the resulting file to be read into the underlying :class:'~astropy.nddata.CCDData` class.

        :param datafile: The FITS file containing the data as a function of spatial coordinates
        :type datafile: str
        :param error: The errors on the data Possible values for error are:

             - a filename with the same shape as datafile containing the error values per pixel
             - a percentage value 'XX%' must have the "%" symbol in it
             - 'rms' meaning use the rms parameter if given, otherwise look for the RMS keyword in the FITS header of the datafile

        :type error: str
        :param outfile: The output file to write the result in (FITS format)
        :type outfile: str
        :param rms:  If error == 'rms', this value may give the rms in same units as data (e.g 'erg s-1 cm-2 sr-1').
        :type rms: float or :class:`astropy.units.Unit`
        :param masknan: Whether to mask any pixel where the data or the error is NaN. Default:true
        :type masknan: bool
        :param overwrite: If `True`, overwrite the output file if it exists. Default: `False`.
        :type overwrite: bool
        :param unit: Intensity unit to use for the data, this will override BUNIT in header if present.
        :type unit: :class:`astropy.units.Unit` or str

        :raises Exception: on various FITS header issues
        :raises OSError: if `overwrite` is `False` and the output file exists.

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
        if error == 'rms':
            _error = deepcopy(_data)
            if rms is None:
                rms = _data[0].header.get("RMS",None)
                if rms is None:
                    raise Exception("rms not given as parameter and RMS keyword not present in data header")
                else:
                    print("Found RMS in header: %.2E %s"%(rms,_error[0].data.shape))
            #tmp = np.full(_error[0].data.shape,rms)
            _error[0].data[:] = rms
        elif "%" in error:
            percent = float(error.strip('%')) / 100.0
            _error = deepcopy(_data)
            _error[0].data = _data[0].data*percent
        else:
            _error = fits.open(error)
            needsclose = True

        fb = _data[0].header.get('bunit',str(unit)) #use str in case Unit was given
        eb = _error[0].header.get('bunit',str(unit))
        if fb != eb:
            raise Exception("BUNIT must be the same in both data (%s) and error (%s) maps"%(fb,eb))
        # Sigh, this is necessary since there is no mode available in
        # fits.open that will truncate an existing file for writing
        if overwrite and exists(outfile):
            remove(outfile)
        _out = fits.open(name=outfile,mode="ostream")
        _out.append(_data[0])
        _out[0].header['bunit'] = fb
        _out.append(_error[0])
        _out[1].header['extname']='UNCERT'
        _out[1].header['bunit'] = eb
        _out[1].header['utype'] = 'StdDevUncertainty'
        if masknan:
            fmasked = ma.masked_invalid(_data[0].data)
            emasked = ma.masked_invalid(_error[0].data)
            final_mask = utils.mask_union([fmasked,emasked])
            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduMask = fits.ImageHDU(final_mask.astype(np.uint8), name='MASK')
            _out.append(hduMask)
        _out.writeto(outfile,overwrite=overwrite)
        _data.close()
        _out.close()
        if needsclose: 
            _error.close()

    @property
    def value(self):
        '''Return the underlying data array

        :rtype: :class:`numpy.ndarray`
        '''
        return self.data
    @property
    def error(self):
        '''Return the underlying error array

        :rtype: :class:`numpy.ndarray`
        '''
        if self.uncertainty is None:
            return None
        return self.uncertainty._array

    @property
    def SN(self):
        '''Return the signal to noise ratio (value/error)

        :rtype: :class:`numpy.ndarray`
        '''
        if self.uncertainty is None:
            return None
        return self.value/self.error

    @property
    def id(self):
        '''Return the string ID of this measurement, e.g., CO_10

        :rtype: str
        '''
        return self._identifier

    def identifier(self,id):
        '''Set the string ID of this measurement, e.g., CO_10

        :param id: the identifier
        :type id: str
        '''
        self._identifier = id

    @property
    def beam(self):
        '''Return the beam parameters as astropy Quantities or None if beam is not set'''
        if "BMAJ" in self.header and self.header["BMAJ"] is not None:
            return [self.header["BMAJ"],self.header["BMIN"],self.header["BPA"]]*u.degree
        else:
            return None

    def is_ratio(self):
        '''Indicate if this `Measurement` is a ratio..
        This method looks for the '/' past the first character  of the` Measurement` *identifier*, such as "CII_158/CO_32"
        See also pdrutils.is_ratio(string)

        :returns: True if the Measurement is a ratio, False otherwise
        :rtype: bool'''
        return utils.is_ratio(self.id) #pdrutils method

    @property
    def title(self):
        '''A formatted title (e.g., LaTeX) that can be in plotting.

        :rtype: str or None
        '''
        return self._title

    @property
    def filename(self):
        '''The FITS file that created this measurement, or None if it didn't originate from a file

        :rtype: str or None
        '''
        return self._filename

    def write(self,filename,**kwd):
        '''Write this Measurement to a FITS file with value in 1st HDU and error in 2nd HDU. See :meth:`astropy.nddata.CCDData.write`.

        :param filename:  Name of file.
        :type filename: str
        :param kwd: All additional keywords are passed to :py:mod:`astropy.io.fits`
        '''
        hdu = self.to_hdu()
        hdu.writeto(filename,**kwd)

    def _set_up_for_interp(self,kind='linear'):
        #@TODO this will always return nan if there are nan in the data.
        # See eg. https://stackoverflow.com/questions/35807321/scipy-interpolation-with-masked-data
        """
        We don't want to have to do a call to get a pixel value at a particular WCS every time it's needed.
        So make one call that converts the entire NAXIS1 and NAXIS2 to an array of world coordinates and stash that away
        so we can pass it to scipy.interp2d when needed
        """
        self._world_axis = utils.get_xy_from_wcs(self,quantity=False,linear=False)
        self._world_axis_lin = utils.get_xy_from_wcs(self,quantity=False,linear=True)
        #print("M WORLD AXIS LOG: ",self._world_axis)
        #print("LEN WALOG",len(self._world_axis[0]),len(self._world_axis[1]))
        #print("M WORLD AXIS LIN: ",self._world_axis_lin)
        #print("LEN WALIN",len(self._world_axis_lin[0]),len(self._world_axis_lin[1]))
        self._interp_log = interp2d(self._world_axis[0],self._world_axis[1],z=self.data,kind=kind,bounds_error=True)
        self._interp_lin = interp2d(self._world_axis_lin[0],self._world_axis_lin[1],z=self.data,kind=kind,bounds_error=True)

    def get_pixel(self,world_x,world_y):
        '''Return the nearest pixel coordinates to the input world coordinates

        :param world_x: The horizontal world coordinate
        :type world_x: float
        :param world_y: The vertical world coordinate
        :type world_y: float
        '''
        if self.wcs is None:
            raise Exception(f"No wcs in this Measurement {self.id}")
        return tuple(np.round(self.wcs.world_to_pixel_values(world_x,world_y)).astype(int))

    def get(self,world_x,world_y,log=False):
        """Get the value(s) at the give world coordinates

        :param world_x: the x value in world units of naxis1
        :type world_x: float or array-like
        :param world_y: the y value in world units of naxis2
        :type world_y: float or array-lke
        :param log: True if the input coords are logarithmic Default:False
        :type log: bool
        :returns: The value(s) of the Measurement at input coordinates
        :rtype: float
        """
        if log:
            return self._interp_log(world_x,world_y)
        else:
            return self._interp_lin(world_x,world_y)

    @property
    def levels(self):
        if self.value.size != 1:
            raise Exception("This only works for Measurements with a single pixel")
        return np.array([float(self.value-self.error),float(self.value),float(self.value+self.error)])

    def _modify_id(self,other,op):
        """Handle ID string for arithmetic operations with Measurements or numbers
        :param other: a Measurement or number
        :type other: :class:`Measurement` or number
        :param op: descriptive string of operation, e.g. "+", "*"
        :type op: str
        """
        if getattr(other,"id", None) is not None:
            return self.id + op + other.id
        else:
            return self.id

    def add(self,other):
        """Add this Measurement to another, propagating errors, units,  and updating identifiers.  Masks are logically or'd.

        :param other: a Measurement or number to add
        :type other: :class:`Measurement` or number
        """
        # need to do tricky stuff to preserve unit propogation.
        # super().add() does not work because it instantiates a Measurement
        # with the default unit "adu" and then units for the operation are
        # not conformable.  I blame astropy CCDData authors for making that
        # class so hard to subclass.
        z=CCDData.add(self,other,handle_mask=np.logical_or)
        z=Measurement(z,unit=z._unit)
        z._identifier = self._modify_id(other,'+')
        z._unit = self.unit
        return z

    def subtract(self,other):
        '''Subtract another Measurement from this one, propagating errors, units,  and updating identifiers.  Masks are logically or'd.

        :param other: a Measurement or number to subtract
        :type other: :class:`Measurement` or number
        '''
        z=CCDData.subtract(self,other,handle_mask=np.logical_or)
        z=Measurement(z,unit=z._unit)
        z._identifier = self._modify_id(other,'-')
        return z

    def multiply(self,other):
        '''Multiply this Measurement by another, propagating errors, units,  and updating identifiers.  Masks are logically or'd.

        :param other: a Measurement or number to multiply
        :type other: :class:`Measurement` or number
        '''
        z=CCDData.multiply(self,other,handle_mask=np.logical_or)
        z=Measurement(z,unit=z._unit)
        z._identifier = self._modify_id(other,'*')
        return z

    def divide(self,other):
        '''Divide this Measurement by another, propagating errors, units,  and updating identifiers.  Masks are logically or'd.

        :param other: a Measurement or number to divide by
        :type other: :class:`Measurement` or number
        '''
        z=CCDData.divide(self,other,handle_mask=np.logical_or)
        z=Measurement(z,unit=z._unit)
        z._identifier = self._modify_id(other,'/')
        return z

    def is_single_pixel(self):
        ''' Is this Measurement a single value?
        :returns: True if a single value (pixel)
        :rtype: bool
        '''
        return self.data.size == 1

    def __add__(self,other):
        '''Add this Measurement to another using + operator, propagating errors, units,  and updating identifiers'''
        z=self.add(other)
        return z
    def __sub__(self,other):
        '''Subtract another Measurement from this one using - operator, propagating errors, units,  and updating identifiers'''
        z=self.subtract(other)
        return z

    def __mul__(self,other):
        '''Multiply this Measurement by another using * operator, propagating errors, units,  and updating identifiers'''
        z=self.multiply(other)
        return z

    def __truediv__(self,other):
        '''Divide this Measurement by another using / operator, propagating errors, units,  and updating identifiers'''
        z=self.divide(other)
        return z

    def __repr__(self):
        m = "%s +/- %s %s" % (np.squeeze(self.data),np.squeeze(self.error),self.unit)
        return m


    def __str__(self):
        # this fails for array data
        #return  "{:3.2e} +/- {:3.2e} {:s}".format(self.data,self.error,self.unit)
        # m = "%s +/- %s %s" % (self.data,self.error,self.unit)
        m = "%s +/- %s %s" % (np.squeeze(self.data),np.squeeze(self.error),self.unit)
        return m

    def __format__(self,spec):
        #todo look more closely how Quantity does this
        #print("using __format__")
        if spec=="":
            return str(self)
        # this can't possibly be the way you are supposed to use this, but it works
        spec = "{:"+spec+"}"
        a = np.array2string(np.squeeze(self.data), formatter={'float': lambda x: spec.format(x)})
        b = np.array2string(np.squeeze(self.error), formatter={'float': lambda x: spec.format(x)})
        # this does not always work
        # a = np.vectorize(spec.__mod__,otypes=[np.float64])(self.data)
        #b = np.vectorize(spec.__mod__,otypes=[np.float64])(self.error)
        return "%s +/- %s %s" % (a,b,self.unit)

    def __getitem__(self,index):
        '''Allows us to use [] to index into the data array
        '''
        return self._data[index]

    @staticmethod
    def from_table(filename,format='ipac',array=False):
        r'''Table file reader for Measurement class.
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

        :param filename: Name of table file.
        :type filename: str
        :param format: `Astropy Table format <https://docs.astropy.org/en/stable/table/io.html>`_ Supported formats are ascii, ipac, votable. Default is `IPAC format  <https://docs.astropy.org/en/stable/api/astropy.io.ascii.Ipac.html#astropy.io.ascii.Ipac>`_
        :type format: str
        :param array: Controls whether a list of Measurements or a single Measurement is returned. If `array` is True,  one Measurement instance will be created for each row in the table and a Python list of Measurements will be returned.  If `array` is False,  one Measurement containing all the points in the `data` member will be returned. If `array` is False, the *identifier* and beam parameters of the first row will be used. If feeding the return value to a plot method such as :meth:`~pdrtpy.plot.modelplot.ModelPlot.phasespace`, choose `array=False`. Default:False.
        :type array: bool

        :rtype: :class:`~pdrtpy.measurement.Measurement` or list of :class:`~pdrtpy.measurement.Measurement`
        '''
        #@todo support input of a astropy.Table directly
        t = Table.read(filename,format=format)
        required = ["data","uncertainty","identifier"]
        options = ["bmaj","bmin","bpa"]
        errmsg = ""
        for r in required:
            if r not in t.colnames:
                errmsg += "{0} is a required column. ".format(r)
        if errmsg != "":
            raise Exception("Insufficient information in table to create Measurement. {0}".format(errmsg))

        # check for beam parameters in table.
        # IFF all beam parameters present, they will be added to the Measurements.
        if sorted(list(set(options)& set(t.colnames))) == sorted(options):
            hasBeams = True
        else:
            hasBeams = False

        if t["data"].unit is None:
            t["data"].unit = ""
        if t["uncertainty"].unit is None:
            t["uncertainty"].unit = ""
        if array:
            a = list()
            for x in t: # x is a astropy.table.row.Row
                if t.columns["uncertainty"].unit == "%":
                    err = StdDevUncertainty(array=x["uncertainty"]*x["data"]/100.0,unit=t.columns["data"].unit)
                else:
                    err = StdDevUncertainty(array=x["uncertainty"],unit=t.columns["uncertainty"].unit)
                if hasBeams:
                    # NB: I tried to do something tricky here with Qtable, but it actually became *more* complicated
                    m = Measurement(data=x["data"].data,identifier=x["identifier"],
                                unit=t.columns["data"].unit,
                                uncertainty=err,
                                bmaj=x["bmaj"]*t.columns["bmaj"].unit,
                                bmin=x["bmin"]*t.columns["bmaj"].unit,
                                bpa=x["bpa"]*t.columns["bpa"].unit)
                else:
                    m = Measurement(data=x["data"].data,identifier=x["identifier"],
                                unit=t.columns["data"].unit,
                                uncertainty=err)
                a.append(m)
            return a
        else:
            if t.columns["uncertainty"].unit == "%":
                err = StdDevUncertainty(t["uncertainty"]*t["data"]/100.0,unit=t.columns["data"].unit)
            else:
                err = StdDevUncertainty(t["uncertainty"],unit=t.columns["uncertainty"].unit)
            if hasBeams:
                m = Measurement(data=t["data"].data,identifier=t["identifier"][0],
                                unit=t.columns["data"].unit,
                                uncertainty=err,
                                bmaj=t["bmaj"][0]*t["bmaj"].unit,
                                bmin=t["bmin"][0]*t["bmaj"].unit,
                                bpa=t["bpa"][0]*t["bpa"].unit)
            else:
                m = Measurement(data=t["data"].data,identifier=t["identifier"][0],
                                unit=t.columns["data"].unit,
                                uncertainty=err)
            return m


def fits_measurement_reader(filename, hdu=0, unit=None,
                        hdu_mask='MASK', hdu_flags=None,
                        key_uncertainty_type='UTYPE', **kwd):
    '''FITS file reader for Measurement class, which will be called by :meth:`Measurement.read`.

    :param filename: Name of FITS file.
    :type filename: str

    :param identifier: string indicating what this is an observation of, e.g., "CO_10" for CO(1-0)
    :type identifier: str

    :param squeeze: If ``True``, remove single dimension axes from the input image. Default: ``True``
    :type squeeze: bool

    :param hdu: FITS extension from which Measurement should be initialized.
         If zero and and no data in the primary extension, it will
         search for the first extension with data. The header will be
         added to the primary header.  Default is 0.
    :type hdu: int, optional

    :type unit: :class:`astropy.units.Unit`, optional
    :param unit:
         Units of the image data. If this argument is provided and there is a
         unit for the image in the FITS header (the keyword ``BUNIT`` is used
         as the unit, if present), this argument is used for the unit.
         Default is ``None``.

    :type hdu_uncertainty: str or None, optional
    :param hdu_uncertainty: FITS extension from which the uncertainty
         should be initialized. If the extension does not exist the
         uncertainty of the Measurement is ``None``.  Default is
         ``'UNCERT'``.

    :type hdu_mask: str or None, optional
    :param hdu_mask: FITS extension from which the mask should be initialized. If the extension does not exist the mask of the Measurement is ``None``.  Default is ``'MASK'``.

    :type hdu_flags: str or None, optional
    :param hdu_flags: Currently not implemented.  Default is ``None``.

    :type key_uncertainty_type: str, optional
     :param key_uncertainty_type: The header key name where the class name of the uncertainty  is stored in the hdu of the uncertainty (if any).  Default is ``UTYPE``.


    :param kwd: Any additional keyword parameters are passed through to the FITS reader in :mod:`astropy.io.fits`

    :raises TypeError: If the conversion from CCDData to Measurement fails
    '''

    _id = kwd.pop('identifier', 'unknown')
    _title = kwd.pop('title', None)
    _squeeze = kwd.pop('squeeze', True)
    # suppress INFO messages about units in FITS file. e.g. useless ones like:
    # "INFO: using the unit erg / (cm2 s sr) passed to the FITS reader instead of the unit erg s-1 cm-2 sr-1 in the FITS file."
    log.setLevel('WARNING')
    z = CCDData.read(filename,unit=unit)#,hdu,uu,hdu_uncertainty,hdu_mask,hdu_flags,key_uncertainty_type, **kwd)
    if _squeeze:
        z = utils.squeeze(z)

    # @TODO if uncertainty plane not present, look for RMS keyword
    # @TODO header values get stuffed into WCS, others may be dropped by CCDData._generate_wcs_and_update_header
    try:
        z=Measurement(z,unit=z._unit,title=_title)
    except Exception:
        raise TypeError('could not convert fits_measurement_reader output to Measurement')
    z.identifier(_id)
    # astropy.io.registry.read creates a FileIO object before calling the registered
    # reader (this method), so the filename is FileIO.name.
    z._filename=filename.name
    log.setLevel('INFO') # set back to default
    return z



with registry.delay_doc_updates(Measurement):
    registry.register_reader('fits', Measurement, fits_measurement_reader)
