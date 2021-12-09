
import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.io.fits.header import Header
import astropy.wcs as wcs
import astropy.units as u
from astropy.table import Table, Column
from astropy.nddata import NDData, StdDevUncertainty
import warnings
from lmfit import Parameters, fit_report
from lmfit.model import Model, ModelResult

from .. import pdrutils as utils
from ..measurement import Measurement

"""We need a class that can store fit objects in a data array but have all the nice WCS properties of NDData"""
class FitMap(NDData):
    def __init__(self, data, *args, **kwargs):
        debug = kwargs.pop('debug', False)
        if debug: 
            print("args=",*args)
            print("kwargs=",*kwargs)
        self._name = kwargs.pop('name',None)

        # NDData wants a nddata array so give it a fake one
        # and sub our object array afterwards
        _data = np.zeros(data.shape)
        super().__init__(_data,*args,**kwargs)
        self._data = data
        if np.shape(self._data) == ():
            self._data = np.array([self._data])
        
    @property 
    def name(self):
        return self._name
    
    def __getitem__(self,i):
        """get the value object at array index i"""
        return self._data[i]
