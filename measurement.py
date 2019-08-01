#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy

import astropy.units as u
from astropy.io import fits,registry
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance
import numpy as np

class Measurement(CCDData):
    '''Measurement represents one or more observations of a given spectral
       line or continuum.  It is made up of a value array, an
       uncertainty array, units, and a string identifier It is based
       on `~astropy.nddata.CCDData`.  It can represent a single pixel
       observation or an image.   Mathematical operations using Measurements
       will correctly propagate errors.
       
    Parameters
    -----------
    data :  `numpy.ndarray`-like
        The actual data contained in this `Measurement` object.
        Note that the data will always be saved by *reference*, so you should
        make a copy of the ``data`` before passing it in if that's the desired
        behavior.

    uncertainty : `~astropy.nddata.StdDevUncertainty`, \
            `~astropy.nddata.VarianceUncertainty`, \
            `~astropy.nddata.InverseVariance`, `numpy.ndarray` or 
        Uncertainties on the data. If the uncertainty is a `numpy.ndarray`, it
        it assumed to be, and stored as, a `~astropy.nddata.StdDevUncertainty`.
        Required.

    unit : `~astropy.units.Unit` or str. Required.
        The units of the data.
        
    identifier  : string indicating what this is an observation of, 
                  e.g., "CO(1-0)"
    
    Methods
    -------
    read(\\*args, \\**kwargs)
        ``Classmethod`` to create an Measurement instance based on a ``FITS`` file.
        This method uses :func:`fits_measurement_reader` with the provided
        parameters.

    '''
    def __init__(self,*args, **kwargs):
        debug = kwargs.pop('debug', False)
        if debug: 
            print("args=",*args)
            print("kwargs=",*kwargs)
        self._identifier = kwargs.pop('identifier', 'unknown')
        #print('calling init')

        #On arithmetic operations, this raises the exception. 
        #if self._identifier is None:
        #    raise ValueError("an identifier for Measurement must be specified.")
        #On arithmetic operations, this causes an annoying 
        # log.info() message from CCDData about overwriting Quantity 
        #_unit = kwargs.pop('unit', 'adu')

        super().__init__(*args, **kwargs)#, unit=_unit)
        
    @property
    def flux(self):
        '''Return the underlying flux data array'''
        return self.data

    @property
    def error(self):
        '''Return the underlying error array'''
        return self.uncertainty._array
    
    @property
    def id(self):
        '''Return the string ID of this measurement, e.g. "CO(1-0)""'''
        return self._identifier

    def identifier(self,id):
        '''Set the string ID of this measurement, e.g. "CO(1-0)""'''
        self._identifier = id

    @property
    def levels(self):
        return np.array([np.float(self.flux-self.error),np.float(self.flux),np.float(self.flux+self.error)])

    
    def add(self,other):
        '''Add this Measurement to another, propagating errors, units,  and updating identifiers'''
        z=super().add(other)
        z._identifier = self.id + '+' + other.id
        z._unit = self.unit
        return z
   
    def subtract(self,other):
        '''Subtract another Measurement from this one, propagating errors, units,  and updating identifiers'''
        z=super().subtract(other)
        z._identifier = self.id + '-' + other.id
        z._unit = self.unit
        return z
    
    def multiply(self,other):
        '''Multiply this Measurement by another, propagating errors, units,  and updating identifiers'''
        z=super().multiply(other)
        z._identifier = self.id + '*' + other.id
        z._unit = self.unit*self.unit
        return z
        
    def divide(self,other):
        '''Divide this Measurement by another, propagating errors, units,  and updating identifiers'''
        z=super().divide(other)
        z._identifier = self.id + '/' + other.id
        z._unit = self.unit/self.unit
        return z
    
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
        m = "%s +/- %s %s" % (self.data,self.error,self.unit)
        return m
    def __str__(self):
        m = "%s +/- %s %s" % (self.data,self.error,self.unit)
        return m

def fits_measurement_reader(filename, hdu=0, unit=None, 
                        hdu_uncertainty='UNCERT',
                        hdu_mask='MASK', hdu_flags=None,
                        key_uncertainty_type='UTYPE', **kwd):
    _id = kwd.pop('identifier', 'unknown')
    z = CCDData.read(filename,hdu,unit,hdu_uncertainty,hdu_mask,key_uncertainty_type, **kwd)
    try:
       z = Measurement(z)
    except Exception:
       raise TypeError('could not convert fits_measurement_reader output to Measurement')
    z.identifier(_id)
    return z

with registry.delay_doc_updates(Measurement):
    registry.register_reader('fits', Measurement, fits_measurement_reader)
    

if __name__ == "__main__":

    m1 = Measurement(data=[30.,20.],uncertainty = StdDevUncertainty([5.,5.]),identifier="OI_145",unit="adu")
    m2 = Measurement(data=10.,uncertainty = StdDevUncertainty(2.),identifier="CI_609",unit="adu")
    m3 = Measurement(data=10.,uncertainty = StdDevUncertainty(1.5),identifier="CO_21",unit="adu")
    m4 = Measurement(data=100.,uncertainty = StdDevUncertainty(10.),identifier="CII_158",unit="adu")

    print(m1/m2)
    print(m2/m3)
    print(m1*m2)
    print(m2/m4)
    print(m4*m3)
    print(m4+m3)
    print(m3-m1)

    print(m3.levels)
