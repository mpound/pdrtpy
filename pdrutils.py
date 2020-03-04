#!/usr/bin/env python

# Utility code for PDR Toolbox

import datetime
import os.path
import sys

import astropy.units as u
from astropy.table import Table

_VERSION_ = "2.0-Beta"

# Radiation Field Strength units in cgs
_RFS_UNIT_ = u.erg/(u.second*u.cm*u.cm)

# ISRF in other units
habing_unit = u.def_unit('Habing',1.6E-3*_RFS_UNIT_)
u.add_enabled_units(habing_unit)
draine_unit = u.def_unit('Draine',2.704E-3*_RFS_UNIT_)
u.add_enabled_units(draine_unit)
mathis_unit = u.def_unit('Mathis',2.028E-3*_RFS_UNIT_)
u.add_enabled_units(mathis_unit)
    
#See https://stackoverflow.com/questions/880530/can-modules-have-properties-the-same-way-that-objects-can
# only works python 3.8+??
def module_property(func):
    """Decorator to turn module functions into properties.
    Function names must be prefixed with an underscore."""
    module = sys.modules[func.__module__]

    def base_getattr(name):
        raise AttributeError(
            f"module '{module.__name__}' has no fucking attribute '{name}'")

    old_getattr = getattr(module, '__getattr__', base_getattr)

    def new_getattr(name):
        if f'_{name}' == func.__name__:
            return func()
        else:
            return old_getattr(name)


    module.__getattr__ = new_getattr
    return func

#@module_property
#def _version():
def version():
    '''Version of the PDRT code'''
    return _VERSION_

#@module_property
#def _now():
def now():
    '''Return a string representing the current date and time in ISO format'''
    return datetime.datetime.now().isoformat()

#@module_property
#def _root_dir():
def root_dir():
    '''Project root directory, including trailing slash'''
    return os.path.dirname(os.path.abspath(__file__)) + '/'

#@module_property
#def _model_dir():
def model_dir():
    '''Project model directory, including trailing slash'''
    return os.path.join(root_dir(),'models/')

def table_dir():
    '''Project ancillary tables directory, including trailing slash'''
    return os.path.join(root_dir(),'tables/')

def _tablename(filename):
    '''Return fully qualified path of the input table.
       Parameters:
          filename - input table file name
    '''
    return table_dir()+filename


def get_table(filename,format='ipac',path=None):
    '''Return an astropy Table read from the input filename.  
       is 'ipac'
       Parameters:
          filename - input filename, no path
          format - file format, Default: ipac
          path - path to filename relative to models directory.  Default of None means look in "tables" directory 
    '''
    if path is None:
        return Table.read(_tablename(filename),format=format)
    else:
        return Table.read(model_dir()+path+filename,format=format)

def firstkey(d):
    """Return the 'first' key in a dictionary
       Parameters:
           d - the dictionary
    """
    return list(d)[0]

#@module_property
#def _wolfire():
def wolfire():
    '''Wolfire/Kaufman models'''
    return get_table("wolfire_models.tab")

def kosmatau():
    '''KOSMA TAU models'''
    return get_table("kosmatau_models.tab")

def smcmodels():
    '''Wolfire models for Small Magellanic Cloud'''
    return get_table("smc_models.tab")

def check_units(input_unit,compare_to):
    '''Return True if the input unit is equivalent to compare unit 
       
       Parameters:
          input_unit - astropy.Unit, astropy.Quanitity or string describing the unit to check.
          compare_to - astropy.Unit, astropy.Quanitity or string describing the unit to check against.

       Returns:
          True if units are equivalent, False otherwise
    '''
    if isinstance(input_unit,u.Unit):
        test_unit = input_unit
    if isinstance(input_unit,u.Quantity):
        test_unit = input_unit.unit
    else: # assume it is a string
        test_unit = u.Unit(input_unit)

    if isinstance(compare_to,u.Unit):
        compare_unit = compare_to
    if isinstance(compare_to,u.Quantity):
        compare_unit = compare_to.unit
    else: # assume it is a string
        compare_unit = u.Unit(compare_to)

    return test_unit.is_equivalent(compare_unit)

################################################################
# Conversions between various units of Radiation Field Strength
# See table on page 18 of 
# https://ism.obspm.fr/files/PDRDocumentation/PDRDoc.pdf
################################################################

def to(unit,image):
  '''Convert the image values to another unit.
     While generally this is intended for converting radiation field
     strength maps between Habing, Draine, cgs, etc, it will work for
     any image that has a unit member variable. So, e.g., it would work
     to convert density from cm^-3 to m^-3.

     Parameters:
        unit - the string or `astropy.units.Unit` identifying the unit to
        convert to 

        image - the image to convert. It must have a `numpy.ndarray`
        data member and `astropy.units.Unit` unit member.

     Returns:
        an image with converted values and units
  '''
  value = image.unit.to(unit)
  newmap = image.copy()
  newmap.data = newmap.data * value
  newmap.unit = u.Unit(unit)
  return newmap

def toHabing(image):
  '''Convert a radiation field strength image to Habing units (G_0).
     1 Habing = 1.6E-3 erg/s/cm^2
     See table on page 18 of 
     https://ism.obspm.fr/files/PDRDocumentation/PDRDoc.pdf
  '''
  return to('Habing',image)
   
def toDraine(image):
  '''Convert a radiation field strength image to Draine units (\chi).
     1 Draine = 2.704E-3 erg/s/cm^2
     See table on page 18 of 
     https://ism.obspm.fr/files/PDRDocumentation/PDRDoc.pdf
  '''
# 1 Draine = 1.69 G0 (Habing)\n",
  return to('Draine',image)

def toMathis(image):
  '''Convert a radiation field strength image to Mathis units
     1 Mathis = 2.028E-3 erg/s/cm^2
     See table on page 18 of 
     https://ism.obspm.fr/files/PDRDocumentation/PDRDoc.pdf
  '''
  return to('Mathis',image)

def tocgs(image):
  '''Convert a radiation field strength image to erg/s/cm^2'''
  return to(_RFS_UNIT_,image)

# use if current_models.tab is unavailable
def _make_default_table():
    ratiodict = {
    "OI_145/OI_63"   : "oioi",
    "OI_145/CII_158" : "o145cii",
    "OI_63/CII_158"  : "oicp",
    "CII_158/CI_609" : "ciici609",
    "CI_370/CI_609"  : "cici",
    "CII_158/CO_10"  : "ciico",
    "CI_609/CO_10"   : "cico",
    "CI_609/CO_21"   : "cico21",
    "CI_609/CO_32"   : "cico32",
    "CI_609/CO_43"   : "cico43",
    "CI_609/CO_54"   : "cico54",
    "CI_609/CO_65"   : "cico65",
    "CO_21/CO_10"    : "co2110",
    "CO_32/CO_10"    : "co3210",
    "CO_32/CO_21"    : "co3221",
    "CO_43/CO_21"    : "co4321",
    "CO_65/CO_10"    : "co6510",
    "CO_65/CO_21"    : "co6521",
    "CO_65/CO_54"    : "co6554",
    "CO_76/CO_10"    : "co7610",
    "CO_76/CO_21"    : "co7621",
    "CO_76/CO_43"    : "co7643",
    "CO_76/CO_54"    : "co7654",
    "CO_76/CO_65"    : "co7665",
    "CO_87/CO_54"   : "co8754",
    "CO_87/CO_65"   : "co8765",
    "CO_98/CO_54"   : "co9854",
    "CO_98/CO_65"   : "co9865",
    "CO_109/CO_54"   : "co10954",
    "CO_109/CO_65"   : "co10965",
    "CO_1110/CO_54"   : "co111054",
    "CO_1110/CO_65"   : "co111065",
    "CO_1211/CO_54"   : "co121154",
    "CO_1211/CO_65"   : "co121165",
    "CO_1312/CO_54"   : "co131254",
    "CO_1312/CO_65"   : "co131265",
    "CO_1413/CO_54"   : "co141354",
    "CO_1413/CO_65"   : "co141365",
    "OI_63+CII_158/FIR"     : "fir",
    "OI_145+CII_158/FIR"  : "firoi145",
    "SIII_Z1/FEII_Z1"  : "siii35feii26z1",
    "SIII_Z3/FEII_Z3"  : "siii35feii26z3",
    "H200S1_Z1/H200S0_Z1" : "h200s1s0z1",
    "H200S1_Z3/H200S0_Z3" : "h200s1s0z3",
    "H200S2_Z1/H200S0_Z1" : "h200s2s0z1",
    "H200S2_Z3/H200S0_Z3" : "h200s2s0z3",
    "H200S2_Z1/H200S1_Z1" : "h200s2s1z1",
    "H200S2_Z3/H200S1_Z3" : "h200s2s1z3",
    "H200S3_Z1/H200S1_Z1" : "h200s3s1z1",
    "H200S3_Z3/H200S1_Z3" : "h200s3s1z3",
    "H200S1_Z1/SIII_Z1" : "h200s1siiiz1",
    "H200S1_Z3/SIII_Z3" : "h200s1siiiz3",
    "H200S2_Z1/SIII_Z1" : "h200s2siiiz1",
    "H200S2_Z3/SIII_Z3" : "h200s2siiiz3",
    "H264Q1_Z1/H210S1_Z1" : "h264q110s1z1",
    "H264Q1_Z3/H210S1_Z3" : "h264q110s1z3"
    }
    b = list()
    for r in ratiodict:
        nd = r.split("/")
        if ("Z3" in r):
            z=3
        else:
            z=1
        b.append((nd[0],nd[1],r,ratiodict[r]+"web",z))
        
    t = Table(rows=b,names=("numerator","denominator","label","filename","z"))
    t.add_index("label")
    t.write("current_models.tab",format="ascii.ipac",overwrite=True)

