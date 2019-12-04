#!/usr/bin/env python

# Utility code for PDR Toolbox

import datetime
import os.path
import sys

import astropy.units as u
from astropy.table import Table

_VERSION_ = "2.0-Beta"
    
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

def model_table(filename):
    return root_dir()+filename

#@module_property
#def _wolfire():
def wolfire():
    return model_table("current_models.tab")

def kosmatau():
    return model_table("kosmatau_models.tab")

def smcmodels():
    return model_table("smc_models.tab")

#########################################################
# Conversions between various units of ISRF
# See table on page 18 of 
# https://ism.obspm.fr/files/PDRDocumentation/PDRDoc.pdf
#########################################################

def toHabing(image):
  try:
    habing_unit = u.def_unit('Habing',1.6E-3*u.erg/(u.second*u.cm*u.cm))
    u.add_enabled_units(habing_unit)
  except ValueError:
    # already added. I don't know of any other way to check for this
    pass
  value = image.unit.to('Habing')
  newmap = image.copy()
  newmap.data = newmap.data * value
  newmap.unit = u.Unit('Habing')
  return newmap
   
def toDraine(image):
# 1 Draine = 1.69 G0 (Habing)\n",
  try:
    draine_unit = u.def_unit('Draine',2.704E-3*u.erg/(u.second*u.cm*u.cm))
    u.add_enabled_units(draine_unit)
  except ValueError:
    # already added. I don't know of any other way to check for this
    pass
  value = image.unit.to('Draine')
  newmap = image.copy()
  newmap.data = newmap.data * value
  newmap.unit = u.Unit('Draine')
  return newmap

def toMathis(image):
# 1 Mathis = 0.75*Draine
  try:
    mathis_unit = u.def_unit('Mathis',2.028E-3*u.erg/(u.second*u.cm*u.cm))
    u.add_enabled_units(mathis_unit)
  except ValueError:
    # already added. I don't know of any other way to check for this
    pass
  value = image.unit.to('Mathis')
  newmap = image.copy()
  newmap.data = newmap.data * value
  newmap.unit = u.Unit('Mathis')
  return newmap

def tocgs(image):
  value = image.unit.to(u.erg/(u.second*u.cm*u.cm))
  newmap = image.copy()
  newmap = newmap * value
  newmap.unit = u.Unit(u.erg/(u.second*u.cm*u.cm))

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

