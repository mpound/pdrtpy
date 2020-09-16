"""Manage pre-computed PDR models"""

import itertools
import collections
from copy import deepcopy
import numpy as np
from astropy.table import Table, unique, vstack
import astropy.units as u
from .pdrutils import get_table,model_dir,habing_unit,draine_unit,mathis_unit
from .measurement import Measurement

#@ToDo:
#   addModelSet() - for custom model sets. See model convention white paper
class ModelSet(object):
    """Class for computed PDR Model Sets. :class:`ModelSet` will interface with a directory containing the model FITS files.

    :param name: identifier.
    :type name: str
    :param z:  metallicity in solar units.  
    :type z: float
    :raises ValueError: If identifier or z not recognized/found.
    """
    def __init__(self,name,z):
        self._all_models = get_table("all_models.tab")
        self._all_models.add_index("name")
        if name not in self._all_models["name"]:
            raise ValueError("Unrecognized model %s. Choices are: %s"%(name,self._possible_models))

        matching_rows = np.where(self._all_models["z"]==z)
        possible_z =  self._all_models.loc[name]["z"]
        if matching_rows[0].size == 0:
            raise ValueError("Z=%2.1f not found in %s. Choices are: %s"%(z,name,possible_z.data))
        self._tabrow = self._all_models[matching_rows].loc[name]
        self._table = get_table(path=self._tabrow["path"],filename=self._tabrow["filename"])
        self._table.add_index("ratio")
        self._set_identifiers()
        self._set_ratios()

    @property
    def description(self):
        """The description of this model

        :rtype: str 
        """
        return self._tabrow["description"]+", Z=%2.1f"%self.z

    @property
    def name(self):
        """The name of this model

        :rtype: str 
        """
        return self._tabrow["name"]

    @property
    def version (self):
        """The version of this model

        :rtype: str
        """
        return self._tabrow["version"]

    @property
    def z(self):
        """The metallicity of this model

        :rtype: float
        """
        return self.metallicity

    @property
    def metallicity(self):
        """The metallicity of this model

        :rtype: float
        """
        return self._tabrow["z"]

    #@property
    #def isSolarMetallicity(self):
    #    return self._metallicity == 1

    @property
    def table(self):
        """The table containing details of the models

        :rtype: :class:`astropy.table.Table` 
        """
        return self._table

    @property
    def supported_lines(self):
        """Table of lines and continuum that are covered by this ModelSet

        :rtype: :class:`astropy.table.Table` 
        """
        return self._identifiers

    @property
    def supported_ratios(self):
        """The emission ratios that are covered by this ModelSet

        :rtype: :class:`astropy.table.Table` 
        """
        return self._supported_ratios

    def ratiocount(self,m):
        """The number of valid ratios in this ModelSet, given a list of observation (:class:`~pdrtpy.measurement.Measurement`) identifiers.

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: The number of model ratios found for the given list of measurement IDs
        :rtype: int
        """
        return len(self._get_ratio_elements(m))
        # Cute, but no longer needed:
        # Since find_files is a generator, we can't use len(), so do this sum.
        # See https://stackoverflow.com/questions/393053/length-of-generator-output
        #return(sum(1 for _ in self.find_files(m)))

    def find_pairs(self,m):
        """Find the valid model ratios labels in this ModelSet for a given list of measurement IDs

        :param m: list of string `Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: An iterator of model ratios labels for the given list of measurement IDs
        :rtype: iterator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
            
        for q in itertools.product(m,m):
            if q[0] == "FIR" and (q[1] == "OI_145" or q[1] == "OI_63") and "CII_158" in m:
                s = q[1] + "+CII_158/" + q[0]
            else:
                s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                yield(s)

    def find_files(self,m,ext="fits"):
        """Find the valid model ratios files in this ModelSet for a given list of measurement IDs.  See :meth:`~pdrtpy.measurement.Measurement.id`

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :param ext: file extension. Default: "fits"
        :type ext: str
        :returns: An iterator of model ratio files for the given list of Measurement IDs
        :rtype: iterator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)):
            raise Exception("m must be an array of strings")
        for q in itertools.product(m,m):
            # must deal with OI+CII/FIR models. Note we must check for FIR first, since
            # if you check q has OI,CII and m has FIR order you'll miss OI/CII.
            if q[0] == "FIR" and (q[1] == "OI_145" or q[1] == "OI_63") and "CII_158" in m:
                #print("SPECIAL doing %s %s"%(q[0],q[1]))
                s = q[1] + "+CII_158/" + q[0]
            else:
                #print("doing %s %s"%(q[0],q[1]))
                s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                fullpath = self._tabrow["path"]+self.table.loc[s]["filename"]+"."+ext
                tup = (s,fullpath)
                yield(tup)
            
    def model_ratios(self,m):
        ratios = list()
        for p in self.find_files(m):
            ratios.append(p[0])
        return ratios
    
    def get_models(self,identifiers,unit=u.dimensionless_unscaled,ext="fits"):
        #if identifier not in self._identifiers["ID"]:
        #    raise Exception("There is no model in ModelSet %s with the identifier %s"%(self.name,identifier))
        models=dict()
        d = model_dir()
        self._supported_ratios.remove_indices('ratio label')
        self._supported_ratios.add_index('ratio label')
        for k in self.model_ratios(identifiers):
            _thefile = d+self._tabrow["path"]+self.table.loc[k]["filename"]+"."+ext
            _title = self._supported_ratios.loc[k]['title']
            _model = Measurement.read(_thefile,unit=unit,title=_title)
            _wcs = _model.wcs
            if self.name == "wk2006" or self.name == "smc":
            # fix WK2006 model headers
                if _wcs.wcs.cunit[0] == "":
                    _model.header["CUNIT1"] = "cm^-3"
                    _wcs.wcs.cunit[0] = u.Unit("cm^-3")
                if _wcs.wcs.cunit[1] == "":
                    _model.header["CUNIT2"] = "Habing"
                    # Raises UnitScaleError:
                    # "The FITS unit format is not able to represent scales that are not powers of 10.  Multiply your data by 1.600000e-03."
                    # This causes all sorts of downstream problems.  Workaround in LineRatioFit.read_models().
                    #_model.wcs.wcs.cunit[1] = habing_unit
            else:
                # copy wcs cunit to header. used later.
                _model.header["CUNIT1"] = str(_wcs.wcs.cunit[0])
                _model.header["CUNIT2"] = str(_wcs.wcs.cunit[1])
            models[k] = _model

        return models


    def _find_ratio_elements(self,m):
        # TODO handle case of OI+CII/FIR so it is not special cased in lineratiofit.py
        """Find the valid model numerator,denominator pairs in this ModelSet for a given list of measurement IDs. See :meth:`~pdrtpy.measurement.Measurement.id`

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: iterator -- An dictionary iterator where key is numerator and value is denominator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
            
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            z = dict()
            if s in self.table["ratio"]:
                z={"numerator":self.table.loc[s]["numerator"],
                   "denominator":self.table.loc[s]["denominator"]}
                yield(z)

    def _get_ratio_elements(self,m):   
        """Get the valid model numerator,denominator pairs in this ModelSet for a given list of measurement IDs. See :meth:`~pdrtpy.measurement.Measurement.id`

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: dict -- A dictionary where key is numerator and value is denominator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
        k = list()   
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                z={"numerator":self.table.loc[s]["numerator"],
                   "denominator":self.table.loc[s]["denominator"]}
                k.append(z)
        self._get_oi_cii_fir(m,k)
        return k

    def _get_oi_cii_fir(self,m,k):
        """Utility method for determining ratio elements, to handle special 
           case of ([O I] 63 micron + [C II] 158 micron)/I_FIR.

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :param k: list of existing ratios to append to  
        :type k: list
        """
        if "CII_158" in m and "FIR" in m:
            if "OI_63" in m:
                num = "OI_63+CII_158"
                den = "FIR"
                #lab="OI_63+CII_158/FIR"
                z = {"numerator":num,"denominator":den}
                k.append(z)
            if "OI_145" in m:
                num = "OI_145+CII_158"
                den = "FIR"
                #lab="OI_145+CII_158/FIR"
                z = {"numerator":num,"denominator":den}
                k.append(z)

    def _set_ratios(self):
        """make a useful table of ratios covered by this model"""
        self._supported_ratios = Table( [ self.table["title"], self.table["ratio"] ],copy=True)
        self._supported_ratios['title'].unit = None
        self._supported_ratios['ratio'].unit = None
        self._supported_ratios.rename_column("ratio","ratio label")

    def _set_identifiers(self):
        """make a useful table of identifiers of lines covered by this model"""
        n=deepcopy(self._table['numerator'])
        n.name = 'ID'
        d=deepcopy(self._table['denominator'])
        d.name='ID'

        t1 = Table([self._table['title'],n],copy=True)
        # discard the summed fluxes as user would input them individually
        for id in ['OI_145+CII_158','OI_63+CII_158']:
            a = np.where(t1['ID']==id)[0]
            for z in a:
                t1.remove_row(z)
        # now remove denominator from title (everything from / onwards)
        for i in range(len(t1['title'])):
            t1['title'][i] = t1['title'][i][0:t1['title'][i].index('/')]
        
        t2 = Table([self._table['title'],d],copy=True)
        # remove numermator from title (everything before and including /)
        for i in range(len(t2['title'])):
            t2['title'][i] = t2['title'][i][t2['title'][i].index('/')+1:]
        t = vstack([t1,t2])
        t = unique(t,keys=['ID'],keep='first',silent=True)
        t['title'].unit = None
        t['ID'].unit = None
        t.rename_column('title','canonical name')
        self._identifiers = t
        

    # ============= Static Methods =============
    @staticmethod
    def list():
        """List the names and descriptions of available models (not just this one)"""
        t = get_table("all_models.tab")
        t.remove_column("path")
        t.remove_column("filename")
        t.pprint_all(align="<")
