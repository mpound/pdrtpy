import itertools
import collections
import pdrutils as utils
import numpy as np

#@ToDo:
#   addModelSet() - for custom model sets. See model convention white paper
class ModelSet(object):
    """Class for PDR Model Sets. Model Sets will interface with a  
       directory containing the model FITS files.
    :param name: identifier.
    :type name: str.
    :param z:  metallicity in solar units.  
    :type z: float
    """
    def __init__(self,name,z):
        self._all_models = utils.get_table("all_models.tab")
        self._all_models.add_index("name")
        if name not in self._all_models["name"]:
            raise ValueError("Unrecognized model %s. Choices are: %s"%(name,self._possible_models))

        zzz = self._all_models.loc[name]["z"]
        self._row = np.where(zzz==z)
        if self._row[0].size == 0:
            raise ValueError("Z=%2.1f not found in %s. Choices are: %s"%(z,name,zzz))
        self._row = self._row[0][0]
        self._tabrow = self._all_models[self._row]
        self._table = utils.get_table(path=self._tabrow["path"],filename=self._tabrow["filename"])
        self._table.add_index("ratio")

    @property
    def description(self):
        """
        :returns: str -- The description of this model
        """
        return self._tabrow["description"]+", Z=%2.1f"%self.z

    @property
    def name(self):
        """
        :returns: str -- The name of this model
        """
        return self._tabrow["name"]

    @property
    def version (self):
        """
        :returns: str -- The version of this model
        """
        return self._tabrow["version"]

    @property
    def z(self):
        """
        :returns: float -- The metallicity of this model
        """
        return self.metallicity

    @property
    def metallicity(self):
        """
        :returns: float -- The metallicity of this model
        """
        return self._tabrow["z"]

    #@property
    #def isSolarMetallicity(self):
    #    return self._metallicity == 1

    @property
    def table(self):
        """
        :returns: :class:`astropy.table.Table` -- a Table containing details of the models
        """
        return self._table

    @property
    def supported_lines(self):
        """
        :returns: set -- Set of lines and continuum that are covered by this ModelSet
        """
        #TODO see below
        return set(np.append(self._table["numerator"].data,self._table["denominator"].data))

    @property
    def supported_ratios(self):
       """
       :returns: :class:`astropy.table.Column` -- The emission ratios that are covered by this model
       """
       #TODO: keep as Table column or as something else. Should be consistent with return type of supported_lines
       return self.table["ratio"]

    def ratiocount(self,m):
        """
        :param m: list of string Measurement IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: int -- The number of model ratios found for the given list 
           of measurement IDs
        """
        return len(self._get_ratio_elements(m))
        # Cute, but no longer needed:
        # Since find_files is a generator, we can't use len(), so do this sum.
        # See https://stackoverflow.com/questions/393053/length-of-generator-output
        #return(sum(1 for _ in self.find_files(m)))

    def find_pairs(self,m):
        """Find the valid model ratios labels in this ModelSet for a given list 
           of measurement IDs
        :param m: list of string Measurement IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: iterator -- An iterator of model ratios labels for the given list of measurement IDs
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
        """Find the valid model ratios files in this ModelSet for a given list 
           of measurement IDs
        :param m: list of string Measurement IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :param ext: file extenstion. Default: 'fits'
        :type ext: str
        :returns: iterator -- An iterator of model ratio files for the given list of Measurement IDs
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)):
            raise Exception("m must be an array of strings")
        for q in itertools.product(m,m):
            # must deal with OI+CII/FIR models. Note we must check for FIR first, since
            # if you check q has OI,CII and m has FIR order you'll miss OI/CII.
            if q[0] == "FIR" and (q[1] == "OI_145" or q[1] == "OI_63") and "CII_158" in m:
                s = q[1] + "+CII_158/" + q[0]
            else:
                s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                fullpath = self._tabrow["path"]+self.table.loc[s]["filename"]+"."+ext
                tup = (s,fullpath)
                yield(tup)
            
    
    def _find_ratio_elements(self,m):
        # TODO handle case of OI+CII/FIR so it is not special cased in lineratiofit.py
        """Find the valid model numerator,denominator pairs in this ModelSet for a given list of measurement IDs
        :param m: list of string Measurement IDs, e.g. ["CII_158","OI_145","FIR"]
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
        """Get the valid model numerator,denominator pairs in this ModelSet for a given list of measurement IDs
        :param m: list of string Measurement IDs, e.g. ["CII_158","OI_145","FIR"]
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
           case of ([O I] 63 micron + [C II] 158 micron)/I_FIR
        :param m: list of string Measurement IDs, e.g. ["CII_158","OI_145","FIR"]
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

    # ============= Static Methods =============
    @staticmethod
    def list():
        """List the names and descriptions of available models"""
        t = utils.get_table("all_models.tab")
        t.remove_column("path")
        t.remove_column("filename")
        t.pprint_all(align="<")

    @staticmethod
    def WolfireKaufman():
    #ToDo later upgrade to variable z when 2020 models are available
        """Easy way to get the latest Wolfire-Kaufman models, z=1
        :returns: :class:`ModelSet` 
        """
        return ModelSet("wk2006",z=1)

    @staticmethod
    def KosmaTau():
        """Easy way to get the latest KOSMA TAU models, z=1
        :returns: :class:`ModelSet` 
        """
        return ModelSet("kosmatau",z=1)
#
#    def has_metallicity(self,z):
#        """Check if this model set contains a given metallicity 
#           Parameters:
#              z  - metallicity in solar units.  
#           Returns: 
#           True if the model set has at least one model of this metallicity
#        """
#        return np.any(np.isin(z,self._table["z"]),axis=0)
#
