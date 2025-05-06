#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:11:41 2025
Retrieve the partition function tabulations for selected molecules from HITRAN database
See https://hitran.org/docs/iso-meta/
@author: mpound
"""

import wget
from astropy.table import Table
import pdrtpy.pdrutils as utils

baseurl = "https://hitran.org/"
info = "https://hitran.org/docs/iso-meta/"
# Molecule name and HITRAN ID number
mols = {"12C16O": 26, "13C16O": 27, "12C18O": 28, "12C17O": 29, "H2": 103}

for m in mols:
    file = f"data/Q/q{mols[m]}.txt"
    out = f"PartFun_{m}.txt"
    tabout = f"PartFun_{m}.tab"
    url = f"{baseurl}{file}"
    wget.download(url=url, out=out)
    t = Table.read(out, format="ascii")
    t.rename_columns(["col1", "col2"], ["T", "Q"])
    t["T"].unit = "K"
    t.meta["Comment"] = (
        f"Partition function tabulation for {m}. Retrieved from {url} on {utils.now()}. See also {info}."
    )
    t.write(tabout, overwrite=True, format="ascii.ecsv")
