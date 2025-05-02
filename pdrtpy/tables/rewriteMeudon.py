#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:42:40 2025

Convert Meudon line transition data files to pdrtpy format
@author: mpound
"""

import pdrtpy.pdrutils as util
import astropy.units as u
import numpy as np

# from astropy.table import Table

t = util.get_table("meudon/Lines/line_co.dat", format="ascii")
levels = util.get_table("meudon/Levels/level_co.dat", format="ascii", data_start=1)
colnames = ["n", "nu", "nl", "dE", "A", "quant", "vu", "Ju", "vl", "Jl", "info", "freq", "freq unit"]
t.rename_columns(t.colnames, colnames)
t.remove_columns(["quant", "info"])
funit = list(set(t["freq unit"]))
if len(funit) != 1:
    raise Exception(f"bad number of frequency units {len(funit)}")
units = ["", "", "", "K", "1/s", "", "", "", "", funit[0]]
t.remove_column("freq unit")
levels.rename_columns(["col2", "col3", "col5", "col6"], ["gu", "Tu", "vu", "Ju"])
i = 0
for c in t.columns:
    t[c].unit = units[i]
    i += 1

t["Tu"] = np.zeros(len(t))
t["Tu"].unit = "K"
t["gu"] = np.ones(len(t))
t["Line"] = "abcdefghijklmonpqrstuvwxyz"
t["Transition"] = "abcdefghijklmonpqrstuvwxyz"
for k in ["vu", "vl", "Ju", "Jl"]:
    for i in range(len(t)):
        s = t[k][i]
        # t[k].dtype = int
        s = s.replace(";", "")
        t[k][i] = s
    t[k] = t[k].astype(int)
for i in range(len(t)):
    levindex = np.where((levels["vu"] == t["vu"][i]) & (levels["Ju"] == t["Ju"][i]))
    if len(levindex) != 1:
        raise Exception(f"Bad {levindex=}")
    t["Tu"][i] = levels["Tu"][levindex[0]]
    t["gu"][i] = levels["gu"][levindex[0]]
    t["Line"][i] = f"COv{t['vu'][i]}-{t['vl'][i]}J{t['Ju'][i]}-{t['Jl'][i]}"
    t["Transition"][i] = f"v{t['vu'][i]}-{t['vl'][i]} J{t['Ju'][i]}-{t['Jl'][i]}"
# if t["vu"][i] > 0:
#     t["Line"][i] = f"COv{t['vu'][i]}-{t['vl'][i]}J{t['Ju'][i]}-{t['Jl'][i]}"
# else:
#     t["Line"][i] = f"CO{t['Ju'][i]}-{t['Jl'][i]}"
t["lambda"] = t["freq"].to("micron", equivalencies=u.spectral())
t["Species"] = "CO"
t.write(util.table_dir() + "12co_transition.tab", format="ascii.ecsv", overwrite=True)
