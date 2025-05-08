#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:42:40 2025

Convert Meudon line transition data files to pdrtpy format
@author: mpound
"""

import astropy.units as u
import numpy as np
from astropy.table import Column, vstack

import pdrtpy.pdrutils as util

# from astropy.table import Table

ext = "13c_o.dat"
mol = "13CO"
t = util.get_table(f"meudon/Lines/line_{ext}", format="ascii")
levels = util.get_table(f"meudon/Levels/level_{ext}", format="ascii", data_start=1)

if mol == "CO":
    # CO
    colnames = ["n", "nu", "nl", "dE", "A", "quant", "vu", "Ju", "vl", "Jl", "info", "freq", "freq unit"]
    levels.rename_columns(["col2", "col3", "col5", "col6"], ["gu", "Tu", "vu", "Ju"])
    print(f"Found {t.colnames}, {len(t.colnames)=}")
    t.rename_columns(t.colnames, colnames)
    t.remove_columns(["quant", "info"])
    replaceme = ["vu", "vl", "Ju", "Jl"]
    funit = list(set(t["freq unit"]))
    if len(funit) != 1:
        raise Exception(f"bad number of frequency units {len(funit)}")
    t.remove_columns(["n", "nu", "nl", "freq unit"])
    units = ["K", "1/s", "", "", "", "", funit[0]]
    i = 0
    for c in t.columns:
        t[c].unit = units[i]
        i += 1
    t["lambda"] = t["freq"].to("micron", equivalencies=u.spectral())
elif mol == "13CO":
    # 13CO
    #   n     nu     nl                E(K)         Aein(s-1)             quant:  Ju    Jl   info:        E(GHz)
    colnames = ["n", "nu", "nl", "dE", "A", "quant", "Ju", "Jl", "info", "freq", "freq unit"]
    print(f"Found {t.colnames}, {len(t.colnames)=}")
    t.rename_columns(t.colnames, colnames)
    t.remove_columns(["quant", "info"])
    levels.rename_columns(["col2", "col3", "col5"], ["gu", "Tu", "Ju"])
    # levels.remove_column("col6")
    levels["vu"] = 0
    t["vu"] = 0
    t["vl"] = 0
    replaceme = ["Ju", "Jl"]
    funit = list(set(t["freq unit"]))
    if len(funit) != 1:
        raise Exception(f"bad number of frequency units {len(funit)}")
    t.remove_columns(["n", "nu", "nl", "freq unit"])
    units = ["K", "1/s", "", "", funit[0], "", ""]
    i = 0
    for c in t.columns:
        t[c].unit = units[i]
        i += 1
    t["lambda"] = t["freq"].to("micron", equivalencies=u.spectral())
else:
    raise Exception(f"Don't know table for {mol}")

t["Tu"] = np.zeros(len(t))
t["Tu"].unit = "K"
t["gu"] = np.ones(len(t))
t["Line"] = "abcdefghijklmonpqrstuvwxyz"
t["Transition"] = "abcdefghijklmonpqrstuvwxyz"

for k in replaceme:
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
    t["Line"][i] = f"{mol}v{t['vu'][i]}-{t['vl'][i]}J{t['Ju'][i]}-{t['Jl'][i]}"
    t["Transition"][i] = f"v{t['vu'][i]}-{t['vl'][i]} J{t['Ju'][i]}-{t['Jl'][i]}"

# if t["vu"][i] > 0:
#     t["Line"][i] = f"COv{t['vu'][i]}-{t['vl'][i]}J{t['Ju'][i]}-{t['Jl'][i]}"
# else:
#     t["Line"][i] = f"CO{t['Ju'][i]}-{t['Jl'][i]}"

t["species"] = mol
if mol == "13CO":
    # add the v=10 data from Ilane
    tv1 = util.get_table("13co_transition_v10.tab", format="ascii.ipac")
    tv1["Transition"] = "abcdefghijklmonpqrstuvwxyz"
    x = Column(name="Line", data=["abcdefghijklmonpqrstuvwxyz"] * len(tv1), dtype=str)
    tv1.replace_column("Line", x)
    tv1["freq"] = tv1["lambda"].to(u.GHz, equivalencies=u.spectral())
    for i in range(len(tv1)):
        tv1["Line"][i] = f"{mol}v{tv1['vu'][i]}-{tv1['vl'][i]}J{tv1['Ju'][i]}-{tv1['Jl'][i]}"
        tv1["Transition"][i] = f"v{tv1['vu'][i]}-{tv1['vl'][i]} J{tv1['Ju'][i]}-{tv1['Jl'][i]}"
    tcopy = t.copy()
    t = vstack([t, tv1[t.colnames]])
t.write(f"{util.table_dir()}{mol.lower()}_transition.tab", format="ascii.ecsv", overwrite=True)
