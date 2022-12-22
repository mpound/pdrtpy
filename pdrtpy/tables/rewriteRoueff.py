#!/usr/bin/env python
# change Roueff et al H2 data table to IPAC format
from astropy.io import ascii
from astropy.constants import h,k_B,c

f = open("RoueffEtAltable2.dat","r")
header = f.readline().split('|')
hdvar = list()
# branch notation key=Ju-Jl
branch = {-2:"O",-1:"P",0:"Q",1:"R",2:"S"}
for i in range(1,len(header)-1):
    hdvar.append(header[i].strip())
#print(hdvar)
data = f.readlines()
f.close()
t = ascii.read(data)

units = [None,None,None,None, "cm-1","cm-1","micron","micron","s-1","s-1","s-1","s-1", "cm-1","cm-1","K",None]
i=0
for col in t.colnames:
    t[col].unit = units[i]
    t.rename_column(col,hdvar[i])
    i=i+1

# calculate energy of the transition from the wavenumber
# and add it as a column
dE = h*c*t["sigma"].to("1/m")/k_B
t["dE"] = dE

# create transition notation
diff =  t["Ju"]-t["Jl"]
b = [branch[i] for i in diff]
line = list()
trans= list()
for i in range(len(t)):
    line.append(f"H2{t['vu'][i]}{t['vl'][i]}{b[i]}{t['Jl'][i]}")
    trans.append(f"{t['vu'][i]}-{t['vl'][i]} {b[i]}({t['Jl'][i]})")
t.add_column(line,name="Line")
t.add_column(trans,name="Transition")
t.write("RoueffEtAl.tab",format='ipac',overwrite=True)
