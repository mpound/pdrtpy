#!/usr/bin/env python
from astropy.table import Table
import glob
import os

# to be run in z=XX dir to update models.tab
# must copy a oldmodels.tab into avperp=XX dirs first
dirs = glob.glob('losangle*/avperp*')
for a in dirs:
    #angmods = glob.glob(d+"/avperp*")
    #for a in angmods:
    avperp = a[-1]
    losangle = a[a.find('=')+1:a.find('=')+3]
    path = a + "/models.tab"
    t = Table.read(path,format='ascii.ipac')
    z=len(t)
    if 'losangle' not in t.colnames: 
        t.add_columns(names=['losangle','avperp'],cols=[[losangle]*z,[avperp]*z])
        t.write(path,format='ascii.ipac',overwrite=True)
    else:
        print(f"{path} already has columns; skipping.")
    #print(f"{a=}, {path=}")
