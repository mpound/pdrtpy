#! /usr/bin/env python
import glob
import os
import shutil

# for distributing Mark's FITS images from dir structure to our dir structure.
dirs = glob.glob('*')
for d in dirs:
    angmods = glob.glob(d+"/*")
    for a in angmods:
        avperp = f'avperp={a[-1]}'
        losangle = f'losangle={a[-5:-3]}'
        outdir=losangle + "/" + avperp
        os.makedirs(outdir,exist_ok=True)
        fits = glob.glob(a+"/*.fits")
        for file in fits:
            #print(f"mv {file} {outdir}")
            shutil.move(file,outdir)

