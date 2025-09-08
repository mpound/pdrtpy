#! /usr/bin/env python
import glob
import os
import shutil
import sys
import argparse
import tarfile

# for distributing Mark's FITS images from dir structure to our dir structure.
parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument("--file",        "-f", action="store",       help="input filename", required=True)
args= parser.parse_args()
with tarfile.open(args.file,mode='r:gz') as f:
    f.extractall(path='./00NEWMODELS')
os.chdir('00NEWMODELS')
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

