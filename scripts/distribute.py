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
parser.add_argument("--verbose",     "-v", action="store_true",  help="verbose output")
parser.add_argument("--dry-run",     "-d", action="store_true",  help="dry run, just print what it would do",)
args= parser.parse_args()
if args.dry_run:
    print(args)
with tarfile.open(args.file,mode='r:gz') as f:
    f.extractall(path='./00NEWMODELS')
os.chdir('00NEWMODELS')
dirs = glob.glob('LINEFITS/*')
for d in dirs:
    angmods = glob.glob(d+"/*")
    for a in angmods:
        avperp = f'avperp={a[-1]}'
        losangle = f'losangle={a[-5:-3]}'
        outdir=losangle + "/" + avperp
        if args.verbose or args.dry_run:
            print(f"{a=} {outdir=}")
        if not args.dry_run:
            os.makedirs(outdir,exist_ok=True)
        fits = glob.glob(a+"/*.fits")
        for file in fits:
            if args.verbose or args.dry_run:
                print(f"mv {file} {outdir}")
            if not args.dry_run:
                shutil.move(file,outdir)

