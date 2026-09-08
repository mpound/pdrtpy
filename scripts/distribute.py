#! /usr/bin/env python
import argparse
import glob
import os
import shutil
import sys
import tarfile

# for distributing Mark's FITS images from dir structure to our dir structure.
parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument("--file", "-f", action="store", help="input filename")
parser.add_argument("--verbose", "-v", action="store_true", help="verbose output")
parser.add_argument("--no-untar", "-n", action="store_true", help="don't untar the file (it's already untarred)")
parser.add_argument(
    "--copyts", "-t", action="store_true", help="copy the surface temperature (T_s) maps to all leaf directories"
)
parser.add_argument(
    "--dry-run",
    "-d",
    action="store_true",
    help="dry run, just print what it would do",
)
args = parser.parse_args()
if args.dry_run:
    print(args)
if not args.no_untar:
    if args.file is None:
        print(f"{sys.argv[0]}: error: argument --file/-f is required if untarring")
        exit(255)
    with tarfile.open(args.file, mode="r:gz") as f:
        f.extractall(path="./00NEWMODELS")
os.chdir("00NEWMODELS")
dirs = glob.glob("LINEFITS/*")
tsfile = None
for d in dirs:
    angmods = glob.glob(d + "/*")
    for a in angmods:
        # for surface temperature, grab the location of the FITS file
        # if it is being copied
        if "tsav0p01" in a.lower():
            if args.copyts and "sm.fits" in a:
                tsfile = a
        else:
            # avperp = f'avperp={a[-1]}'
            b = a.replace("Av0", "")  # remove extraneous Av on some filenames
            losangle = f"losangle={b[-2:]}"
            outdir = losangle  # + "/" + avperp
            if args.verbose or args.dry_run:
                print(f"{a=} {outdir=}")
            if not args.dry_run:
                os.makedirs(outdir, exist_ok=True)
            fits = glob.glob(a + "/*.fits")
            for file in fits:
                base = os.path.basename(file)
                outfile = f"{outdir}/{base}"
                if args.verbose or args.dry_run:
                    print(f"cp {file} {outfile}")
                if not args.dry_run:
                    shutil.copy(file, outfile)
            if args.copyts and tsfile is not None:
                tsout = f"{outdir}/TSAV0p01sm.fits"
                if args.verbose:
                    print(f"cp {tsfile} {tsout}")
                shutil.copy(tsfile, tsout)
