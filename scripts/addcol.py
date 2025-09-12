#!/usr/bin/env python
import argparse
from astropy.table import Table
import glob
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument("--file",     "-f", action="store",       help="output filename", default="models.tab")
parser.add_argument("--verbose",  "-v", action="store_true",  help="verbose output")
parser.add_argument("--template", "-t", action="store",       help="template input file")
args= parser.parse_args()

# to be run in z=XX dir to update models.tab
dirs = glob.glob('losangle*/avperp*')
if len(dirs) == 0:
    print("## Found no directories to process. You must be in a z=XX dir.")
    sys.exit(255)
if args.verbose:
    print(f"Found {dirs}=")

for a in dirs:
    avperp = a[-1]
    losangle = a[a.find('=')+1:a.find('=')+3]
    path = Path(a + "/" +args.file)
    if path.exists():
        t = Table.read(path,format='ascii.ipac')
    else:
        # read from template if provided
        if args.template is None:
            raise ValueError(f"Must provide template file since {path} doesn't exists yet")
        else:
            t = Table.read(args.template,format='ascii.ipac')
    if 'losangle' not in t.colnames: 
        z=len(t)
        t.add_columns(names=['losangle','avperp'],cols=[[losangle]*z,[avperp]*z])
        t.write(path,format='ascii.ipac',overwrite=True)
    else:
        print(f"{path} already has columns [losangle, avperp]; skipping.")
