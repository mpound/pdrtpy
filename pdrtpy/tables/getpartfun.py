#!/usr/bin/env python3
"""
Created on Tue May  6 12:11:41 2025
Retrieve the partition function tabulations for selected molecules from HITRAN database or EXOMOL database
See https://hitran.org/docs/iso-meta/ and https://www.exomol.com
@author: mpound
"""

import argparse
import os
import sys

import pdrtpy.utils as utils
import wget
from astropy.table import Table

parser = argparse.ArgumentParser(prog=sys.argv[0])
parser.add_argument(
    "--mol", "-m", action="store", help="molecule to use. Default is an internal list depending on database chosen"
)
parser.add_argument("--verbose", "-v", action="store_true", help="verbose output")
parser.add_argument(
    "--database", "-d", action="store", help="database to use: hitran or exomol", default="hitran", required=True
)
parser.add_argument(
    "--trans", "-t", action="store_true", help="for exomol, also fetch the transition and state files", default=True
)
parser.add_argument("--overwrite", "-o", action="store_true", help="overwrite existing files", default=False)
parser.add_argument("--nofetch", "-n", action="store_true", help="do not (re)fetch the data files", default=False)

args = parser.parse_args()

if args.database == "hitran":
    baseurl = "https://hitran.org/"
    info = "https://hitran.org/docs/iso-meta/"
    # Molecule name and HITRAN ID number
    mols = {"12C16O": 26, "13C16O": 27, "12C18O": 28, "12C17O": 29, "H2": 103, "13C18O": 30}
elif args.database == "exomol":
    baseurl = "https://www.exomol.com/"
    info = "https://www.exomol.com/data/molecules/ and Pearce+2024, MNRAS 527, 10726"
    # Molecule name and exomol file root
    mols = {"CH_p": "12C-1H_p", "13CH_p": "13C-1H_p"}
    key = "CH_p"
else:
    raise ValueError(f"Urecognized database: {args.database}. Choices are hitran or exomol")

if args.database == "hitran":
    for m in mols:
        if args.verbose:
            print(f"Doing {mols[m]} from {args.database}...")
        file = f"data/Q/q{mols[m]}.txt"
        out = f"PartFun_{m}.txt"
        tabout = f"PartFun_{m}.tab"
        url = f"{baseurl}{file}"
        if not args.nofetch:
            if args.verbose:
                print(f"\nwget url={url} out={out}")
            if args.overwrite:
                if os.path.exists(out):
                    os.remove(out)
            wget.download(url=url, out=out)
        t = Table.read(out, format="ascii")
        t.rename_columns(["col1", "col2"], ["T", "Q"])
        t["T"].unit = "K"
        t.meta["Comment"] = (
            f"Partition function tabulation for {m}. Retrieved from {url} on {utils.now()}. See also {info}."
        )
        t.write(tabout, overwrite=args.overwrite, format="ascii.ecsv")
elif args.database == "exomol":
    for m in mols:
        if args.verbose:
            print(f"\nDoing {mols[m]} from {args.database}...")
        file = f"db/{key}/{mols[m]}/PYT/{mols[m]}__PYT.pf"
        out = f"PartFun_{m}.txt"
        tabout = f"PartFun_{m}.tab"
        url = f"{baseurl}{file}"
        if not args.nofetch:
            if args.verbose:
                print(f"\nwget url={url} out={out} overwrite={args.overwrite}")
            if args.overwrite:
                if os.path.exists(out):
                    os.remove(out)
            wget.download(url=url, out=out)
        t = Table.read(out, format="ascii")
        t.rename_columns(["col1", "col2"], ["T", "Q"])
        t["T"].unit = "K"
        t.meta["Comment"] = (
            f"Partition function tabulation for {m}. Retrieved from {url} on {utils.now()}. See also {info}."
        )
        t.write(tabout, overwrite=args.overwrite, format="ascii.ecsv")
        if args.trans:
            out1 = f"{mols[m]}__PYT.trans.bz2"
            out2 = f"{mols[m]}__PYT.states.bz2"
            transfile = f"/db/{key}/{mols[m]}/PYT/{out1}"
            statesfile = f"/db/{key}/{mols[m]}/PYT/{out2}"
            url = f"{baseurl}{transfile}"
            if not args.nofetch:
                if args.verbose:
                    print(f"\nwget url={url} out={out1} overwrite={args.overwrite}")
                if args.overwrite:
                    if os.path.exists(out1):
                        os.remove(out1)
                wget.download(url=url, out=out1)
                url = f"{baseurl}{statesfile}"
                if args.verbose:
                    print(f"\nwget url={url} out={out2} overwrite={args.overwrite}")
                if args.overwrite:
                    if os.path.exists(out2):
                        os.remove(out2)
                wget.download(url=url, out=out2)
            c = [f"col{n}" for n in range(1, 5)]
            r = ["f", "i", "A", "wfi"]
            units = ["", "", "s-1", "cm-1"]
            t = Table.read(out1, format="ascii", units=units)
            t.rename_columns(c, r)
            t.meta["Comment"] = (
                f"Transition data for {m}. Retrieved from {url} on {utils.now()}. See also {info}. Columns are f: counting number upper state, i: counting number lower state, A: Einstein coefficient, wfi: wavenumber f->i"
            )
            tabout = f"{m}_trans.tab"
            t.write(tabout, overwrite=True, format="ascii.ecsv")

            c = [f"col{n}" for n in range(1, 16)]
            r = [
                "i",
                "Energy",
                "gi",
                "J",
                "Unc",
                "tau",
                "+/-",
                "e/f",
                "State",
                "v",
                "Lambda",
                "Sigma",
                "Omega",
                "Label",
                "Calc",
            ]
            units = ["", "cm-1", "", "", "cm-1", "s", "", "", "", "", "", "", "", "", "cm-1"]
            t = Table.read(out2, format="ascii", units=units)
            t.rename_columns(c, r)
            t.meta["Comment"] = (
                f"States data for {m}. Retrieved from {url} on {utils.now()}. See also {info}. Notes. i: state counting number, Energy: upper(?) level energy (cm-1), gi: total statistical weight, J: total angular momentum, Unc: uncertainty (cm-1), tau : lifetime (s), +/-: total parity, e/f: rotational parity, v: vibrational level quantum number, Lambda : projection of electronic angular momentum, Sigma : projection of electron spin, Omega : projection of total angular momentum (Omega  = Lambda  + Sigma ), Label: 'Ma' denotes MARVEL energy level, 'Ca' denotes LEVEL calculated energy level, and Calc.: LEVEL calculated energy level value."
            )
            tabout = f"{m}_states.tab"
            t.write(tabout, overwrite=args.overwrite, format="ascii.ecsv")

print("\n")
