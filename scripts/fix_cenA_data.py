#!/usr/bin/env python
# quick script to fix headers of Lorenzo's CenA FITS files
from astroy.io import fits
from pdrtpy.util import get_testdata

maps = {}
for j in range(1, 9):
    key = f"H200S{j}"
    file = get_testdata(f"{key}_CenA.fits")
    hdulist = fits.open(file)
    for h in hdulist:
        for c in ["-X", "-Y", "-Z", "DX", "DY", "DZ"]:
            h.header.pop(f"OBSGEO{c}", None)
    hdulist.verify("fix")
    hdulist.writeto(file, overwrite=True, output_verify="fix")
    hdulist.close()
