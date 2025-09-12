#!/usr/bin/env python
import glob
from astropy.io import fits
foo="Wolfire/Kaufman        wk2020    2020 wolfirekaufman/version2020/constant_density/z=1//losangle=90/avperp=3/ models.tab constant density    1.0      "
bar="null                                                 Wolfire/Kaufman 2020 constant density models, $A_V=7$, no freeze-out"
for file in glob.glob( "losangle=[034679]*/*" ):
     #print(file)
     with fits.open(file+"/CO109_CO54.fits") as f:
        losangle=f[0].header["LOSANGLE"]
        avperp=f[0].header["AVPERP"]
        avlos=f[0].header["AVLOS"]
        print(f" {foo} {losangle:.0f} {avperp:-6.0f} {avlos:-6.4f}    {bar}")

