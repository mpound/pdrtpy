## PDRT `tables` directory

This directory contains tables necessary for operations of the PDR Toolbox.

Python scripts
-------------------
These scripts write tables that are used by molecular excitation diagram fitting ``ExcitationFit.``

* `getpartfun.py` - retrieves partition function tables for selected molecules from HITRAN or exomol databases and writes them in an astropy Table ecsv format.
* `rewriteRoueff.py` - reformats the original H<sub>2</sub> excitation data table from Roueff et al 2019 into an astropy table format for use by `H2ExcitationFit.`
* `rewriteMeudon.py`  - takes `meudon/Lines` and `meudon/Levels` data files from the meudon PDR code and combines them into excitation data table format for use by `ExcitationFit`

Data tables
----------------

* `all_models.tab` - Description of all available models used by `ModelSet`
*  `av.tab` - Short table of Av line of sight and Av perpendicular as a function of viewing angle (losangle). Used by `ModelSet` to avoid having to read this info from  FITS headers.
* `*_transition.tab(.gz)` - Excitation data tables used by `ExcitationFit`
* `template_transition.tab` - template file for creating excitation data tables.
* `PartFun*.tab` - partition function tables Q(T), used by `ExcitationFit`
* `RoueffEtAlTable2.dat` - original data file from Roueff et al.
* `RoueffEtalReadme` - explanation file from Roueff et al.
Required Columns for Excitation Data Tables
--------------------------------------------------------------
