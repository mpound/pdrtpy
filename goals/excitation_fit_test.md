# Goal

To design and implement a unified way to test excitation fitting for any molecule, so that test data do not have to be hardcoded in each test.

## Details
The unified test structure should have a way of reading in a data file that contains

  - the molecule to be tested, e.g. "CO"
  - a data structure which contains:
       - intensity values
       - uncertainties on those values
       - string identifier for each value, e.g.  ""COv0-0J4-3"
       - string units of the values, e.g. "erg sr-1 cm-2"

  - whether the fit should be one or two component
  - value and uncertainty on the expected fitted temperature
  - value and uncertainty on the expected fitted column densities
  - A string reference for the origin of the test data

``Measurement.from_table`` may be useful, though the test data won't be strictly tabular.

## Important
- Data files should be stored in pdrtpy/testdata
- Data files should be a standard human-readable format like json or an astropy table. Claude may suggest an appropriate format.
- The work should be done on branch "add_more_mols"
- Claude should plan first and discuss the plan with Marc before implementing changes
