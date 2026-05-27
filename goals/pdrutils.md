# Goal

pdrutils.py has become a hodge-podge of helper functions and globally defined variables.  Organization is needed.

One possible solution is to move the code in pdrutils.py to a utils subdirectory and split the code into separate files containing related operations (e.g., units, Measurements/images, Path-related functions).   You may propose an alternative.

It would be best if legacy code that uses "import pdrutils as utils" still functions.  However, ultimately we would want to update package code to use the new subpackage/submodule.


## Important
- The work must be done on a branch from master
- Claude should plan first, come up with a few options, and discuss the plan with Marc before implementing changes
