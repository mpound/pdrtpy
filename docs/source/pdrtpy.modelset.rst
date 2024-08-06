ModelSets: The interface to models in the Toolbox
=================================================

PDRT supports a variety of PDR models to be used to fit your data. These are
represented in the Python class `ModelSet`.  Broadly three classes are
available:

    1. Wolfire/Kaufman 2020 models for constant density media (metallicities Z=0.5,1)
    2. Wolfire/Kaufman 2006 models for constant density media (Z=0.1,1,3)
    3. Kosma-:math:`\tau` 2013 models for clumpy and non-clumpy media (Z=1)

Models are stored in FITS format as ratios of intensities as a function
of radiation field  and hydrogen nucleus volume density.

For example how to use ModelSets, see the notebook
`PDRT_Example_ModelSets.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_ModelSets.ipynb>`_

----------

.. automodule:: pdrtpy.modelset
   :members:
   :undoc-members:
   :show-inheritance:
