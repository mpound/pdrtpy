ModelSets
=========

PDRT supports a variety of PDR models to be used to fit your data. These are
represented in the Python class `ModelSet`. The
current default are the Wolfire/Kaufman 2006 models, which have
both constant density and constant thermal pressure versions, for
metallicities z=1 and z=3 (limited spectral lines). Models are stored
as ratios of intensities as a function of radiation field  :math:`G_0` and
hydrogen nucleus volume density  :math:`n` . We expect to update these soon with new physics
and a wider range of lines and metallicities. Any PDR models can be used
if they are stored in the correct FITS format. We are currently working
with Marcus Rollig to import the Kosma-:math:`\tau` models.

For example how to use ModelSets, see the notebook 
`PDRT_Example_ModelSets.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_ModelSets.ipynb>`_

----------

.. automodule:: pdrtpy.modelset
   :members:
   :undoc-members:
   :show-inheritance:
