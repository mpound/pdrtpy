ModelSets: The interface to models in the Toolbox
=================================================

PDRT supports a variety of PDR models to be used to fit your data. These are
represented in the Python class `~pdrtpy.modelset.ModelSet`.  Broadly three classes are
available:

    1. Wolfire/Kaufman 2020 models for constant density media (metallicities Z=0.5,1) and viewing angles 0 (face-on), 30, 45, 60, 75 degrees.
    2. Wolfire/Kaufman 2006 face-on models for constant density media (Z=0.1,1,3)
    3. Kosma-:math:`\tau` 2013 models for clumpy and non-clumpy media (Z=1)

The viewing angle models calculate the emitted line intensity along a
line-of-sight at angle, theta, with respect to the illuminated face of
the PDR. The angle theta=0 is a line that is perpendicular to the face
while the angle theta=90 is a line that is parallel to the face. The
line intensity for each transition as well as the integrated far-infrared
intensity is calculated as in Pabst et al. 2017, A&A, 606, A29. We make
the assumption that each line of sight passes through all layers of
the PDR to an optical depth of Av=7 from the face.  This means that the
integral along the line-of-sight as well as Av increases as 7/cos(theta),
and the thickness of the PDR increases as 7*tan(theta). The user should
keep in mind that this assumption may lead to unrealistically large Av
that should be checked against observations if possible. The different
angles and Avs are given in the fits headers as follows LOSANGLE angle
in degrees with respect to illuminated face of PDR.

Note that the face-on intensity is angle averaged over intensities emitted
from the PDR face as in Tielens & Hollenbach 1985, but the angle-on
intensity is the intensity along the ray at the given angle.

Models are stored in FITS format as ratios of intensities as a function
of radiation field  and hydrogen nucleus volume density.
The FITS headers list three different extinction values: 

- *AV*: optical depth in magnitudes of visual extinction of PDR along a line 
through the illuminated face to the deepest layers. All models are
fixed at :math:`A_V = 7`.

- *AVPERP*: optical depth in magnitudes of visual extinction of the PDR
perpendicular to Av

- *AVLOS*:  optical depth in magnitudes of visual extinction of the PDR along
the line-of-sight. 


For example how to use ModelSets, see the notebook
`PDRT_Example_ModelSets.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_ModelSets.ipynb>`_

----------

.. automodule:: pdrtpy.modelset
   :members:
   :undoc-members:
   :show-inheritance:
