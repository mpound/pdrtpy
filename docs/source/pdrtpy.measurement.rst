Measurements: How you put observations to the Toolbox
========================================================

To use PDR Toolbox, you need to create `Measurements` from your
observations. A Measurement consists of a value and an error.
These can be single-valued or an array of values.  In the typical
case of an image, the Measurement is a representation of a FITS file
with two HDUs, the first HDU is the spatial map of intensity and the
2nd HDU is the spatial map of the errors.  It is based on `astropy's
CCDData <https://docs.astropy.org/en/stable/api/astropy.nddata.CCDData.html>`_
if you are familiar with that. Typical sub-millimeter maps we get from
telescopes don't have the error plane, but PDRT makes it easy for you to
create one if you know the magnitude of the error. Typical FITS images will
be in intensity units, equivalent to :math:`{\rm erg~s^{-1}~cm^{-2}~sr^{-1}}`, 
or in :math:`{\rm K~km~s^{-1}}`.  For the latter, PDRT will do appropriate conversion as necessary
when it uses your images (the original Measurement remains unchanged).

For example how to use Measurements, see the notebook `PDRT_Example_Measurements.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Measurements.ipynb>`_.

----------

.. automodule:: pdrtpy.measurement
   :members:
   :undoc-members:
   :show-inheritance:
