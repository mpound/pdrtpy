.. pdrtpy documentation master file, created by
   sphinx-quickstart on Fri Mar 13 10:44:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PhotoDissociation Region Toolbox Python
=======================================

``pdrtpy`` is the new and improved version of the classic `PhotoDissociation Region Toolbox <http://dustem.astro.umd.edu/pdrt>`_, rewritten in Python 
with new capabilities and giving more flexibility to end users. 


Installation
------------

Until I work out the errors in getting project uploaded to PyPi.org, the way to install is from the git repository.

.. code-block:: sh

   git clone https://github.com/mpound/pdrtpy
   cd pdrtpy
   pip install -e .

Requirements
------------
Python 3 and recent versions of  astropy, numpy, scipy, matplotlib. And jupyter if you want to run the example notebooks.

What is a PDR? 
--------------
Photodissociation regions (PDRs) include all of the neutral gas in the ISM
where far-ultraviolet (FUV) photons dominate the chemistry and/or heating.

What can I do with ``pdrtpy``?
------------------------------
Sales pitch goes here.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

What's your motto?
------------------
Reliable astrophysics at everyday low, low prices! |reg|

Modules
-------

.. autosummary::
   :toctree: 

   pdrtpy.measurement
   pdrtpy.modelset
   pdrtpy.pdrutils
   pdrtpy.plot
   pdrtpy.tool

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
