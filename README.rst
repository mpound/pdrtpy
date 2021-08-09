*******************************************
PhotoDissociation Region Toolbox --- Python
*******************************************

*Reliable astrophysics at everyday low, low prices!* |reg| 

------------------------------------------------------------

.. image:: https://img.shields.io/badge/ascl-1102.022-blue.svg?colorB=262255
   :target: http://ascl.net/1102.022
   :alt: Astrophysics Source Code Library 1102.022

.. image:: http://www.repostatus.org/badges/latest/active.svg?style=plastic
 :target: http://www.repostatus.org/#active
 :alt: Project Status: Active - The project has reached a stable, usable state and is being actively developed.

.. image:: https://img.shields.io/pypi/pyversions/pdrtpy.svg?style=plastic
 :target: https://img.shields.io/pypi/pyversions/pdrtpy.svg?style=plastic
 :alt: Python version

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg?style=plastic
 :target: https://www.gnu.org/licenses/gpl-3.0
 :alt: GNU GPL v3 License

``pdrtpy`` is the new and improved version of the classic `PhotoDissociation Region Toolbox <http://dustem.astro.umd.edu/pdrt>`_, rewritten in Python with new capabilities and giving more flexibility to end users.  (The Perl/CGI version of PDRT is deprecated and no longer supported).

The new PDR Toolbox will cover many more spectral lines and metallicities
and allows map-based analysis so users can quickly compute spatial images
of density and radiation field from map data.  We provide Jupyter `Example Notebooks`_ for data analysis.  It also can support models from other PDR codes
enabling comparison of derived properties between codes.

The underlying model code has improved physics and chemistry. Critical updates include those discussed in 
`Neufeld & Wolfire 2016 <https://ui.adsabs.harvard.edu/abs/2016ApJ...826..183N/abstract>`_, plus photo rates from 
`Heays et al. 2017 <https://ui.adsabs.harvard.edu/abs/2017A%26A...602A.105H/abstract>`_, oxygen chemistry rates from 
`Kovalenko et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...856..100K/abstract>`_ and 
`Tran et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...854...25T/abstract>`_, 
and carbon chemistry rates from 
`Dagdigian 2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.3427D/abstract>`_. We have also implemented new collisional
excitation rates for |OI| from
`Lique et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.2313L/abstract>`_ (and Lique private
communication) and have included |13C| chemistry along with the
emitted line intensities for  |13CII| and |13CO|.

What is a PDR? 
==============
Photodissociation regions (PDRs) include all of the neutral gas in the
ISM where far-ultraviolet (FUV) photons dominate the chemistry and/or
heating.  In regions of massive star formation, PDRS are created at
the boundaries between the HII regions and neutral molecular cloud,
as photons with energies 6 eV < h \nu < 13.6 eV.
photodissociate molecules and photoionize other elements.  The gas is
heated from photo-electrons and cools mostly through far-infrared fine
structure lines like   |OI| and  |CII|.

For a full review of PDR physics and chemistry, see `Hollenbach & Tielens 1997 <https://ui.adsabs.harvard.edu/abs/1997ARA&A..35..179H>`_.

Getting Started
===============

Installation
------------

Requirements
^^^^^^^^^^^^

``pdrtpy`` requires Python 3 and recent versions of  `astropy <https://astropy.org>`_, `numpy <https://numpy.org>`_, `scipy <https://scipy.org>`_, `lmfit <https://lmfit.github.io/lmfit-py/>, and `matplotlib <https://matplotlib.org/>`_. If you want to run the `Example Notebooks`_, you also need `jupyter <https://jupyter.org>`_.

First make sure you are using Python 3:

.. code-block:: sh

   python --version

should show e.g., *3.7.6*. 


.. Also, make sure *setuptools* is up to date:

.. .. code-block:: sh

..   pip install -U setuptools

Install the package
^^^^^^^^^^^^^^^^^^^

Python has numerous ways to install packages; the easiest is with *pip*. 
The code is hosted at the `Python Packaging Index <https://pypi.org/project/pdrtpy/>`_, so you can type:

.. code-block:: sh

   pip install pdrtpy

If you do not have permission to install into your Python system package area, you will need to do a `user-install <https://pip.pypa.io/en/latest/user_guide/#user-installs>`_, which will install the package locally.

.. code-block:: sh

   pip install --user pdrtpy


Then go ahead and install the `Example Notebooks`_.

.. _notebooks:

Example Notebooks
-----------------

We have prepared jupyter iPython notebooks with examples of how to use ``pdrtpy``.  You can download these as follows.

.. code-block:: sh

    git clone https://github.com/mpound/pdrtpy-nb.git

If you don't have git, you can 
`download a zip file of the repository <https://github.com/mpound/pdrtpy-nb/archive/master.zip>`_.

To familiarize yourself with the capabilities of ``pdrtpy``, we suggest you do the notebooks in this order:

- PDRT_Example_Measurements.ipynb 
- PDRT_Example_ModelSets.ipynb
- PDRT_Example_Model_Plotting.ipynb
- PDRT_Example_Find_n_G0_Single_Pixel.ipynb  
- PDRT_Example_Make_n_G0_maps.ipynb       
- PDRT_Example_H2_Excitation.ipynb

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
.. |13C|    replace:: :sup:`13`\ C
.. |13CO|   replace:: :sup:`13`\ CO
.. |13CII|  replace:: [\ :sup:`13`\ C II]
.. |OI|  replace:: [O I]
.. |CII|  replace:: [C II]
.. |nu|     unicode:: 0x3bd .. greek nu
