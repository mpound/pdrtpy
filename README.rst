*******************************************
PhotoDissociation Region Toolbox --- Python
*******************************************

*Reliable astrophysics at everyday low, low prices!* |reg| 

------------------------------------------------------------

.. image:: https://api.codacy.com/project/badge/Grade/f9a3b10790e74ba887ea7f3a7525189d
   :alt: Codacy Badge
   :target: https://app.codacy.com/gh/mpound/pdrtpy?utm_source=github.com&utm_medium=referral&utm_content=mpound/pdrtpy&utm_campaign=Badge_Grade_Settings

.. image:: https://img.shields.io/badge/ascl-1102.022-blue.svg?colorB=262255&style=plastic
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
 
.. image:: https://readthedocs.org/projects/pdrtpy/badge/?version=latest&style=plastic
 :target: https://pdrtpy.readthedocs.io/en/latest/?badge=latest
 :alt: Documentation status
 
.. image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg?style=plastic
 :target: https://github.com/mpound/pdrtpy/blob/master/CODE_OF_CONDUCT.md
 :alt: Contributor Covenant Code of Conduct  

.. image:: https://github.com/mpound/pdrtpy/actions/workflows/run-integration-tests.yml/badge.svg?branch=active-devel
 :alt: Integration test status

.. image:: https://github.com/mpound/pdrtpy/blob/master/coverage.svg?branch=master
 :alt: Code coverage

``pdrtpy`` is the new and improved version of the formerly web-based `PhotoDissociation Region Toolbox <http://dustem.astro.umd.edu/>`_, rewritten in Python with new capabilities and giving more flexibility to end users.  (The web-based /CGI version of PDRT is deprecated and no longer supported). 

The PDR Toolbox is a science-enabling tool for the community, designed to
help astronomers determine the physical parameters of photodissociation
regions from observations. Typical observations of both Galactic
and extragalactic PDRs come from ground- and space-based millimeter,
submillimeter, and far-infrared telescopes such as ALMA, SOFIA, JWST,
Spitzer, and Herschel. Given a set of observations of spectral line or
continuum intensities, PDR Toolbox can compute best-fit FUV incident
intensity and cloud density based on our models of PDR emission.

The PDR Toolbox will cover a wide range of spectral lines and metallicities
and allows map-based analysis so users can quickly compute spatial
images of density and radiation field from map data.  We provide Jupyter
`Example Notebooks`_ for data analysis.  It also can support models from
other PDR codes enabling comparison of derived properties between codes.

The underlying PDR model code has improved physics and chemistry. Critical updates include those discussed in 
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

We also support fitting of temperatures and column densities to |H2| excitation diagrams.

Up to date documentation can be found at `pdrtpy.readthedocs.io <http://pdrtpy.readthedocs.io/>`_.

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

``pdrtpy`` requires Python 3 and recent versions of  `astropy <https://astropy.org>`_, `numpy <https://numpy.org>`_, `scipy <https://scipy.org>`_, `lmfit <https://lmfit.github.io/lmfit-py/>`_, and `matplotlib <https://matplotlib.org/>`_. If you want to run the `Example Notebooks`_, you also need `jupyter <https://jupyter.org>`_.

First make sure you are using Python 3:

.. code-block:: sh

   python --version

should show e.g., *3.7.6*. 


.. Also, make sure *setuptools* is up to date:

.. .. code-block:: sh

..   pip install -U setuptools

Install the package
^^^^^^^^^^^^^^^^^^^

With pip
--------
Python has numerous ways to install packages; the easiest is with *pip*. 
The code is hosted at the `Python Packaging Index <https://pypi.org/project/pdrtpy/>`_, so you can type:

.. code-block:: sh

   pip install pdrtpy

If you do not have permission to install into your Python system package area, you will need to do a `user-install <https://pip.pypa.io/en/latest/user_guide/#user-installs>`_, which will install the package locally.

.. code-block:: sh

   pip install --user pdrtpy

* For installation from github, see `For Developers`_ below.

Then go ahead and install the `Example Notebooks`_.

.. _notebooks:

Example Notebooks
-----------------

We have prepared Jupyter iPython notebooks with examples of how to use ``pdrtpy``.  You can download these as follows.

.. code-block:: sh

    git clone https://github.com/mpound/pdrtpy-nb.git

If you don't have git, you can 
`download a zip file of the repository <https://github.com/mpound/pdrtpy-nb/archive/master.zip>`_.

To familiarize yourself with the capabilities of ``pdrtpy``, we suggest you do the notebooks in this order:

- `Working with Measurements <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Measurements.ipynb>`_
- `Introduction to ModelSets <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_ModelSets.ipynb>`_
- `Exploring Models <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Model_Plotting.ipynb>`_
- `Determining Radiation Field and Intensity <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Find_n_G0_Single_Pixel.ipynb>`_
- `Image Radiation Field and Intensity for Maps <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Make_n_G0_maps.ipynb>`_
- `Fitting |H2| Excitation Diagrams <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_H2_Excitation.ipynb>`_
- `Adding Custom Models <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Adding_Models.ipynb>`_

Getting Help & Giving Feedback
==============================
If you have a question or wish to give feedback about using PDR Toolbox or about the example notebooks, head on over to our `PDR Toolbox online forum <https://groups.google.com/g/pdrt>`_.  There you can post your question and engage in discussion with the developers and other users.  Feature requests from the community are welcome.

Reporting Issues
================
If you find a bug or something you think is in error, please report it on
the `github issue tracker <https://github.com/mpound/pdrtpy/issues>`_. 
(You must have a `Github account <https://github.com/>`_ to submit an issue).
If you aren't sure if something is a bug or not, or if you don't wish to
create a Github account, you can post to the `PDR Toolbox forum
<https://groups.google.com/g/pdrt>`_.

Contribute Code or Documentation
=================================
We welcome contributions and ideas to improve the PDR Toolbox!  **All contributors agree to follow our** `Code of Conduct <https://github.com/mpound/pdrtpy/blob/master/CODE_OF_CONDUCT.md>`_ .  Please look at our 
`Roadmap of Functionality <https://github.com/mpound/pdrtpy/blob/master/roadmap.md>`_ 
to see the main new features we want to build.  You can help out with those or suggest new features. 

For Developers
--------------
If you plan to tinker with the code, you should fork the repo and work on your own fork.  Point your browser to 
`https://github.com/mpound/pdrtpy <https://github.com/mpound/pdrtpy>`_
and click on *fork* in the upper right corner.   After you have made your changes, create a pull request to merge them into the master branch.

You may want to use a virtual environment to protect from polluting your daily working environment (especially if you have a stable version of `pdrtpy` installed).

.. code-block:: sh
  
   sudo apt-get install python3-venv
   python -m venv ~/pdrtpy_venv
   source ~/pdrtpy_venv/bin/activate[.csh] 
   cd pdrtpy
   pip install -r requirements.txt
   pip install -e .


.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
.. |13C|    replace:: :sup:`13`\ C
.. |13CO|   replace:: :sup:`13`\ CO
.. |13CII|  replace:: [\ :sup:`13`\ C II]
.. |OI|  replace:: [O I]
.. |CII|  replace:: [C II]
.. |H2|  replace:: H\ :sub:`2`
.. |nu|     unicode:: 0x3bd .. greek nu
