#!/usr/bin/env python
# NOTE: If any required directories are added, put them in MANIFEST.in or
# readthedocs build will break

from setuptools import setup, find_packages
import sys
import pdrtpy

def check_python(major,minor):
    try:
        assert sys.version_info >= (major,minor)
    except AssertionError:
        raise Exception("pdrtpy requires you use Python %d.%d or above"%(major,minor))

def readme():
    with open('README.rst') as f:
        return f.read()

# Ensure they are using Python 3.7 or above
check_python(3,7)

#excludelist= ["build","dist"]
excludelist= []
#print("Found packages ",find_packages(exclude=excludelist))

setup(
    name="pdrtpy",
    version = pdrtpy.VERSION,
    author  = pdrtpy.AUTHORS,
    author_email = "mpound@umd.edu",
    description = pdrtpy.DESCRIPTION,
    keywords = pdrtpy.KEYWORDS,
    long_description = readme(),
    packages = find_packages(exclude=excludelist),
    include_package_data = True,
    install_requires = [
        'astropy',
        'numpy',
        'scipy',
        'matplotlib',
        'lmfit',
        'numdifftools'
    ],
    url = "http://dustem.astro.umd.edu",
    project_urls = {
        "Documentation": "https://pdrtpy.readthedocs.io",
        "Source Code": "https://github.com/mpound/pdrtpy",
    },
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
    ],
    license = "GPLv3",
    zip_safe = False,
    python_requires = '>=3.7'
)
