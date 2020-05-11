#!/usr/bin/env python
# NOTE: if any required diretories are added, put them in MANIFEST.in or
# readthedocs build will break

from setuptools import setup, find_packages,find_namespace_packages
import pdrtpy

def readme():
    with open('README.rst') as f:
        return f.read()

#print(find_namespace_packages())

#excludelist= ["build","dist"]
excludelist= []
print("Found packages ",find_packages(exclude=excludelist))

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
    ],
    url = "https://dustem.astro.umd.edu",
    project_urls = {
        "Documentation": "https://pdrtpy.readthedocs.io",
        "Source Code": "https://github.com/mpound/pdrtpy",
    },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
    ],
    license = "GPLv3",
    zip_safe = False,
    python_requires = '>=3.6'
)
