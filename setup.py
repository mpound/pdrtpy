#!/usr/bin/env python
from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="pdrtpy",
    version="2.0b",
    author = "Marc W. Pound",
    author_email  = "mpound@umd.edu",
    description="Photodissociation region analysis tools",
    keywords="PDR photodissociation",
    long_description=readme(),
    packages=find_packages(),
    include_package_data = True,
    install_requires = [
        'astropy',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    url="https://dustem.astro.umd.edu",
    project_urls={
        "Documentation": "https://pdrtpy.readthedocs.io",
        "Source Code": "https://github.com/mpound/pdrtpy",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
    ],
    license = "GPLv3",
    zip_safe=False,
    python_requires='>=3.6'
)
