#!/usr/bin/env python
from setuptools import setup, find_packages,find_namespace_packages

def readme():
    with open('README.md') as f:
        return f.read()

#print(find_namespace_packages())

excludelist= ["models","tables","notebooks","testdata","build","dist"]
print("Found packages ",find_packages(exclude=excludelist))


if False:

    setup(
        name="pdrtpy",
        version="2.0b",
        author = "Marc W. Pound",
        author_email  = "mpound@umd.edu",
        description="Photodissociation region analysis tools",
        keywords="PDR photodissociation",
        long_description=readme(),
        packages=find_packages(exclude=excludelist)
        include_package_data = True,
        package_data = {'pdrtpy': ['tables/*', 'notebooks/*', 'models/*','testdata/*'] },
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
