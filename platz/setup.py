#!/usr/bin/env python

from setuptools import setup

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""

setup(
    name="platz",
    version="0.0.1",
    author="James Priestley",
    author_email="jbp2150@columbia.edu",
    description=("Package for detection/analysis of neural spatial receptive"
                 + " fields"),
    license="GNU GPLv2",
    keywords="hippocampus place fields spatial tuning",
    packages=['platz'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=[],
)
