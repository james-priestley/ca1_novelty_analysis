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
    name="lab3",
    version="0.0.1",
    author="Losonczy Lab",
    author_email="software@losonczylab.org",
    description=("Python3 migration and revision of the Losonczy Lab "
                 + "Analysis Bundle"),
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience behavior",
    url="http://www.losonczylab.org/",
    packages=['lab3'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=[],

)
