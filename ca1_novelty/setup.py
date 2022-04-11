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
    name="ca1_novelty",
    version="0.0.1",
    author="James Priestley",
    author_email="jbp2150@columbia.edu",
    description=("Analysis and simulation routines for CA1 novel context "
                 + "calcium imaging experiments"),
    license="GNU GPLv2",
    keywords="imaging microscopy neuroscience behavior",
    packages=['ca1_novelty'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=[],
)
