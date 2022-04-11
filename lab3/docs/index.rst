.. lab3 documentation master file, created by
   sphinx-quickstart on Wed Oct  9 02:05:59 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lab3 - Lab Analysis Bundle for Python3
================================

Overview
--------

lab3 is a redesign of the shared code repository. The core philosophy is to
centralize much of the analysis code that formerly lived in hard-to-maintain
scripts, and provide common frameworks to add new analysis methods. The module
provides a set of tools for most stages of data processing and analysis,
from signal extraction, formatting, analysis, and statistical modeling. Many
strategies exist for each step, and so the user has many options when building
their workflow. We aim for the tools to be modular and flexible, rather than
prescribing a long, specific sequence of steps, which we leave for the user to
assemble in their own personal scripts.


Contributing to the repository
------------------------------

Code added to this repository should be as general as possible and re-use
existing structures where applicable:

* New analysis strategies should conform to the interface design of existing
  strategies so methods can be easily interchanged and behavior is predictable,
  for example, implementing a new way to calculate dF/F calculations by
  subclassing DFOFStrategy.
* When augmenting the features of existing classes or functions, think
  carefully about whether these features fit the general purpose of the class,
  or whether they  should be made separate to suit more specific use cases.
  This will help to reduce code interference, make debugging easier, and
  maintain a stable core functionality.
* When scripts are necessary, they should leverage the tools in the lab3
  module to tersely process data, rather than contain extensive functions
  themselves that define the analysis. These live outside of the main module
  itself, in `scripts/`.
* Document your code! We conform to the `numpy doc string conventions
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_. When your code is
  merged into the master branch, this website will be automatically updated
  with the documentation written into the python files, so be thorough!


Documentation
-------------
.. toctree::
   :maxdepth: 2

   Introduction <index>
   install
   tutorial
   lab3/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
