Serendipyty
===========
\(n\) The occurrence and development of events by chance in a happy or beneficial way.

.. image:: http://img.shields.io/pypi/v/verde.svg?style=flat-square
    :alt: Latest version on PyPI
    :target: https://pypi.python.org/pypi/serendipyty

.. image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/serendipyty/serendipyty/master?filepath=notebooks%2FSeismic_modeling_and_visualization_v1.ipynb

Serendipyty is a Python library for learning and teaching Geophysics.
The goal of this project is to introduce the concept of
reproducible computational science to students with little or no programming experience.
Using Python and Jupyter, the purpose of this project is to present students
with an understanding of the role computation can play in providing reproducible
and verifiable answers to scientific questions arising from various fields of physical sciences.

Installation
============
Install with

.. code-block:: shell

    conda create -n yourenv python=3.6
    source activate yourenv

    # Setup
    conda install numpy cython

    conda install matplotlib joblib pandas

    conda install jupyter
    python -m ipykernel install --user --name yourenv --display-name "Python (yourenv)"

    pip install --upgrade -v serendipyty

Documentation
=============
Documentation is generated using `Sphinx <http://www.sphinx-doc.org/en/master/#>`_ and
uses the `numpy docstring style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

Collaborators
=============
* Filippo Broggini, ETH Zürich
* Erik Koene, ETH Zürich
* Simon Schneider, Utrecht University
* Ivan Vasconcelos, Utrecht University

Contacting us
=============
* Feel free to `open an issue
  <https://github.com/serendipyty/serendipyty/issues/new>`_
  to get in touch with the developers.

Acknowledgments
===============
* `Exploration and Environmental Geophysics group <http://www.eeg.ethz.ch/>`_ at ETH Zürich
* `TIDES COST <http://www.tides-cost.eu/>`_ through a Short Term Scientific Mission at Utrecht University

Credits
=======
* `PySIT <https://github.com/pysit/pysit>`_ Copyright (c) 2011-2017 MIT and PySIT Developers
* `Fatiando <https://www.fatiando.org/>`_  Copyright (c) 2010-2016, Leonardo Uieda
