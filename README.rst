Python porting of PRNU extractor and helper functions
=====================================================

The original work is based on

|DOI| 

Original Travis build

|TRAVIS|

About
-----

Python package which provides several functions to compute and test
cameras PRNU.

We have added, to the original work, the possibility to carry out noise
extraction using a PyTorch model which accepts normalized images whose
values are between 0 and 1.

Since the project has been adapted to deal with the following `FFDNet
implementation <https://www.ipol.im/pub/art/2019/231/?utm_source=doi>`__,
the input of such model should be an image with size [1, channels,
heigth, width] and the standard deviation (sigma) of the noise.

Authors
-------

Original authors
~~~~~~~~~~~~~~~~

-  Luca Bondi (luca.bondi@polimi.it)
-  Paolo Bestagini (paolo.bestagini@polimi.it)
-  Nicolò Bonettini (nicolo.bonettini@polimi.it)

Later authors
~~~~~~~~~~~~~

-  Simone Alghisi (simone.alghisi-1@studenti.unitn.it)
-  Samuele Bortolotti (samuele.bortolotti@studenti.unitn.it)
-  Massimo Rizzoli (massimo.rizzoli@studenti.unitn.it)

Usage
-----

Set up
~~~~~~

Clone this repository

.. code:: bash

   git clone https://github.com/samuelebortolotti/prnu-python

Move to the project folder

.. code:: bash

   cd prnu-python

Install the package locally on your machine:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installation with ``pip`` can be performed as follows

.. code:: bash

   pip install .

Or directly from GitHub

::

   pip install git+git://github.com/samuelebortolotti/prnu-python@v[version]

Where [version] is the version of the

::

   pip install git+git://github.com/samuelebortolotti/prnu-python@v2.0

Or you can add the package in your ``requirements.txt`` file, and
install it later, by including the following line

::

   git+git://github.com/samuelebortolotti/prnu-python@v[version]

For example:

::

   git+git://github.com/samuelebortolotti/prnu-python@v2.0

Now you can import the ``prnu`` package whenever and wherever you want.

Install the package in the virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the ``GNU Makefile`` to generate the virtual environment by
typing

.. code:: bash

   make env

Activate the virtual environment

.. code:: bash

   source venv/prnu/bin/activate

Install the requirements

.. code:: bash

   make install

Documentation
~~~~~~~~~~~~~

The documentation is generated using
`Sphinx <https://www.sphinx-doc.org/en/master/>`__.

First, install the development requirements

.. code:: bash

   make install-dev

Then generate the Sphinx layout

.. code:: bash

   make doc-layout

Generate the documentation content; the documentation will be generated
in the ``docs`` folder.

.. code:: bash

   make doc

Then, you can open the documentation through ``xdg-open`` by typing

.. code:: bash

   make open-doc

Test
----

You can run the tests by typing

.. code:: bash

   cd test
   python -m unittest test_prnu.TestPrnu

Tested with Python >= 3.6

Credits
-------

Reference MATLAB implementation by Binghamton university:
http://dde.binghamton.edu/download/camera_fingerprint/

.. |DOI| image:: https://zenodo.org/badge/158570703.svg
   :target: https://zenodo.org/badge/latestdoi/158570703

.. |TRAVIS| image:: https://travis-ci.org/polimi-ispl/prnu-python.svg?branch=master&status=passed
   :target: https://travis-ci.org/polimi-ispl/prnu-python
