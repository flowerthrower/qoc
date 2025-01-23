************
Installation
************

.. _quickstart:

Quick start
===========
To install the package ``qutip-qoc`` from PyPI, use

.. code-block:: bash

    pip install qutip-qoc


.. _prerequisites:

Prerequisites
=============
This package is built upon QuTiP, of which the installation guide can be found at on `QuTiP Installation <http://qutip.org/docs/latest/installation.html>`_.

In particular, the following packages are necessary for running ``qutip-qoc``:

.. code-block:: bash

    numpy scipy cython qutip qutip-qtrl

The following packages are required for using the JOPT algorithm:

.. code-block:: bash

    jax jaxlib qutip-jax

The following packages are required for the RL (reinforcement learning) algorithm:

.. code-block:: bash

    gymnasium stable-baselines3

The following package is used for testing:

.. code-block:: bash

    pytest

In addition,

.. code-block:: bash

    sphinx numpydoc sphinx_rtd_theme

are used to build and test the documentation.

Install ``qutip-qoc`` from source code
======================================

To install the package, download to source code from `GitHub website <https://github.com/flowerthrower/qutip-qoc.git>`_ and run

.. code-block:: bash

    pip install .

under the directory containing the ``setup.cfg`` file.

If you want to edit the code, use instead

.. code-block:: bash

    pip install -e .

To test the installation from a download of the source code, run from the ``qutip-qoc`` directory

.. code-block:: bash

    pytest tests
