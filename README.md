# QuTiP - Quantum Optimal Control

The `qutip-qoc` package builds up on the `qutip-qtrl` [package](https://github.com/qutip/qutip-qtrl).

It enhances it by providing two additional algorithms to optimize analytically defined control functions.
The first one is an extension of Gradient Optimization of Analytic conTrols [(GOAT)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.150401).
The second one (JOPT) leverages QuTiPs version 5 new diffrax abilities to directly calculate gradients of JAX defined control functions using automatic differentiation.

Both algorithms consist of a two-layer approach to find global optimal values for parameterized analytical control functions.
The global optimization layer provides `scipy.optimize.dual_annealing` and `scipy.optimize.basinhopping`, while the local minimization layer supports all
gradient driven `scipy.optimize.minimize` methods.

The package also aims for a more general way of defining control problems with QuTiP and makes switching between the four control algorithms (GOAT, JOPT, and GRAPE and CRAB implemented in qutip-qtrl) very easy.

As with `qutip-qtrl`, the `qutip-qoc` package aims at providing advanced tools for the optimal control of quantum devices.
Compared to other libraries for quantum optimal control, `qutip-qoc` puts additional emphasis on the physics layer and the interaction with the QuTiP package.
Along with the extended GOAT and JOPT algorithms the package offers support for both the CRAB and GRAPE methods defined in `qutip-qtrl`.

If you would like to know the future development plan and ideas, have a look at the [qutip roadmap and ideas](https://qutip.readthedocs.io/en/latest/development/roadmap.html).

## Quick start

To install the package, use

```
pip install qutip-qoc
```

By default, the dependencies required for JOPT and for the RL (reinforcement learning) algorithm are omitted.
They can be included by using the targets `qutip-qoc[jopt]` and `qutip-qoc[rl]`, respectively (or `qutip-qoc[full]` for all dependencies).

## Documentation and tutorials

The documentation of `qutip-qoc` updated to the latest development version is hosted at [qutip-qoc.readthedocs.io](https://qutip-qoc.readthedocs.io/en/latest/).
Tutorials related to using quantum optimal control in `qutip-qoc` can be found [_here_](https://qutip.org/qutip-tutorials/#optimal-control).

## Installation from source

If you want to edit the source code, please download the source code and run the following command under the root `qutip-qoc` folder,

```
pip install --upgrade pip
pip install -e .
```

which makes sure that you are up to date with the latest `pip` version. Contribution guidelines are available [_here_](https://qutip-qoc.readthedocs.io/en/latest/contribution-code.html).

To build and test the documentation, additional packages need to be installed:

```
pip install pytest matplotlib sphinx sphinxcontrib-bibtex numpydoc sphinx_rtd_theme sphinxcontrib-bibtex
```

Under the `doc` directory, use

```
make html
```

to build the documentation, or

```
make doctest
```

to test the code in the documentation.

## Testing

To test the installation, choose the correct branch that matches with the version, e.g., `qutip-qoc-0.2.X` for version 0.2. Then download the source code and run from the `qutip-qoc` directory.

```
pytest tests
```

## Citing `qutip-qoc`

If you use `qutip-qoc` in your research, please cite the original QuTiP papers that are available [here](https://dml.riken.jp/?s=QuTiP).

## Support

This package is supported and maintained by the same developers group as QuTiP.

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)

QuTiP development is supported by [Nori's lab](http://dml.riken.jp/)
at RIKEN, by the University of Sherbrooke, by Chalmers University of Technology, by Macquarie University and by Aberystwyth University,
[among other supporting organizations](http://qutip.org/#supporting-organizations).

## License

[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE.txt file are satisfied.
