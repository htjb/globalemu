========================================================
globalemu: Robust and Fast Global 21-cm Signal Emulation
========================================================

Introduction
------------

:globalemu: Robust Global 21-cm Signal Emulation
:Author: Harry Thomas Jones Bevins
:Version: 1.4.0
:Homepage: https://github.com/htjb/globalemu
:Documentation: https://globalemu.readthedocs.io/

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/htjb/globalemu/master?filepath=notebooks%2F
.. image:: https://readthedocs.org/projects/globalemu/badge/?version=latest
 :target: https://globalemu.readthedocs.io/en/latest/?badge=latest
 :alt: Documentation Status
.. image:: https://codecov.io/gh/htjb/globalemu/branch/master/graph/badge.svg?token=4KLLNT45TQ
 :target: https://codecov.io/gh/htjb/globalemu
.. image:: https://badge.fury.io/py/globalemu.svg
 :target: https://badge.fury.io/py/globalemu
.. image:: https://github.com/htjb/globalemu/workflows/CI/badge.svg?event=push
 :target: https://github.com/htjb/globalemu/actions?query=workflow%3ACI
 :alt: github CI
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
 :target: https://pypi.org/project/globalemu/
 :alt: MIT License
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4767759.svg
  :target: https://doi.org/10.5281/zenodo.4767759
.. image:: https://img.shields.io/badge/ascl-2104.028-blue.svg?colorB=262255
 :target: https://ascl.net/2104.028
 :alt: ascl:2104.028

Installation
------------

The software can be pip installed from the PYPI repository via,

.. code:: bash

  pip install globalemu

or alternatively it can be installed from the git repository via.

.. code:: bash

  git clone https://github.com/htjb/globalemu.git # or the equivalent using ssh keys
  cd globalemu
  python setup.py install --user

Emulating the Global 21-cm Signal
---------------------------------

``globalemu`` is a fast and robust approach for emulating the Global or
sky averaged 21-cm signal and the associated neutral fraction history.
In the cited MNRAS paper below we show that it is
a factor of approximately 102 times faster and 2 times as accurate
as the previous state of the art
`21cmGEM <https://academic.oup.com/mnras/article/495/4/4845/5850763>`__. The
code is also flexible enough for it to be retrained on detailed simulations
containing the most up to date physics. We release two trained networks, one
for the Global signal and one for the neutral fraction history, details of
which can be found in the MNRAS paper below.

You can download trained networks with the following code after pip installing
or installing via the github repository:

.. code:: python

  from globalemu.downloads import download

  download().model() # Redshift-Temperature Network
  download(xHI=True).model() # Redshift-Neutral Fraction Network

which will produce two files in your working directory 'T_release/' and
'xHI_release/'. Each file has the respective network model in and related
pre and post processing files. You can then go on to evaluate each network for
a set of parameters by running:

.. code:: python

  from globalemu.eval import evaluate

  # [fstar, vc, fx, tau, alpha, nu_min, R_mfp]
  params = [1e-3, 46.5, 1e-2, 0.0775, 1.25, 1.5, 30]

  predictor = evaluate(base_dir='T_release/') # Redshift-Temperature Network
  signal, z = predictor(params)

  # note the parameter order is different for the neutral fraction emulator
  # [fstar, vc, fx, nu_min, tau, alpha, R_mfp]
  params = [1e-3, 46.5, 1e-2, 1.5, 0.0775, 1.25, 30]

  predictor = evaluate(base_dir='xHI_release/', xHI=True) # Redshift-Neutral Fraction Network
  signal, z = predictor(params)

Results are accessed via 'res.z' and 'res.signal'.

The code can also be used to train a network on your own Global 21-cm signal
or neutral fraction simulations using the built in ``globalemu`` pre-processing
techniques. There is some flexibility on the required astrophysical input
parameters and the pre-processing steps which is detailed in the documentation.
More details about training your own network can be found in the documentation.

``globalemu`` GUI
-----------------

``globalemu`` also features a GUI that can be invoked from the command line
and used to explore how the structure of the Global 21-cm signal varies with
the values of the astrophysical inputs. The GUI needs a configuration file to
run and this can be generated using a built in ``globalemu`` function.
**GUI configuration files can be generated for any trained model.** For example,
if we wanted to generate a configuration file for the released Global signal
emulator we would run,

.. code:: python

  from globalemu.gui_config import config

  paramnames = [r'$\log(f_*)$', r'$\log(V_c)$', r'$\log(f_X)$',
                r'$\tau$', r'$\alpha$', r'$\nu_\mathrm{min}$',
                r'$R_\mathrm{mfp}$']

  config('T_release/', paramnames, 'data/')

where the directory 'data/' contains the training and testing data (in this
case that corresponding to
`21cmGEM <https://zenodo.org/record/4541500#.YKOTiibTWWg>`__).

The GUI can then be invoked from the terminal via,

.. code:: bash

  globalemu /path/to/base_dir/T_release/etc/

An image of the GUI is shown below.

.. image:: https://github.com/htjb/globalemu/raw/master/docs/images/gui.png
  :width: 400
  :align: center
  :alt: graphical user interface

The GUI can also be used to investigate the physics of the neutral fraction
history by generating a configuration file for the released trained model.
There is no need to specify that the configuration file is for a neutral
fraction emulator.

Configuration files for the released models are provided.

Documentation
-------------

The documentation is available at: https://globalemu.readthedocs.io/

It can be compiled locally after downloading the repo and installing
the relevant packages (see below) via,

.. code:: bash

  cd docs
  sphinx-build source html-build

You can find a tutorial notebook
`here <https://mybinder.org/v2/gh/htjb/globalemu/master?filepath=notebooks%2F>`__.

T_release/ and xHI_release/
---------------------------

The currently released global signal trained model, ``T_release/`` is trained
on the same training data set as 21cmGEM which is available
`here <http://doi.org/10.5281/zenodo.4541500>`__. The data used to train the
neutral fraction history network, ``xHI_release/`` is not publicly available
but comes from the same large scale simulations used to model the global signal.

For both models the input parameters and ranges are given below.

.. list-table::
  :header-rows: 2

  * - Parameter
    - Description
    - ``T_release/``
    - ``xHI_release/``
    - Min
    - Max
  * -
    -
    - Input Order
    - Input Order
    -
    -
  * - f\ :sub:`*`
    - Star Formation Efficiency
    - 1
    - 1
    - 0.0001
    - 0.5
  * - V\ :sub:`c`
    - Minimal Virial Circular Veloity
    - 2
    - 2
    - 4.2 km/s
    - 100 km/s
  * - f\ :sub:`x`
    - X-ray Efficiency
    - 3
    - 3
    - 0
    - 1000
  * - tau
    - CMB Optical Depth
    - 4
    - 5
    - 0.04
    - 0.17
  * - alpha
    - Power of X-ray SED slope
    - 5
    - 6
    - 1.0
    - 1.5
  * - nu :sub:`min`
    - Low Energy Cut Off of X-ray SED
    - 6
    - 4
    - 0.1 keV
    - 3 keV
  * - R\ :sub:`mfp`
    - Mean Free Path of Ionizing Photons
    - 7
    - 7
    - 10.0 Mpc
    - 50.0 Mpc

Licence and Citation
--------------------

The software is free to use on the MIT open source license. If you use the
software for academic puposes then we request that you cite the
``globalemu`` paper below.

`MNRAS pre-print <https://arxiv.org/abs/2104.04336>`__
(referred to in the documentation as the ``globalemu`` paper),

  Bevins, H., W. J. Handley, A. Fialkov, E. D. L. Acedo and K. Javid.
  “GLOBALEMU: A novel and robust approach for emulating the sky-averaged 21-cm
  signal from the cosmic dawn and epoch of reionisation.” (2021). arXiv:2104.04336

Below is the bibtex,

.. code:: bibtex

  @article{Bevins2021,
    title = {{GLOBALEMU}: {A} novel and robust approach for emulating the sky-averaged 21-cm signal from the cosmic dawn and epoch of reionisation},
    url = {http://arxiv.org/abs/2104.04336},
    urldate = {2021-04-12},
    journal = {arXiv:2104.04336 [astro-ph]},
    author = {Bevins, H. T. J. and Handley, W. J. and Fialkov, A. and Acedo, E. de Lera and Javid, K.},
    month = apr,
    year = {2021},
    note = {arXiv: 2104.04336}
  }

Requirements
------------

To run the code you will need to following additional packages:

- `numpy <https://pypi.org/project/numpy/>`__
- `tensorflow <https://pypi.org/project/tensorflow/>`__
- `pandas <https://pypi.org/project/pandas/>`__
- `matplotlib <https://pypi.org/project/matplotlib/>`__
- `Pillow <https://pypi.org/project/Pillow/>`__

When installing via pip or from source via setup.py the above packages will
be installed if absent.

To compile the documentation locally you will need:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

To run the test suit you will need:

- `pytest <https://docs.pytest.org/en/stable/>`__

Contributing
------------

Contributions to ``globalemu`` are very much welcome and can be made via,

- Opening an issue to report a bug/propose a new feature.
- Making a pull request. Please consider opening an issue first to discuss
  any proposals and ensure the PR will be accepted.
