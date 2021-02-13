===============================================
GlobalEmu: Robust Global 21-cm Signal Emulation
===============================================

TO DO:
------

* Paper: Need to run 'release' versions of trained networks. Probably on 21cmGEM
  train data without any effort to make the data 'uniform' as not needed
  to illustrate network works. *Make sure to reorder xHI network inputs appropriately
  and fix gui'*
* Sort GUI tex labels
* Tests: Some can't be run until the repo is public because of download
  requirements but I have tested the relevant code briefly already.
* setup.py: May need some edits
* Travis and Circle ci
* Code cov once tests are sorted
* Finish README
* Documentation
* Example Jupyter notebooks

Introduction
------------

:GlobalEmu: Robust Global 21-cm Signal Emulation
:Author: Harry Thomas Jones Bevins
:Version: 1.0.0
:Homepage: https://github.com/htjb/globalemu
:Documentation: https://globalemu.readthedocs.io/

.. image:: https://github.com/htjb/GlobalEmu/workflows/CI/badge.svg?event=push
  :target: https://github.com/htjb/GlobalEmu/actions?query=workflow%3ACI
  :alt: github CI

Installation
------------

Emulating the Global 21-cm Signal
---------------------------------

You can download trained networks with the following code after pip installing
or installing via the github repository:

.. code::

  from globalemu.downloads import download

  download(False).model() # Redshift-Temperature Network
  download(True).model() # Redshift-Neutral Fraction Network

which will produce two files in your working directory 'T_release/' and
'xHI_release/'. Each file has the respective network model in and related
pre and post processing files. You can then go on to evaluate each network for
a set of parameters by running:

.. code::

  from globalemu.eval import evaluate

  # [fstar, vc, fx, tau, alpha, nu_min, R_mfp]
  params = [1e-3, 46.5, 1e-2, 0.0775, 1.25, 1.5, 30]

  res = evaluate(params, base_dir='T_release/') # Redshift-Temperature Network
  res = evaluate(params, base_dir='xHI_release/', xHI=True) # Redshift-Neutral Fraction Network

Results are accessed via 'res.z' and 'res.signal'.

The code can also be used to train a network on your own Global 21-cm signal
or neutral fraction simulations using the built in globalemu pre-processing
techniques. There is some flexibility on the required astrophysical input
parameters but the models are required to subscribe to the astrophysics free
baseline calculation detailed in the GlobalEmu paper (see below for a reference).
More details about training your own network can be found in the documentation.


GlobalEmu GUI
-------------

Licence and Citation
--------------------

Requirements
------------

Contributing
------------

21cmGEM Data
------------

The training data is available `here <https://people.ast.cam.ac.uk/~afialkov/>`__
