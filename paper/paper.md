---
title: 'globalemu: robust and fast emulation of the sky-averaged 21-signal from
  the cosmic dawn and epoch of reionization'
tags:
  - Python
  - astrophysics
  - cosmology
authors:
  - name: Harry T. J. Bevins
    orcid: 0000-0002-4367-3550
    affiliation: "1"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, United Kingdom
   index: 1
date: 31 July 2020
bibliography: paper.bib

---

# Summary

``globalemu`` is a robust framework for training and emulating the sky-averaged
or Global 21-cm signal from the Cosmic Dawn and Epoch of Recombination. While
a detection of this signal has not yet been confirmed the physics of the signal
is broadly understood within a theoretical framework. Detailed simulations of
the signal exist and are continually being updated as our understanding of the
physics improves [@Visbal2012, @Fialkov2014, @Cohen2017, @Reis2021]. Each
simulation takes a few hours to perform on a desktop [@Monsalve2019] and
consequently the use of neural networks to emulate the signal has been pursued.
``globalemu`` is capable of emulating a high resolution Global signal in 8 ms
which is a factor of 20 improvement on the previous state of the art [@Cohen2020]. It is
designed to be a flexible emulator that can be easily retrained by a user on
the latest set of simulations containing the most up to date understanding
of the physics.

# Statement of Need

A detection of an absorption trough at $78$ MHz was recently made by the
Experiment to Detect the Global Epoch of Reionization Signature (EDGES) [@Bowman2018].
Efforts are currently underway to determine whether this signal is indeed the
Global 21-cm signal and there are concerns about the data analysis in the
EDGES experiment [@Hills2018, @Singh2019, @Sims2020, @Bevins2021].

The EDGES collaboration modelled the absorption feature in the data with a
flattened gaussian profile characterised by an amplitude, width, flattening
factor and central frequency. This type of non-physical signal modelling is
common in the data analysis for experiments attempting to detect the Global
21-cm signal. However, the models tell us very little about the physics of the
early universe.

A better approach is to use physical modelling of the signal where the signals
structure is determined by a set of characteristic astrophysical parameters
such as the star formation efficiency of early galaxies. Detailed physical
simulations of the global signal exist [@Visbal2012, @Fialkov2014, @Cohen2017, @Reis2021]
and are commonly used to explore the parameter space. However, to produce
one signal from the above simulations between $z = 6 - 50$ with $\delta z = 1$
takes several hours on a desktop computer [@Monsalve2019]. This is impractical
when attempting to fit physical signal models, particularly when using
nested sampling algorithms [@Anstey2020, @Liu2020, @Chatterjee2021].

As a result it has been proposed and demonstrated that we can use neural networks
to emulate the physical signal models in a fraction of a second given enough
available training data. The previous state of the art, ``21cmGEM`` [@Cohen2020],
can emulate a high resolution signal, 451 redshift data points, in 160 ms
given a set of seven astrophysical parameters that detail the physics of the
first galaxies and stars to form in the universe.

``21cmGEM`` relies on Principle Component Analysis, a potentially compression of
the parameter space that can result in a loss of information, and several neural networks to
produce an accurate emulation of the Global 21-cm signal. In contrast ``globalemu``
uses the novel approach of having redshift as an input to the neural network,
alongside the characteristic astrophysical parameters, and estimating a
corresponding temperature. Whilst this means multiple calls to ``globalemu``
need to be made to estimate the signal temperature as a function of redshift
we find that this is not an issue.

``globalemu`` therefore has typically around seven inputs and one output in
comparison to seven inputs and $\gtrsim 10$ outputs and multiple networks for
emulators like ``21cmGEM``.
This combined with the fact that ``globalemu`` has a detailed physically motivated
pre-processing (see Bevins et al. (in prep.)) of the data means that we can
emulate the Global signal with one small scale neural network to a high degree of
accuracy.

![**Left Panel:** The structure of a traditional Global 21-cm
signal emulator. We illustrate the input astrophysical parameters with those
corresponding to the data used to train ``21cmGEM`` [@Cohen2020]. We can see that
there are multiple outputs and consequently a large number of hidden layers and
nodes are typically needed for accurate emulaton. **Right Panel:** The structure
of ``globalemu`` with one additional input and only a single output from the
network. Whilst a vectorised call needs to be made to emulate a signal at
multiple redshifts, the novel network design allows for a typically smaller
network architecture and a correspondingly quicker emulation than the
alternative approach. Figuere taken from Bevins et al. (in prep.).](network_design.png)

We demonstrate in Bevins et al. (in prep.) that ``globalemu`` can emulate the
Global signal in 8 ms, a factor of 20 improvement on ``21cmGEM``, and that
it is approximately twice as accurate as ``21cmGEM`` when emulating the same
set of signals.

We release with ``globalemu`` trained neural networks for both the Global 21-cm
signal and the corresponding neutral fraction history. However, the we note that
``globalemu`` is a detailed framework that makes emulating the Global 21-cm signal
easy to do and consequently a user can retrain the emulator on new sets of
simulations with updated astrophysics such as the addition of Lyman-$\alpha$
heating [@Reis2021] or an increased radio background [@Reis2020]. ``globalemu``
will be used extensively by the Radio Experiment for the Analysis of
Cosmic Hydrogen [@Acedo2019] for physical signal modelling.

Documentation for ``globalemu`` is available at [ReadTheDocs](). The code is
pip installable ([PyPI]()) and available on [Github](). Continuous integration
is performed with [Github Actions]() and the corresponding coverage is
reported by [CodeCov]().

# Acknowledgements

Discussions on the applications of the software were provided by Eloy de Lera Acedo,
Will Handley and Anastasia Fialkov. The author is supported by the Science and
Technology Facilities Council (STFC) via grant number ST/T505997/1.

# References