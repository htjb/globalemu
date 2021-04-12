---
title: 'globalemu: A flexible and fast framework for emulating the sky-averaged 21-cm signal from
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
date: 12 April 2021
bibliography: paper.bib

---

# Summary

``globalemu`` is a robust framework for emulating the sky-averaged
or Global 21-cm signal from the Cosmic Dawn and Epoch of Reionisation. While
a detection of this signal has not yet been confirmed the physics of the signal
is broadly understood within a theoretical framework. Detailed simulations of
the signal exist and are continually being updated as our understanding of the
physics improves [@Visbal2012, @Fialkov2014, @Cohen2017, @Reis2021]. Each
simulation takes a few hours to perform on a desktop [@Monsalve2019] and
consequently the use of neural networks to emulate the signal has been pursued.
``globalemu`` is a framework capable of emulating a high resolution Global signal in 1.3 ms
which is a factor of approximately 102 improvement on the previous
state of the art [@Cohen2020]. It is
designed to be a flexible emulator that can be easily retrained by a user on
the latest set of simulations containing the most up to date understanding
of the physics.

# Statement of Need

A detection of an absorption trough at $78$ MHz was recently made by the
Experiment to Detect the Global Epoch of Reionization Signature (EDGES) [@Bowman2018].
Efforts are currently underway to confirm this detection
and there are concerns about the data analysis in the
EDGES experiment [@Hills2018; @Singh2019; @Sims2020; @Bevins2021].

The EDGES collaboration modelled the absorption feature in their data with a
flattened Gaussian profile characterised by an amplitude, width, flattening
factor and central frequency. This type of non-physical signal modelling is
common in the data analysis for experiments attempting to detect the Global
21-cm signal. However, the models tell us very little about the physics of the
early universe.

A better approach is to use physical modelling of the signal where the signals
structure is determined by a set of characteristic astrophysical parameters
such as the star formation efficiency of early galaxies. Detailed physical
simulations of the global signal exist [@Visbal2012; @Fialkov2014; @Cohen2017; @Reis2021]
and are commonly used to explore the parameter space. However, to produce
one signal from the above simulations between $z = 6 - 50$ with $\delta z = 1$
takes several hours on a desktop computer [@Monsalve2019]. This is impractical
when attempting to fit physical signal models, particularly when using
nested sampling algorithms [@Anstey2020; @Liu2020; @Chatterjee2021].

As a result it has been proposed and demonstrated that we can use neural networks
to emulate the physical signal models in a fraction of a second given enough
available training data. The previous state of the art, ``21cmGEM`` [@Cohen2020],
can emulate a high resolution signal, 451 redshift data points, in 133 ms [@Bevins2021b]
given a set of seven astrophysical parameters that detail the physics of the
first galaxies and stars to form in the universe.

``21cmGEM`` relies on Principle Component Analysis, a compression of
the parameter space that can result in a loss of information, and several neural networks to
produce an accurate emulation of the Global 21-cm signal. In contrast ``globalemu``
uses the novel approach of having redshift as an input to the neural network,
alongside the characteristic astrophysical parameters, and estimating a
corresponding temperature.

``globalemu`` therefore typically has around seven inputs and one output in
comparison to ``21cmGEM`` which has seven to twelve inputs, one to
seven outputs, multiple networks (5 or 6) and a decision tree for classification.
This combined with the fact that ``globalemu`` has a detailed physically motivated
pre-processing (see @Bevins2021b) of the data means that we can
emulate the Global signal with one small scale neural network to a high degree of
accuracy.

![**Left Panel:** The structure of one of five or six regression neural networks
used in ``21cmGEM`` to emulate the Global signal. ``21cmGEM`` uses twelve astrophysical
parameters and these are labelled at the input nodes [for details see @Cohen2020].
We can see that there are multiple outputs and each network has one hidden layer with
40 nodes. **Right Panel:** In comparison, the structure
of ``globalemu`` which we have illustrated with seven astrophysical parameters and
redshift as input and a single output from the
network. The novel network design allows for a typically smaller
network architecture and a correspondingly quicker emulation.
Figure taken from @Bevins2021b.](network_design.png)

It is demonstrated in @Bevins2021b that ``globalemu`` can emulate the
Global signal in 1.3 ms, a factor of 102 improvement on ``21cmGEM``, and that
it is approximately twice as accurate as ``21cmGEM`` when emulating the same
set of signals.

We release with ``globalemu`` trained neural networks for both the Global 21-cm
signal and the corresponding neutral fraction history. However, we note that
``globalemu`` is a detailed framework that makes emulating the Global 21-cm signal
easy to do and consequently a user can retrain the emulator on new sets of
simulations with updated astrophysics such as the addition of Lyman-$\alpha$
heating [@Reis2021] or an increased radio background [@Reis2020]. ``globalemu``
will be used extensively by the Radio Experiment for the Analysis of
Cosmic Hydrogen [@Acedo2019] for physical signal modelling.

Documentation for ``globalemu`` is available at
[ReadTheDocs](https://globalemu.readthedocs.io/). The code is
pip installable ([PyPI](https://pypi.org/project/globalemu/)) and available
on [Github](https://github.com/htjb/globalemu/). Continuous integration
is performed with [Github Actions](https://github.com/htjb/globalemu/actions)
and the corresponding coverage is
reported by [CodeCov](https://app.codecov.io/gh/htjb/globalemu).

# Acknowledgements

Discussions on the applications of the software were provided by Eloy de Lera Acedo,
Will Handley and Anastasia Fialkov. The author is supported by the Science and
Technology Facilities Council (STFC) via grant number ST/T505997/1.

# References
