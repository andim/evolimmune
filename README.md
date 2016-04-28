# Diversity of immune strategies explained by adaptation to pathogen statistics

This repository contains the source code associated with the manuscript

Mayer, Mora, Rivoire, Walczak : [Diversity of immune strategies explained by adaptation to pathogen statistics](http://dx.doi.org/10.1101/033159), bioRxiv 2015

It allows reproduction of all numerical results reported in the manuscript.

## Installation requirements

The code uses Python 2.7+.

A number of standard scientific python packages are needed for the numerical simulations and visualizations. An easy way to install all of these is to install a Python distribution such as [Anaconda](https://www.continuum.io/downloads).

- [numpy](http://github.com/numpy/numpy/)
- [scipy](https://github.com/scipy/scipy)
- [pandas](http://github.com/pydata/pandas)
- [matplotlib](http://github.com/matplotlib/matplotlib)

Additionally the code also relies on these packages:

- [shapely](http://github.com/Toblerity/Shapely)
- [palettable](http://github.com/jiffyclub/palettable)
- [scipydirect](http://github.com/andim/scipydirect/)
- [noisyopt](http://github.com/andim/noisyopt)

And optionnaly for nicer progress output install:

- [pyprind](http://github.com/rasbt/pyprind)

## Running the code

The time stepping of the population dynamics is accelerated by a Cython module, which needs to be compiled first. To compile it run `make cython` in the `lib` directory. In the directories for the different figures launch `make run` followed by `make agg` to produce the underlying data. Please copy the `paper.mplstyle` to your custom matplotlib style directory (likely `.config/matplotlib/stylelib/`). We provide both Jupyter notebooks with additional explanatory comments and plain python files for generating the figures.

## Remarks

In the code we use the following simplified notations `c_constitutive = mu1, c_defense = mu2, c_infection = lambda_, c_uptake = cup` and we define the tradeoff `c_defense(c_constitutive)` as a parametric function of a parameter `epsilon` in [0, 1], where 0 corresponds to fully constitutive and 1 to maximally regulated responses.

## Contact

If you run into any difficulties running the code, feel free to contact us at `andisspam@gmail.com`.

## License

The source code is freely available under an MIT license.
