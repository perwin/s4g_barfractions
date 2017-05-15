# Public data, code, and notebooks for bar-frequencies paper using S4G data

This git repository contains data files, Python code, and Python and R Jupyter
notebooks which can be used to reproduce figures from the paper "The Dependence of Bar Frequency 
on Galaxy Mass, Colour, and Gas Content -- and Angular Resolution -- in the Local Universe".

The `data/` subdirectory contains text-file tables with various data compilations
and simulation outputs.

## Prerequisites

The code and notebooks require the following Python modules and packages:

   * [Numpy](https://www.numpy.org), [Scipy](https://www.scipy.org), [matplotlib](https://matplotlib.org)
   * [Astropy](https://www.astropy.org)
   * Michele Cappellari's LOESS code [`cap_loess_1d`](http://www-astro.physics.ox.ac.uk/~mxc/software/#loess)

The R notebook requires the "survey" package.
