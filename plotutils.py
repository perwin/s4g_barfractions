# Plotting-related code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import s4gutils



# Nicer-looking logarithmic axis labeling

def niceLogFunc( x_value, pos ):
	return ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x_value),0)))).format(x_value)	
NiceLogFormatter = ticker.FuncFormatter(niceLogFunc)

def MakeNiceLogAxes( whichAxis="xy", axisObj=None ):
	"""
	Makes one or more axes of a figure display tick labels using non-scientific
	notation (e.g., "0.01" instead of "10^{-2}")
	"""
	
	if axisObj is None:
		ax = plt.gca()
	else:
		ax = axisObj
	if whichAxis in ["x", "xy"]:
		ax.xaxis.set_major_formatter(NiceLogFormatter)
	if whichAxis in ["y", "xy"]:
		ax.yaxis.set_major_formatter(NiceLogFormatter)



def PlotFrequencyWithWeights( values, weights, i_1, i_non1, start, stop, step, offset=0,
							noErase=False, axisObj=None, **kwargs ):

	"""
	Plots frequencies with binomial 68% confidence intervals (as error bars) in
	bins of values, where i_1 are indices defining detections and i_non1 define
	non-detections. Includes the ability to handle weights for individual
	data points.
	
	values = vector of values for each object (e.g., log M_star)
	weights = vector of weights for each object
	i1 = list of indices into values for objects in parent sample 
		and in group 1 (e.g., barred galaxies)
	i_non1 = list of indices into values for objects in parent sample 
		but *not* in group 1 (e.g., unbarred galaxies)
	start, stop, step = start, stop, and spacing for bins

	offset = optional additive offset to x [values] positions of plotted points
	
	noErase = if False, then plt.clf() is called before plotting
	
	axisObj = optional matplotlib Axes object where plot will be placed
	
	**kwargs = extra keyword=value pairs passed to plt.errorbar
	
	Example:
	PlotFrequencyWithWeights(logMstar, galaxyWeights, ii_boxy, ii_nonboxy, 9.0, 11.6, 0.2)
	"""
	binranges = np.arange(start, stop, step)
	n1,bins = np.histogram(np.array(values)[i_1], bins=binranges)
	n2,bins = np.histogram(np.array(values)[i_non1], bins=binranges)
	nBins = len(n1)

	i_all = i_1 + i_non1
	i_all.sort()
		
	(n_1, bin_edges) = np.histogram(values[i_1], binranges, weights=weights[i_1])
	(n_all, bin_edges) = np.histogram(values[i_all], binranges, weights=weights[i_all])

	values_valid_unweighted = np.array([ values[i] for i in i_all if weights[i] > 0 ])
	(n_all_unwt, junk) = np.histogram(values_valid_unweighted, bins=binranges)
	scaleFactors = n_all / n_all_unwt
	n_1_normalized = n_1 / scaleFactors
	
	# we calculate fractions (and confidence limits) using renormalized numbers, so that
	# the total in each bin is the total number of objects in each bin (rather than the
	# *weighted* total)
	(frac_1, delta_low, delta_high) = s4gutils.Binomial(n_1_normalized, n_all_unwt)

	# midpoints of the bins:
	midvals = np.zeros(nBins)
	for i in range(nBins):
		midvals[i] = 0.5*(bin_edges[i] + bin_edges[i + 1])

	if noErase is False:
		plt.clf()
	if axisObj is None:
		plt.errorbar(midvals + offset, frac_1, [delta_low,delta_high], None, elinewidth=1.2, 
					capthick=1.2, capsize=5, **kwargs)
	else:
		axisObj.errorbar(midvals + offset, frac_1, [delta_low,delta_high], None, elinewidth=1.2, 
					capthick=1.2, capsize=5, **kwargs)

	return (midvals, frac_1, delta_low, delta_high)


def PlotFrequency( values, i_1, i_non1, start, stop, step, offset=0, noErase=False, 
					axisObj=None, debug=False, **kwargs ):
					
	"""
	Plots frequencies with binomial 68% confidence intervals (as error bars) in
	bins of values, where i_1 are indices defining detections and i_non1 define
	non-detections.

	values = vector of values (e.g., M_star)
	i1 = list of indices into values for objects in parent sample 
		and in group 1 (e.g., barred galaxies)
	i_non1 = list of indices into values for objects in parent sample 
		but *not* in group 1 (e.g., unbarred galaxies)
	start, stop, step = start, stop, and spacing for bins
	
	offset = optional additive offset to x [values] positions of plotted points
	
	axisObj = optional matplotlib Axes object where plot will be placed
	
	**kwargs = extra keyword=value pairs passed to plt.errorbar

	Example:
	PlotFrequency(logMstar, ii_boxy, ii_nonboxy, 9.0, 11.6, 0.2)
	"""
	binranges = np.arange(start, stop, step)
	n1,bins = np.histogram(np.array(values)[i_1], bins=binranges)
	n2,bins = np.histogram(np.array(values)[i_non1], bins=binranges)
	nBins = len(n1)

	ff = []
	ff_low = []
	ff_high = []
	for i in range(nBins):
		if n1[i] + n2[i] == 0:
			f = f_low = f_high = np.nan
		else:
			f,f_low,f_high = s4gutils.Binomial(n1[i], n1[i] + n2[i])
		ff.append(f)
		ff_low.append(f_low)
		ff_high.append(f_high)
	x = np.array([0.5*(bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
	if debug:
		print(bins)
		print("n1 (i_1) = ", n1)
		print("n2 (i_non1) = ", n2)
		print(ff)
	
	if noErase is False:
		plt.clf()
	if axisObj is None:
		plt.errorbar(x + offset, ff, [ff_low,ff_high], None, elinewidth=1.2, 
					capthick=1.2, capsize=5, **kwargs)
	else:
		axisObj.errorbar(x + offset, ff, [ff_low,ff_high], None, elinewidth=1.2, 
					capthick=1.2, capsize=5, **kwargs)


