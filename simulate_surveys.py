#!/usr/bin/env python
# -*- coding: utf8 -*-

# Python code for simulating barred-galaxy observations in limited-resolution
# surveys, esp. SDSS-based

# The basic idea: generate a set of N galaxies at various redshifts by bootstrap
# sampling from a distance-limited subset of the S4G galaxies. Then determine
# the bar fraction by "observing" each galaxy to see if its (projected) bar size
# is larger than some user-specified angular size limit (e.g., some multiple of
# the typical seeing FWHM for the reference survey).

# 	Use existing S4G dataset
# 		1. Simple version: use *observed* bar sizes and inclinations
# 		2. More complex version [what we used for the paper]: use deprojected bar sizes, 
#		then apply random projection (random bar PA from uniform sampling, random 
#		inclination from correct inclination sampling)
#			-- Also requires adjusting vmaxg *if* we're doing H I flux-limit
#			checks, since different inclination -> different vmaxg
# 		
# 	Assume some cutoff on bar detection based on size (e.g. 2 x FWHM)
# 	
# 	Randomly select redshift from volume-dependent density (in Euclidean approx.,
# 	P(z) prop.to z^3 out to sample redshift limit)
# 	
# 	Select S4G galaxy at random from S4G sample (sample with replacement)
# 		If sample is mag-limited (Nair & Abraham), determine galaxy apparent magnitude
# 		and see if it stays in sample
# 		
# 	Store galaxy stellar mass, gas mass fraction, etc.
# 	
# 	Compute observed bar size in arcsec, compare with cutoff
# 		if detected, mark galaxy as "barred" & store observed bar size
# 		if not detected, mark galaxy as "unbarred"



import os, math, random
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# Cosmology: Flat LambdaCDM with H_0 = 70, Omega_matter = 0.29
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.29)

import s4gutils
import datautils


# Read in composite data table for Parent Disk Sample from S4G
columnHeaderRow = 25
s4gdata = datautils.ReadCompositeTable('data/s4gbars_table.dat', columnRow=columnHeaderRow, 
										dataFrame=True)


# Generate spline-interpolation objects for cosmology calculations (*much* faster
# than repeated calls to original astropy.cosmology.FlatLambdaCDM methods)
print("Generating comology interpolation functions...")
zz = np.arange(0.001, 1.1, 0.001)
lumDistances = np.array([cosmo.luminosity_distance(z).value for z in zz])
arcsecScales = np.array([cosmo.arcsec_per_kpc_proper(z).value for z in zz])
luminosityDistFn = InterpolatedUnivariateSpline(zz, lumDistances)
arcsecPerKpcFn = InterpolatedUnivariateSpline(zz, arcsecScales)


maxInclination = 60.0
minCosValue = math.cos(math.radians(maxInclination))


PI_OVER_TWO = math.pi/2.0

random.seed()


# construct dataset vectors, using min(logMstar) = 9.0 and max(distance) = 25 Mpc
nDisksTot = len(s4gdata.name)
ii_gmr = [i for i in range(nDisksTot) if s4gdata.gmr_tc[i] > -1]
nTot_gmr = len(ii_gmr)

index_MHI = 9
index_W = 10
index_inc = -1


def MakeS4GSubsample( distLimit=30.0, logMstarLimit=9.0, tLimit=-0.4 ):
	"""
	Generates a subsample of S4G galaxies.

    Parameters
    ----------

	distLimit: float, optional
		only galaxies with distances <= this value are kept
		
	logMstarLimit: float, optional
		only galaxies with log(M_star/M_sun) >= this value are kept
		
	tLimit : float, optional
		only galaxies with Hubble type T >= this value are kept
		[default = -0.4, so all spirals are kept]
	
    Returns
    -------
	list of galaxy-data tuples, one per galaxy
		Each tuple has (name, logmstar, sma, sma_kpc, sma_dp_kpc2, gmr_tc, logfgas,
			vmax_weight, weight_BmVtc, M_HI, W_gas_dp, M_B, inclination)
	"""
	dset = []
	for i in range(nDisksTot):
		if ((s4gdata.dist[i] <= distLimit) and (s4gdata.logmstar[i] >= logMstarLimit)
			and (s4gdata.t_s4g[i] > tLimit)):
			distMpc = s4gdata.dist[i]
			Btc = s4gdata.B_tc[i]
			M_B = Btc - s4gutils.dtomm(Btc)
			W_gas = s4gdata.W_gas[i]
			inclination = s4gdata.inclination[i]
			W_gas_dp = W_gas / math.sin(math.radians(inclination))
			if distLimit == 25.0:
				vmax_weight = s4gdata.w25[i]
			elif distLimit == 40.0:
				vmax_weight = s4gdata.w40[i]
			else:
				vmax_weight = s4gdata.w30[i]
			galTuple = (s4gdata.name[i], s4gdata.logmstar[i], s4gdata.sma[i], s4gdata.sma_kpc[i], 
						s4gdata.sma_dp_kpc2[i], s4gdata.gmr_tc[i], s4gdata.logfgas[i], 
						vmax_weight, s4gdata.weight_BmVtc[i], s4gdata.M_HI[i], W_gas_dp, 
						M_B, inclination)
			dset.append(galTuple)
	return dset

# Make some useful standard S4G subsamples:	
# dset_25 = D <= 25 Mpc and logMstar >= 9
dset_d25 = MakeS4GSubsample(distLimit=25, tLimit=-3.4)
# dset_30 = D <= 30 Mpc and logMstar >= 9
dset_d30 = MakeS4GSubsample(tLimit=-3.4)
# same as previous, but without S0 galaxies
dset_d30_sp = MakeS4GSubsample()
# dset_30m95 = D <= 30 Mpc and logMstar >= 9.5
dset_d30m95 = MakeS4GSubsample(logMstarLimit=9.5, tLimit=-3.4)
# same as previous, but without S0 galaxies
dset_d30m95_sp = MakeS4GSubsample(logMstarLimit=9.5)

	

def WithinMagLimit( galaxyData, distModulus, maglimit ):
	"""Returns True if galaxy has m_B brighter than maglimit
	
    Parameters
    ----------
	galaxyData : tuple of individual-galaxy data, as generated by MakeS4GSubsample
	
	distModulus : float
		distance modulus for galaxy
	
	maglimit : float
		apparent B-magnitude limit
	
    Returns
    -------
    bool
	"""
	M_B = galaxyData[-1]
	m_B = M_B + distModulus
	if m_B > maglimit:
		return True
	else:
		return False


def HIFluxLimit( W50 ):
	"""Calculate limiting H I flux for galaxy observed by ALFALFA, using
	Eqn. 7 of Haynes+2011 (AJ, 142, 170)

    Parameters
    ----------
	W50 : float
		50% width of H I line, or equivalent
		
    Returns
    -------
    limit_logS_21 : float
    	Limiting H I flux in log(S_21), where S_21 is Jy km s^-1
	"""
	logW50 = math.log10(W50)
	# calculate ALFALFA alpha40 50% limiting flux for this W_50
	if logW50 < 2.5:
		limit_logS_21 = 0.5*logW50 - 1.24
	else:
		limit_logS_21 = logW50 - 2.46
	return limit_logS_21


def WithinHILimit( galaxy, distMpc, inclination ):
	"""Determine if specified galaxy [tuple or list which includes HyperLeda-derived
	M_HI and vmaxg] at distance distMpc would be detected in H I by ALFALFA, using 
	50% detection limit formula from Haynes+2011, implemented in HIFluxLimit().

    Parameters
    ----------
	galaxy : tuple of individual-galaxy data, as generated by MakeS4GSubsample

    distMpc : float
    	galaxy distance in Mpc
    
    inclination : float
    	galaxy inclination in degrees

    Returns
    -------
    bool
	"""
	M_HI = galaxy[index_MHI]
	if (M_HI > 1e12):
		# crazy-high value indicates no m21c data in HyperLeda
		return False
	vmaxg_dp = galaxy[index_W]
	if (vmaxg_dp <= 0):
		# negative W_gas value indicates no vmaxg data in HyperLeda
		return False
	vmaxg_obs = vmaxg_dp * math.sin(math.radians(inclination))
	# use 2 * HyperLeda vmaxg as substitute for W50
	W50 = 2*vmaxg_obs
	logS_21 = math.log10(s4gutils.HIMassToFlux(galaxy[index_MHI], distMpc))
	if (logS_21 > HIFluxLimit(W50)):
		return True
	else:
		return False


def GetRandomGalaxy( dset, z, maglimit=None, useObservedSize=False, useHILimit=False,
					maxInclination=60.0 ):
	"""Returns a randomly selected galaxy from the list dset (assumed to be a subset
	of S4G sample, generated by MakeS4GSubsample); galaxy is at redshift z,
	which may be randomly generated (i.e., not the galaxy's original redshift).

    Parameters
    ----------
	dset : list of galaxy-data tuples from parent S4G subsample, as generated by 
		MakeS4GSubsample
	
	z : float
		redshift which will be assigned to galaxy
	
	maglimit : float or None, optional
		Apparent B-magnitude limit. If not None, then galaxies are randomly selected 
		until apparent magnitude (based on M_B and redshift z) is brighter than
		maglimit.
	
	useObservedSize : bool, optional
		If False, then a random new inclination will be chosen for the galaxy; if H I 
		limits are being used, the HyperLeda vmaxg value of the original galaxy will be 
		adjusted to account for the different inclination before being used for W50 in 
		computing the ALFALFA H I limit. 
		If True, then the original S4G galaxy's inclination will be used, and the
		original HyperLeda vmaxg value will be used for W50.
	
	useHILimit : bool, optional
		If True, then galaxies are randomly selected until H I flux (based on M_HI and 
		redshift z, along with vmaxg value) is bright enough to be detectable at the 
		50% level by ALFALFA (see Haynes+2011).

	maxInclination : float, optional
		Maximum inclination (in degrees) for sampled galaxies

    Returns
    -------
	[g, inclination, z] : list of galaxy-data tuple, float, float
		g = individual-galaxy data tuple
		inclination = galaxy inclination in degrees
		z = redshift for galaxy (same as input)
	"""
	
	minCosValue = math.cos(math.radians(maxInclination))
	
	if maglimit is None and useHILimit is False:
		g = random.choice(dset)
		if useObservedSize is True:
			inclination = g[index_inc]
		else:
			# generate random inclination, weighted by cos(i)
			s = random.uniform(minCosValue, 1.0)
			inclination = math.degrees(math.acos(s))
		return [g, inclination, z]
	
	else:
		# now we have to loop until we find a galaxy that satisfies maglimit
		# and/or H I detectability
		done = False
		if maglimit is not None:
			checkMag = True
		else:
			checkMag = False
		if useHILimit is True:
			checkHI = True
		else:
			checkHI = False
		while not done:
			g = random.choice(dset)
			if useObservedSize:
				inclination = g[index_inc]
			else:
				# generate random inclination, weighted by cos(i)
				s = random.uniform(minCosValue, 1.0)
				inclination = math.degrees(math.acos(s))
			magOK = hiOK = False
			# hyper-accurate but slow approach: use astropy.cosmology method
			#distMpc = cosmo.luminosity_distance(z).value
			# faster approach: use spline interpolation; 
			# wrap call to spline interpolation object in float() to extract the actual
			# value from the 1-element numpy array that spline interpolation generates
			distMpc = float(luminosityDistFn(z))
			distModulus = s4gutils.dtomm(distMpc)
			if checkMag:
				magOK = WithinMagLimit(g, distModulus, maglimit)
			else:
				magOK = True
			if checkHI:
				hiOK = WithinHILimit(g, distMpc, inclination)
			else:
				hiOK = True
			if magOK and hiOK:
				return [g, inclination, z]



def MakeGalaxySample( dset, zRange, nGalaxies, maglimit=None, useObservedSizes=True, 
						useHILimit=False, maxInclination=60.0 ):
	"""
	Generate a sample of galaxies for a simulated survey
	
    Parameters
    ----------
	dset : parent S4G subsample (as generated by MakeS4GSubsample)
		
	zRange : 2-element sequence of float
		lower and upper redshift limits for survey; individual galaxies will be 
		assigned random, volume-weighted redshifts from within these limits
		If z_low = z_high, then all galaxies will have the same redshift.
		
	nGalaxies : int
		total number of galaxies for sample
		
	magLimit : float or None, optional
		If not None, then only galaxies with m_B brighter than this (using S4G galaxy 
		M_B and assigned redshift) will be kept

	useObservedSize : bool, optional
		If False, then a random new inclination will be chosen for the galaxy; if H I 
		limits are being used, the HyperLeda vmaxg value of the original galaxy will 
		be adjusted to account for the different inclination before being used for W50 
		in computing the ALFALFA H I limit. 
		If True, then the original S4G galaxy's inclination will be used, and the
		original HyperLeda vmaxg value will be used for W50.

	useHILimit : bool, optional
		If True, then only galaxies detectable at 50% level in ALFALFA survey will be
		kept.

	maxInclination : float, optional
		Maximum inclination (in degrees) for sampled galaxies
	
    Returns
    -------
	[[galaxyData,inclination,z], ...] : [ [galaxyData, float, float], [galaxyData, float, float], ... ]
		nGalaxies-long list of lists of [galaxyData, inclination, z], where
		galaxyData = tuple of galaxy data for an individual galaxy
		inclination = random inclination for that galaxy (or original S4G galaxy 
		inclination, if useObservedSizes is True)
		z = (randomly selected) redshift at which that galaxy will be observed
	"""
	z1, z2 = zRange
	if (z1 == z2):
		z = z1
	else:
		z = s4gutils.GetRadialSampleFromSphere(z1,z2)
	minCosValue = math.cos(math.radians(maxInclination))
	
	galaxySample = []
	for i in range(nGalaxies):
		g = GetRandomGalaxy(dset, z, maglimit, useObservedSize=useObservedSizes, 
							useHILimit=useHILimit, maxInclination=maxInclination)
		galaxySample.append(g)
	return galaxySample
	

def projectr( deltaPA_rad, i_rad, r ):
	"""Function to calculate a projected length, given an input in-plane position angle 
	(*relative to disk line-of-nodes*, *not* straight position angle east of north!) and 
	inclination, both in degrees, and an input (unprojected) length r.

    Parameters
    ----------
	deltaPA_rad : float
		angle between length (e.g., bar) and line of nodes, in radians
	
	i_rad : float
		galaxy inclination, in radians
	
	r : float
		length of object (e.g., bar) being projected
	
    Returns
    -------
    projected_length : float
	"""
	
	cosi = math.cos(i_rad)
	sindp = math.sin(deltaPA_rad)
	cosdp = math.cos(deltaPA_rad)
	# this is the deprojection scale, which we can invert
	scale = math.sqrt( (sindp*sindp)/(cosi*cosi) + cosdp*cosdp )
	return ( r / scale )


def ComputeBarSize( galaxyDataList, useObservedSize=True, getKpcSize=False ):
	"""
	Computes observed (projected) bar size in arc sec for galaxy described by galaxyDataList;
	optionally also returns the projected size in kpc
	
    Parameters
    ----------
	galaxyDataList : [galaxyData, float, float]
		[galaxyData, inclination, redshift]
		galaxyData = individual-galaxy data tuple, as generated by MakeS4GSubsample
		inclination = galaxy inclination, in degrees
		redshift = galaxy redshift
	
	useObservedSize : bool, optional
		If False, then a random  bar orientation will be chosen for the galaxy, and the 
		"observed" bar size in kpc will be calculated from that (using the supplied 
		inclination).
		If True, then the original S4G galaxy's observed bar size (kpc) will be used.

	getKpcSize : bool, optional
		If True, then the projected bar size in kpc is also returned
	
    Returns
    -------
	barsize_arcsec : float
		OR
	(barsize_arcsec, barsize_kpc) : (float, float)
	"""
	
	galaxyData, inclination, z = galaxyDataList[0], galaxyDataList[1], galaxyDataList[2]
	#arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z).value
	# faster approach: use spline interpolation (see comments in GetRandomGalaxy)
	arcsec_per_kpc = float(arcsecPerKpcFn(z))
	if useObservedSize:
		i_barsize = 3
		a_bar_kpc_obs = galaxyData[3]
	else:
		i_barsize = 4
		a_bar_kpc_dp = galaxyData[4]
		# compute projection using randomly oriented bar
		deltaPA_radians = random.uniform(0.0,PI_OVER_TWO)
		inc_radians = math.radians(inclination)
		a_bar_kpc_obs = projectr(deltaPA_radians, inc_radians, a_bar_kpc_dp)
	barsize_arcsec = arcsec_per_kpc * a_bar_kpc_obs
	if getKpcSize is True:
		return (barsize_arcsec, a_bar_kpc_obs)
	else:
		return barsize_arcsec


def ObserveSample( galaxySample, barSizeLimit, useObservedSizes=True, scaleBarSizes=1.0 ):
	"""
	Given a sample of galaxies produced by MakeGalaxySample(), "observe" each
	galaxy by computing its projected bar size in arc sec with ComputeBarSize()
	and then classifying it as barred if said size is >= barSizeLimit.

    Parameters
    ----------
	galaxySample : list of [galaxyData,inclination,z], as generated by MakeS4GSubsample

	barSizeLimit : float
		lower limit on observable (projected) bar semi-major axis, in arc sec
		
	useObservedSize : bool, optional
		If False, then a random  bar orientation will be chosen for the galaxy, and the 
		"observed" bar size in kpc will be calculated from that (using the supplied 
		inclination).
		If True, then the original S4G galaxy's observed bar size (kpc) will be used.

	scaleBarSizes : float, optional
		optional scaling applied to all bars

    Returns
    -------
	ii_barred : list of int
		List of indices into galaxySample for galaxies with detected bars.
	"""
	nGalaxies = len(galaxySample)
	ii_barred = [i for i in range(nGalaxies) 
			if scaleBarSizes*ComputeBarSize(galaxySample[i], useObservedSize=useObservedSizes) >= barSizeLimit]
	return ii_barred


def ObserveSampleBarSizes( galaxySample, barSizeLimit, useObservedSizes=True, scaleBarSizes=1.0 ):
	"""
	Given a sample of galaxies produced by MakeGalaxySample(), "observe" each
	galaxy by computing its projected bar size in arc sec with ComputeBarSize()
	and then classifying it as barred if said size is >= barSizeLimit.

	Returns a numpy array of observed bar sizes (kpc).

    Parameters
    ----------
	galaxySample : list of [galaxyData,inclination,z], as generated by MakeS4GSubsample

	barSizeLimit : float
		lower limit on observable (projected) bar semi-major axis, in arc sec
		
	useObservedSize : bool, optional
		If False, then a random  bar orientation will be chosen for the galaxy, and the 
		"observed" bar size in kpc will be calculated from that (using the supplied 
		inclination).
		If True, then the original S4G galaxy's observed bar size (kpc) will be used.

	scaleBarSizes : float, optional
		optional scaling applied to all bars

    Returns
    -------
	ii_barred : Numpy 1D 
		Array of observed bar sizes (sma in kpc) for galaxies classified as barred
	"""
	nGalaxies = len(galaxySample)
	obsBarSizes_kpc = []
	logMstarVals = []
	for i in range(nGalaxies):
		(obsBarSize_arcsec, obsBarSize_kpc) = ComputeBarSize(galaxySample[i], useObservedSize=useObservedSizes, getKpcSize=True)
		obsBarSize_arcsec *= scaleBarSizes
		obsBarSize_kpc *= scaleBarSizes
		print(obsBarSize_arcsec, obsBarSize_kpc)
		if obsBarSize_arcsec >= barSizeLimit:
			obsBarSizes_kpc.append(obsBarSize_kpc)
			logMstarVals.append(galaxySample[i][0][1])
			print("   OK.")
	return (np.array(logMstarVals), np.array(obsBarSizes_kpc))



def GenerateAndObserveNTimes( nSamples, dset, zRange, nGalaxies, barSizeLimit, 
							start, stop, delta, useObservedSizes=True, maxInclination=60.0,
							useWeights=True, useBmVWeights=False, useHILimit=False, 
							value="mstar", scaleSizes=1.0, randomSeed=None, debug=False ):
	"""
	Generates and observes samples of galaxies nSamples times, storing the
	median detected bar fraction for each bin of stellar mass, color, or gas
	mass ratio.
	
		nSamples = number of separate sampling+observation iterations
		dset = dataset (e.g., dest_mstar_d25)
		zRange = [z_low, z_high]
		nGalaxies = total number of galaxies in each simulated sample
		barSizeLimit = lower limit on detectable bar size, in arcsec
						(e.g., nFWHM*1.4 for SDSS)
		start, stop, delta = start, end, and delta for dataset quantity
						(e.g., for logMstar: 9.0, 11.5, 0.25)
		useObservedSize : if False, then a random inclination (and bar orientation) will 		
			be chosen for the galaxy, and the "observed" bar size in kpc will be 
			calculated from that; if H I limits are being used,	then the HyperLeda vmaxg 
			value of the original galaxy will be adjusted to account for the different 
			inclination before being used for W50 in computing the ALFALFA H I limit. 
			If True, then the original S4G galaxy's observed bar size (kpc) will be used, 
			and the original HyperLeda vmaxg value will be used for W50.
		maxInclination : maximum inclination (in degrees) for galaxies
		useWeights = if True [default], weight individual galaxies using D_max=30 Mpc
			V_max weights
		useBmVWeights = if True, weight individual galaxies using s4gdata.weight_BmVtc
			in addition to V_max weights
		useHILimit = if True, reject galaxies from samples unless their hypothetical
			H I flux would be brighter than ALFALFA 50% detection limit
		value = which galaxy value to use for bar-fraction histograms: one of 
			["mstar", "gmr", "fgas"]
		
		scaleSizes : float, optional
			scale all bar sizes by this amount
	
	Returns tuple of (bin centers, medians, medians_lowerlimit, medians_upperlimit)
		where "medians" = median f_bar values and "_lowerlimit" and "_upperlimit" are 
		lower and upper 68% confidence limits on f_bar
	"""

	if randomSeed is not None:
		random.seed(randomSeed)
	
	# indices into random-galaxy data dict	
	print("GenerateAndObserveNTimes: using data value \"{0:s}\" ...".format(value))
	if value == "mstar":
		data_index = 1
	elif value == "gmr":
		data_index = 5
	elif value == "fgas":
		data_index = 6
	weight_index = 7
	bmvWeight_index = 8
	binranges = np.arange(start, stop, delta)
	nBins = len(binranges) - 1
	allFractions = np.zeros((nBins, nSamples))
	
	if useWeights is True:
		for nn in range(nSamples):
			if debug is True:
				print("Starting sample #%d ..." % nn)
			newSamp = MakeGalaxySample(dset, zRange, nGalaxies, useObservedSizes=useObservedSizes,
										useHILimit=useHILimit, maxInclination=maxInclination)
			ii_b = ObserveSample(newSamp, barSizeLimit, useObservedSizes=useObservedSizes,
								scaleBarSizes=scaleSizes)
			ii_nonb = [i for i in range(nGalaxies) if i not in ii_b]
			ii_all = ii_b + ii_nonb
			galaxyVals = np.array([g[0][data_index] for g in newSamp])
			weights = np.array([g[0][weight_index] for g in newSamp])
			if useBmVWeights is True:
				bmvWeights = np.array([g[0][bmvWeight_index] for g in newSamp])
				weights = weights * bmvWeights

			(n_b, bin_edges) = np.histogram(galaxyVals[ii_b], binranges, weights=weights[ii_b])
			(n_all, bin_edges) = np.histogram(galaxyVals[ii_all], binranges, weights=weights[ii_all])
			
			values_valid_unweighted = np.array([ galaxyVals[i] for i in ii_all if weights[i] > 0 ])
			(n_all_unwt, junk) = np.histogram(values_valid_unweighted, bins=binranges)
			scaleFactors = n_all / n_all_unwt
			n_b_normalized = n_b / scaleFactors
			# we calculate fractions (and confidence limits) using renormalized numbers, so that
			# the total in each bin is the total number of galaxy in each bin (rather than the
			# *weighted* total)
			(frac_b, delta_low, delta_high) = s4gutils.Binomial(n_b_normalized, n_all_unwt)
			for i in range(nBins):
				if n_all_unwt[i] != 0:
					allFractions[i][nn] = frac_b[i]
				else:
					allFractions[i][nn] = np.nan
			if debug is True:
				print("Finished with sample #%d." % nn)

	else:
		for nn in range(nSamples):
			newSamp = MakeGalaxySample(dset, zRange, nGalaxies, useObservedSizes=useObservedSizes,
										useHILimit=useHILimit, maxInclination=maxInclination)
			ii_b = ObserveSample(newSamp, barSizeLimit, useObservedSizes=useObservedSizes,
								scaleBarSizes=scaleSizes)
			ii_nonb = [i for i in range(nGalaxies) if i not in ii_b]

			galaxyVals = [g[0][data_index] for g in newSamp]
			n1,bin_edges = np.histogram(np.array(galaxyVals)[ii_b], bins=binranges)
			n2,bin_edges = np.histogram(np.array(galaxyVals)[ii_nonb], bins=binranges)
			fractions = []
			for i in range(nBins):
				if n1[i] + n2[i] != 0:
					f,f_low,f_high = s4gutils.Binomial(n1[i], n1[i] + n2[i])
				else:
					f = np.nan
				allFractions[i][nn] = f
	
	finalMedians = np.zeros(nBins)
	finalMedians_low = np.zeros(nBins)
	finalMedians_high = np.zeros(nBins)
	for j in range(nBins):
		fractionsForThisBin = allFractions[j]
		i_real = [i for i in range(nSamples) if np.isfinite(fractionsForThisBin[i] )]
		median = np.median(fractionsForThisBin[i_real])
		finalMedians[j] = median
		if nSamples > 5:
			m_low,m_high = s4gutils.ConfidenceInterval(fractionsForThisBin[i_real])
		else:
			m_low = m_high = median
		finalMedians_low[j] = m_low
		finalMedians_high[j] = m_high

	# midpoints of the bins:
	midvals = np.zeros(nBins)
	for i in range(nBins):
		midvals[i] = 0.5*(bin_edges[i] + bin_edges[i + 1])

	return midvals, finalMedians, finalMedians_low, finalMedians_high

