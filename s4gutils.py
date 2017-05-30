# Miscellaneous code for analysis of S4G bar fractions

import copy
import math
import random
import numpy as np

random.seed()


# lower and upper bounds of 68.3% confidence interval:
ONESIGMA_LOWER = 0.1585
ONESIGMA_UPPER = 0.8415



def Read2ColumnProfile( fname ):
	"""Read in the (first) two columns from a simple text file where the columns
	are separated by whitespace and lines beginning with '#' are ignored.
	
	Returns tuple of (x, y), where x and y are numpy 1D arrays corresponding to
	the first and second column
	"""
	dlines = [line for line in open(fname) if len(line) > 1 and line[0] != "#"]
	x = [float(line.split()[0]) for line in dlines]
	y = [float(line.split()[1]) for line in dlines]
	return np.array(x), np.array(y)



def dtomm( distanceMpc ):
	"""Converts distance in Mpc to distance modulus (M - m, in magnitudes)
	"""
	five_logD = 5.0 * np.log10(distanceMpc)
	return (25.0 + five_logD)


def HIMassToFlux( M_HI, dist_Mpc ):
	"""Converts H I mass (in solar masses) to equivalent H I flux (in Jy km/s)
	based on distance in Mpc.  Equation originally from Giovanelli & Haynes
	(1988, in Galactic and extragalactic radio astronomy (2nd edition), p.522),
	based on Roberts (1975, n A. Sandage, M. Sandage, and J. Kristian (eds.), 
	Galaxies and the Universe. Chicago:	University of Chicago Press; p.	309).
	"""
	return M_HI / (2.356e5 * dist_Mpc**2)


def GetRadialSampleFromSphere( rMin, rMax ):
	"""Get radius sample from spherical Euclidean volume (or spherical shell) using 
	the discarding method: generate a random point within a cube of half-width = rMax; 
	discard and re-generate if radius to that point is outside [rMin, rMax]
	"""
	rMin2 = rMin*rMin
	rMax2 = rMax*rMax
	done = False
	while not done:
		x = random.uniform(-rMax, rMax)
		y = random.uniform(-rMax, rMax)
		z = random.uniform(-rMax, rMax)
		r2 = x*x + y*y + z*z
		if (r2 >= rMin2) and (r2 <= rMax2):
			done = True
	return math.sqrt(r2)	


def ConfidenceInterval( vect ):
	
	nVals = len(vect)
	lower_ind = int(round(ONESIGMA_LOWER*nVals)) - 1
	upper_ind = int(round(ONESIGMA_UPPER*nVals))
	vect_sorted = copy.copy(vect)
	vect_sorted.sort()
	return (vect_sorted[lower_ind], vect_sorted[upper_ind])


def Binomial( n, n_tot, nsigma=1.0, conf_level=None, method="wilson" ):
	"""Computes fraction (aka frequency or rate) of occurances p = (n/n_tot).
	Also computes the lower and upper confidence limits using either the 
	Wilson (1927) or Agresti & Coull (1998) method (method="wilson" or method="agresti");
	default is to use Wilson method.
	Default is to calculate 68.26895% confidence limits (i.e., 1-sigma in the
	Gaussian approximation).
	
	Returns tuple of (p, sigma_minus, sigma_plus).
	"""
	
	p = (1.0 * n) / n_tot
	q = 1.0 - p
	
	if (conf_level is not None):
		print("Alternate values of nsigma or conf_limit not yet supported!")
		alpha = 1.0 - conf_level
		# R code would be the following:
		#z_alpha = qnorm(1.0 - alpha/2.0)
		return None
	else:
		z_alpha = nsigma   # e.g., z_alpha = nsigma = 1.0 for 68.26895% conf. limits
	
	if (method == "wald"):
		# Wald (aka asymptotic) method -- don't use except for testing purposes!
		sigma_minus = sigma_plus = z_alpha * np.sqrt(p*q/n_tot)
	else:
		z_alpha2 = z_alpha**2
		n_tot_mod = n_tot + z_alpha2
		p_mod = (n + 0.5*z_alpha2) / n_tot_mod
		if (method == "wilson"):
			# Wilson (1927) method
			sigma_mod = np.sqrt(z_alpha2 * n_tot * (p*q + z_alpha2/(4.0*n_tot))) / n_tot_mod
		elif (method == "agresti"):
			# Agresti=Coull method
			sigma_mod = np.sqrt(z_alpha2 * p_mod * (1.0 - p_mod) / n_tot_mod)
		else:
			print("ERROR: method \"%s\" not implemented in Binomial!" % method)
			return None
		p_upper = p_mod + sigma_mod
		p_lower = p_mod - sigma_mod
		sigma_minus = p - p_lower
		sigma_plus = p_upper - p
	
	return (p, sigma_minus, sigma_plus)



# Various functions for estimating stellar masses from absolute magnitudes and color-based
# M/L values

def magratio( mag1, mag2, mag1_err=None, mag2_err=None ):
	"""Calculates luminosity ratio given two magnitudes; optionally 
	computes the error on the ratio using standard error propagation 
	(only if at least one of the errors is given; if only one is given, 
	the other is assumed to be = 0)."""
	
	diff = mag1 - mag2
	lumRatio = 10.0**(-diff*0.4)
	if (mag1_err is None) and (mag2_err is None):
		return lumRatio
	else:
		if (mag1_err is None):
			mag1_err = 0.0
		elif (mag2_err is None):
			mag2_err = 0.0
		p1 = ln10*lumRatio*(-0.4) * mag1_err
		p2 = ln10*lumRatio*(0.4) * mag2_err
		lumRatio_err = math.sqrt(p1**2 + p2**2)
		return (lumRatio, lumRatio_err)


# Solar absolute magnitudes from Table 1.2 of Sparke & Gallagher for U, from
# Bell & de Jong (2001) for Johnson B and V, Kron-Cousins R and I, and
# Johnson J, H, and K original sources: Cox 2000; Bessel 1979; Worthey 1994).
# Solar absolute magnitudes for SDSS ugriz (AB mag) are from Bell et al. (2003 ApJS 149: 289).
# Thus, filters are standard Johnson-Cousins UBVRIJHK + SDSS ugriz, with 
# K = standard ("broad") K, *not* K_s.
# K_s value taken from Kormendy+10: "The 2MASS survey uses a Ks bandpass whose 
# effective wavelength is ~ 2.16 microns (Carpenter 2001; Bessell 2005). Following 
# the above papers, we assume that Ks = K - 0.044. Then the Ks-band absolute 
# magnitude of the Sun is 3.29."

solarAbsMag = { "U": 5.62, "B": 5.47, "V": 4.82, "R": 4.46, "I": 4.14,
				"J": 3.70, "H": 3.37, "K": 3.33, "u": 6.41, "g": 5.15,
				"r": 4.67, "i": 4.56, "z": 4.53, "K_s": 3.29 }

def solarL( mag, filterName, mag_err=None,  Ks=False ):
	"""Takes an absolute magnitude and the corresponding bandpass, and 
	returns corresponding solar luminosities.  Uses solar absolute magnitudes 
	from Table 1.2 of Sparke & Gallagher for U and from Bell & de Jong (2001, 
	ApJ 550: 212) for Johnson B and V, Kron-Cousins R and I, and Johnson J, H, 
	and K (original sources: Cox 2000; Bessel 1979; Worthey 1994).  Solar absolute
	magnitudes for SDSS ugriz are from Bell et al. (ApJS 149: 289).
	Thus, filters are standard Johnson-Cousins UBVRIJHK + SDSS ugriz, with 
	K = standard ("broad") K, *not* K_s.
	
	If Ks = True, then we substitute K_s for K
	
	If mag_err is given, then the error on the luminosity is also computed,
	using standard error propagattion [done in magratio() function], assuming
	the solar absolute magnitude has no error."""
	
	if (Ks is True) and (filterName == "K"):
		filterName = "K_s"
	try:
		m_Sun = solarAbsMag[filterName]
	except KeyError as e:
		print("   solarL: unrecognized filter \"%s\"!" % filterName)
		return 0

	if (mag_err is None):
		return magratio(mag, m_Sun)
	else:
		return magratio(mag, m_Sun, mag_err)


def MassToLight( band, colorType, color, err=None, mode="Bell" ):
	"""Calculates stellar mass-to-light ratio for a specified band (one of
	BVRIJHK), given a color index.
	
	band = the desired band for the mass-to-light ratio (one of Johnson-Cousins
	BVRIJHK [Vega magnitudes] or SDSS ugriz [AB magnitudes]).
	
	colorType="B-V", "B-R", "V-I", "V-J", "V-H", or "V-K" for Johnson-Cousins
	colors, or "u-g", "u-r", "u-i", "u-z", "g-r", "g-i", "g-z", "r-i", or "r-z"
	for SDSS colors.
	
	color = value of the specified color index.
	
	Returns M/L (mass in solar masses / luminosity in solar luminosities).
	If err != None, then the error in M/L is also returned (using the
	dex value provided in err, which should be 0.1--0.2).
	
	Based on Table 1 of Bell & de Jong (2001, ApJ 550: 212) and
	Table 7 of Bell et al. (2003, ApJS 149: 289); note that B-V and B-R
	values use Bell+2003, but other optical colors use Bell & de Jong.
	
	Alternately, the fits in Zibetti+2009 (Table B1) can be used instead,
	by specifying mode="Zibetti"
	"""
	
	
	# dictionaries indexed by colorType, holding sub-dictionaries with
	# corresponding coefficients, indexed by band
	
	coefficients_B = {}
	# M/L ratios for Johnson-Cousin bands, from Bell et al. (2003) for B-V
	# and B-R, and from Bell & de Jong (2001) for other colors:
	coefficients_B['B-V'] = {'B': [-0.942, 1.737], 'V': [-0.628, 1.305],
			'R': [-0.520, 1.094], 'I': [-0.399, 0.824], 'J': [-0.261, 0.433],
			'H': [-0.209, 0.210], 'K': [-0.206, 0.135]}
	coefficients_B['B-R'] = {'B': [-1.224, 1.251], 'V': [-0.916, 0.976],
			'R': [-0.523, 0.683], 'I': [-0.405, 0.518], 'J': [-0.289, 0.297],
			'H': [-0.262, 0.180], 'K': [-0.264, 0.138]}
	coefficients_B['V-I'] = {'B': [-1.919, 2.214], 'V': [-1.476, 1.747],
			'R': [-1.314, 1.528], 'I': [-1.204, 1.347], 'J': [-1.040, 0.987],
			'H': [-1.030, 0.870], 'K': [-1.027, 0.800]}
	coefficients_B['V-J'] = {'B': [-1.903, 1.138], 'V': [-1.477, 0.905],
			'R': [-1.319, 0.794], 'I': [-1.209, 0.700], 'J': [-1.029, 0.505],
			'H': [-1.014, 0.442], 'K': [-1.005, 0.402]}
	coefficients_B['V-H'] = {'B': [-2.181, 0.978], 'V': [-1.700, 0.779],
			'R': [-1.515, 0.684], 'I': [-1.383, 0.603], 'J': [-1.151, 0.434],
			'H': [-1.120, 0.379], 'K': [-1.100, 0.345]}
	coefficients_B['V-K'] = {'B': [-2.156, 0.895], 'V': [-1.683, 0.714],
			'R': [-1.501, 0.627], 'I': [-1.370, 0.553], 'J': [-1.139, 0.396],
			'H': [-1.108, 0.346], 'K': [-1.087, 0.314]}
	# M/L ratios for SDSS + Johnson-Cousins NIR bands, from Bell et al. 2003:
	coefficients_B['u-g'] = {'g': [-0.221, 0.485], 'r': [-0.099, 0.345],
			'i': [-0.053, 0.268], 'z': [-0.105, 0.226], 'J': [-0.128, 0.169],
			'H': [-0.209, 0.133], 'K': [-0.260, 0.123]}
	coefficients_B['u-r'] = {'g': [-0.390, 0.417], 'r': [-0.223, 0.299],
			'i': [-0.151, 0.233], 'z': [-0.178, 0.192], 'J': [-0.172, 0.138],
			'H': [-0.237, 0.104], 'K': [-0.273, 0.091]}
	coefficients_B['u-i'] = {'g': [-0.375, 0.359], 'r': [-0.212, 0.257],
			'i': [-0.144, 0.201], 'z': [-0.171, 0.165], 'J': [-0.169, 0.119],
			'H': [-0.233, 0.090], 'K': [-0.267, 0.077]}
	coefficients_B['u-z'] = {'g': [-0.400, 0.332], 'r': [-0.232, 0.239],
			'i': [-0.161, 0.187], 'z': [-0.179, 0.151], 'J': [-0.163, 0.105],
			'H': [-0.205, 0.071], 'K': [-0.232, 0.056]}
	coefficients_B['g-r'] = {'g': [-0.499, 1.519], 'r': [-0.306, 1.097],
			'i': [-0.222, 0.864], 'z': [-0.223, 0.689], 'J': [-0.172, 0.444],
			'H': [-0.189, 0.266], 'K': [-0.209, 0.197]}
	coefficients_B['g-i'] = {'g': [-0.379, 0.914], 'r': [-0.220, 0.661],
			'i': [-0.152, 0.518], 'z': [-0.175, 0.421], 'J': [-0.153, 0.283],
			'H': [-0.186, 0.179], 'K': [-0.211, 0.137]}
	coefficients_B['g-z'] = {'g': [-0.367, 0.698], 'r': [-0.215, 0.508],
			'i': [-0.153, 0.402], 'z': [-0.171, 0.322], 'J': [-0.097, 0.175],
			'H': [-0.117, 0.083], 'K': [-0.138, 0.047]}
	coefficients_B['r-i'] = {'g': [-0.106, 1.982], 'r': [-0.022, 1.431],
			'i': [0.006, 1.114], 'z': [-0.052, 0.923], 'J': [-0.079, 0.650],
			'H': [-0.148, 0.437], 'K': [-0.186, 0.349]}
	coefficients_B['r-z'] = {'g': [-0.124, 1.067], 'r': [-0.041, 0.780],
			'i': [-0.018, 0.623], 'z': [-0.041, 0.463], 'J': [-0.011, 0.224],
			'H': [-0.059, 0.076], 'K': [-0.092, 0.019]}

	coefficients_Z = {}
	# M/L ratios for SDSS colors + SDSS or JHK bands, from Zibetti+2009:
	coefficients_Z['u-g'] = {'g': [-1.628, 1.360], 'r': [-1.319, 1.093],
			'i': [-1.277, 0.980], 'z': [-1.315, 0.913], 'J': [-1.350, 0.804],
			'H': [-1.467, 0.750], 'K': [-1.578, 0.739]}
	coefficients_Z['u-r'] = {'g': [-1.427, 0.835], 'r': [-1.157, 0.672],
			'i': [-1.130, 0.602], 'z': [-1.181, 0.561], 'J': [-1.235, 0.495],
			'H': [-1.361, 0.463], 'K': [-1.471, 0.455]}
	coefficients_Z['u-i'] = {'g': [-1.468, 0.716], 'r': [-1.193, 0.577],
			'i': [-1.160, 0.517], 'z': [-1.206, 0.481], 'J': [-1.256, 0.422],
			'H': [-1.374, 0.393], 'K': [-1.477, 0.384]}
	coefficients_Z['u-z'] = {'g': [-1.559, 0.658], 'r': [-1.268, 0.531],
			'i': [-1.225, 0.474], 'z': [-1.260, 0.439], 'J': [-1.297, 0.383],
			'H': [-1.407, 0.355], 'K': [-1.501, 0.344]}
	coefficients_Z['g-r'] = {'g': [-1.030, 2.053], 'r': [-0.840, 1.654],
			'i': [-0.845, 1.481], 'z': [-0.914, 1.382], 'J': [-1.007, 1.225],
			'H': [-1.147, 1.144], 'K': [-1.257, 1.119]}
	coefficients_Z['g-i'] = {'g': [-1.197, 1.431], 'r': [-0.977, 1.157],
			'i': [-0.963, 1.032], 'z': [-1.019, 0.955], 'J': [-1.098, 0.844],
			'H': [-1.222, 0.780], 'K': [-1.321, 0.754]}
	coefficients_Z['g-z'] = {'g': [-1.370, 1.190], 'r': [-1.122, 0.965],
			'i': [-1.089, 0.858], 'z': [-1.129, 0.791], 'J': [-1.183, 0.689],
			'H': [-1.291, 0.632], 'K': [-1.379, 0.604]}
	coefficients_Z['r-i'] = {'g': [-1.405, 4.280], 'r': [-1.155, 3.482],
			'i': [-1.114, 3.087], 'z': [-1.145, 2.828], 'J': [-1.199, 2.467],
			'H': [-1.296, 2.234], 'K': [-1.371, 2.109]}
	coefficients_Z['r-z'] = {'g': [-1.576, 2.490], 'r': [-1.298, 2.032],
			'i': [-1.238, 1.797], 'z': [-1.250, 1.635], 'J': [-1.271, 1.398],
			'H': [-1.347, 1.247], 'K': [-1.405, 1.157]}
	# M/L ratios for Johnson-Cousin colors and bands
	coefficients_Z['B-V'] = {'B': [-1.330, 2.237], 'V': [-1.075, 1.837],
			'R': [-0.989, 1.620], 'I': [-1.003, 1.475], 'J': [-1.135, 1.267],
			'H': [-1.274, 1.190], 'K': [-1.390, 1.176]}
	coefficients_Z['B-R'] = {'B': [-1.614, 1.466], 'V': [-1.314, 1.208],
			'R': [-1.200, 1.066], 'I': [-1.192, 0.967], 'J': [-1.289, 0.822],
			'H': [-1.410, 0.768], 'K': [-1.513, 0.750]}
	
	if (mode == "Bell"):
		coefficients = coefficients_B
	elif (mode == "Zibetti"):
		coefficients = coefficients_Z
	else:
		print("\n*** bad mode (\"%s\") selected in MassToLight! *** \n" % mode)
		return None
	try:
		a = coefficients[colorType][band][0]
		b = coefficients[colorType][band][1]
	except KeyError as err:
		txt = "\n*** %s is not an allowed color or band (or color/band combination) for %s et al. mass ratios! ***\n" % (err, mode)
		txt += "    (MassToLight called with colorType = '%s', band = '%s')\n" % (colorType, band)
		print(txt)
		return None
	
	logML = a + b*color
	
	if err is None:
		return 10**logML
	else:
		MtoL = 10**logML
		sigma_MtoL = ln10*err*MtoL
		return (MtoL, sigma_MtoL)


def AbsMagToStellarMass( absMag, band, colorType="B-V", color=None, mag_err=None, 
						MtoL_err=0.1, mode="Bell", MtoL=None ):
	"""Calculates a galaxy's stellar mass (in solar masses) given as input an
	absolute magnitude, the corresponding filter (one of BVRIJK), the galaxy
	color type (e.g., "B-V", "B-R", "V-I", "V-J", "V-H", "V-K"; SDSS colors
	such as "u-g", "u-r", "u-i", "u-z", "g-r", "g-i", "g0z", etc., can also
	be used), and the color index.
	
	If mag_err is defined, then error propagation is used and
	(M_stellar, err_M_stellar) is returned.  Note that if mag_err=0.0,
	errors for the M/L ratio will still be propagated.  The default error
	for M/L is 0.1 dex, but this can be changed with the MtoL_err keyword;
	if so, it must be in *log* units.
	
	Uses M/L ratios from Table 1 of Bell & de Jong [see MassToLight() above]
	and solar-luminosity conversion from Table 1.2 of Sparke & Gallagher
	[see solarL() above]; to use the M/L ratios from Zibetti+2009, use
	mode="Zibetti".
	
	Alternatively, a user-supplied M/L value can be given with the MtoL
	keyword.
	"""
	
	if (mag_err is None):
		if MtoL is None:
			MtoL = MassToLight(band, colorType, color, mode=mode)
			if MtoL is None:
				return None
		solarLum = solarL(absMag, band)
		return MtoL * solarLum
	else:
		if MtoL is None:
			(MtoL, err_MtoL) = MassToLight(band, colorType, color, err=MtoL_err, mode=mode)
			if MtoL is None:
				return (None, None)
		(solarLum, err_solarLum) = solarL(absMag, band, mag_err)
		M_stellar = MtoL * solarLum
		p1 = err_MtoL/MtoL
		if (err_solarLum > 0.0):
			p2 = err_solarLum/solarLum
		else:
			p2 = 0.0
		err_M_stellar = math.sqrt(p1**2 + p2**2) * M_stellar
		return (M_stellar, err_M_stellar)

